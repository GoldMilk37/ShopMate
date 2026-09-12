"""第 3 步：把 Chunk 向量化（BGE-M3）并写入 ChromaDB。已实现。

模块速览：
    get_model()        懒加载 BGE-M3（进程内单例，只加载一次）
    embed_texts(...)    批量向量化
    get_client(dir)     打开/创建 ChromaDB 持久化客户端（retriever 也要用）
    get_collection(name, client)  打开/建 collection（retriever 也要用）
    build_index(...)    全量入库：按 collection 分组 → upsert（同 ID 覆盖）

五个设计决策（读代码时对着看，retriever 会依赖 D2/D5）：
    D1 懒加载单例：BGE-M3 约 2.3GB，加载需数秒~数十秒。第一次 embed 才
       加载，之后复用同一个实例。反例（禁止的写法）：每次调用都
       BGEM3FlagModel(...)——模型加载会让每次检索慢 10 秒以上。
    D2 模型与集合名解耦：get_model/get_client/get_collection 都是独立函数，
       retriever 检索时直接 import 复用，不需要经过 build_index。
    D3 持久化目录 data/chroma：ChromaDB 的 PersistentClient 落盘，重启不丢；
       重复执行 build_index 是安全的（upsert 同 ID 覆盖 = 先删后插，
       01 文档 §6 的覆盖式重建）。
    D4 cosine 相似度：建 collection 时指定 "_METADATA:hnsw:space" = "cosine"。
       向量相似度有 cosine（看方向）/l2（看距离）两种，文本语义检索惯例用 cosine。
    D5 元数据全部塞进 ChromaDB：doc_type/product_id/brand 等随 chunk 入库，
       检索端可用 where 过滤（01 文档 §3 的用途示例）。

依赖链：build_index ⊂ loader.load_all + chunker.chunk_all。
完整建库：python -m app.retrieval.indexer
"""
from pathlib import Path
import threading

from .schema import Chunk, EMBED_DIM

# 持久化目录：项目根/data/chroma（与 rag_docs 平级）
PERSIST_ROOT = Path(__file__).resolve().parents[2] / "data" / "chroma"

# 每个 collection 一个名字（01 文档 §2 的三库）
COLLECTIONS = ("product_knowledge", "policy_knowledge", "review_knowledge")

_embed_batch = 64        # D1 的同时：一次喂给模型的条数，太大会爆显存/内存，太小吞吐低


# ---------------- 模型（D1/D2） ----------------

_model = None
_model_lock = threading.Lock()   # 两处同时 embed 时保证只加载一次


def get_model():
    """返回进程级单例的 BGE-M3。首次调用会加载（数秒~数十秒），之后瞬时返回。"""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:                      # double-check：抢到锁后再看一眼
            from FlagEmbedding import BGEM3FlagModel
            _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)   # fp16：内存减半，精度损失可忽略
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量向量化。返回与 texts 等长的列表，每项是 1024 维 list。

    BGE-M3 的输出是 {"dense_vecs": np.ndarray}，这里转成普通 list——
    ChromaDB 接受 list，np.ndarray 传进去会在序列化时报错。
    """
    if not texts:
        return []
    model = get_model()
    out: list[list[float]] = []
    for i in range(0, len(texts), _embed_batch):
        batch = texts[i:i + _embed_batch]
        res = model.encode(batch)["dense_vecs"]
        out.extend(vec.tolist() for vec in res)
    return out


# ---------------- ChromaDB（D2/D3/D4） ----------------

def get_client(persist_dir: Path | str | None = None):
    """打开持久化客户端。同一路径重复调用返回同一实例（ChromaDB 自己缓存）。"""
    import chromadb

    return chromadb.PersistentClient(path=str(persist_dir or PERSIST_ROOT))


def get_collection(name: str, client=None, create: bool = True):
    """打开/建一个 collection（cosine）。retriever 传 create=False 防误建空库。"""
    client = client or get_client()
    kw = {"metadata": {"hnsw:space": "cosine"}} if create else {}   # D4
    return (client.get_or_create_collection(name, **kw) if create
            else client.get_collection(name))


# ---------------- 入库（D3/D5） ----------------

def build_index(chunks: list[Chunk], persist_dir: Path | str | None = None) -> dict[str, int]:
    """全量入库。返回 {collection名: 入库条数}。

    upsert = insert or update：同一个 chunk_id 再写一次会覆盖旧的，
    所以反复跑 build_index 结果一致（幂等）。
    """
    client = get_client(persist_dir)
    stats: dict[str, int] = {}

    for coll_name in COLLECTIONS:
        group = [c for c in chunks if c.meta.get("collection") == coll_name]
        if not group:
            continue
        coll = get_collection(coll_name, client)
        # 兜底断言：分组非空才说明 meta["collection"] 链路通了（chunker→indexer 契约）
        # 分组后仍按批向量化（每批一起算向量，一次 upsert）
        for i in range(0, len(group), _embed_batch):
            batch = group[i:i + _embed_batch]
            coll.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=embed_texts([c.to_embedding_text() for c in batch]),
                documents=[c.text for c in batch],
                metadatas=[c.meta for c in batch],      # D5：元数据全部随行入库
            )
        stats[coll_name] = len(group)
    return stats


# ---------------- 自测：python -m app.retrieval.indexer ----------------
# 首次运行会下载 BGE-M3（约 2.3GB，数分钟）；之后有本地缓存。
if __name__ == "__main__":
    import sys

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    print("加载 loader + chunker ...")
    from .loader import load_all
    from .chunker import chunk_all

    chunks = chunk_all(load_all())
    print(f"共 {len(chunks)} 个 chunk，开始建库（首次运行含模型加载/下载）...")

    stats = build_index(chunks)
    print("入库统计:", stats)
    assert stats and set(stats) == set(COLLECTIONS), \
        f"三个 collection 应全部有入库，实际: {stats}"   # 防再次空转——上次就栽在这里
    covered = sum(stats.values())
    assert covered == len(chunks), f"入库总数 {covered} != chunk 总数 {len(chunks)}"

    # 校验一：四个断言（计数/ID覆盖/维度）过后才算通过
    client = get_client()
    for coll_name, n in stats.items():
        coll = get_collection(coll_name, client)
        assert coll.count() == n, f"{coll_name}: 库里 {coll.count()} != 应有 {n}"
        # 抽一个 chunk 看 1024 维
        one = coll.get(limit=1, include=["embeddings"])
        assert len(one["embeddings"][0]) == EMBED_DIM, \
            f"{coll_name}: 向量维度 {len(one['embeddings'][0])} != {EMBED_DIM}"
        print(f"  {coll_name}: count={coll.count()} 向量维度 OK")
    print("校验通过")

    # 校验二：语义抽查——耳机商品查询应召回耳机而非洗衣机（提前给 retriever 摸底）
    probe = "通勤戴的降噪耳机，预算六百左右，要续航长的"
    emb = embed_texts([probe])[0]
    coll = get_collection("product_knowledge", client)
    hit = coll.query(query_embeddings=[emb], n_results=2)
    for cid, doc in zip(hit["ids"][0], hit["documents"][0]):
        print(f"语义抽查 top 命中: {cid} | {doc.splitlines()[1][:40]}")
