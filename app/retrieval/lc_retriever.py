"""LangChain 版检索器：retriever 的同任务重写——只换向量路的实现。已实现。

用法：
    SHOPMATE_RETRIEVER=lc python -m app.retrieval.eval   # 评测跑 LangChain 版
    python -m app.retrieval.lc_retriever                 # 离线自测（双版对照）

与手写版 retriever 的映射（"同一任务、两种实现"对照表）：

    手写版                                 LangChain 版
    ─────────────────────────────         ─────────────────────────────────
    indexer.embed_texts 直接调用            BGEEmbeddings（langchain Embeddings
                                          接口的十几行包装，内部仍委托它）
    indexer.get_collection + coll.query    langchain_chroma.Chroma 向量库对象
                                         （filter=where 透传 Chroma）
    自写 cosine 距离门槛/RRF/词法门/加权     RRF/门/加权全部复用本包 _fuse_and_format
                                         ——融合语义必须逐字节一致，评测才可比

两个设计决策：
    D1 只换向量路，融合段一行不重写：search_with_product_focus 的"重排+补充"、
       词法置信门、doc_type 加权、阈值兜底是 hit@5 96%→100% 的来源，LangChain
       没有对应物；EnsembleRetriever 只对齐"RRF 融合"这一层。抽出的
       retriever._fuse_and_format 是两版共用的唯一事实源。
    D2 langchain_chroma 的 score 语义要在运行时钉死：similarity_search_with_score
       返回的是 Chroma 原始**距离**（cosine 距离，越小越相关），与手写版
       SEMANTIC_MAX_DIST 的比较口径一致。自测第 2 节用"同一 query 双版距离
       逐项对齐"的断言把这件事钉住——若未来 langchain_chroma 改成返回相似度，
       这里会当场炸，而不是悄悄放错候选进融合。

依赖关系：langchain_chroma + langchain_core（Embeddings 接口）；模型加载/离线
缓存/Chroma 客户端全部复用 indexer，BM25 语料、RRF、门、阈值全部复用 retriever。
"""
import threading

from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from .indexer import PERSIST_ROOT, embed_texts
from .retriever import (_bm25_route, _fuse_and_format, _validate_where, TOP_K)
from .schema import SEMANTIC_MAX_DIST


# ---------------- LangChain Embeddings 接口适配（D1 的十几行） ----------------

class BGEEmbeddings(Embeddings):
    """把 indexer.embed_texts 适配成 LangChain 的 Embeddings 协议。

    文档与查询必须出自同一个模型（同一向量空间），所以包装而不是换实现——
    本类存在的意义是让 langchain_chroma 以为背后是个标准 Embeddings 服务。
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return embed_texts([text])[0]


# ---------------- 向量库对象（按 collection 缓存，同 get_model 单例套路） ----------------

_stores: dict[str, Chroma] = {}
_stores_lock = threading.Lock()


def _lc_store(collection: str) -> Chroma:
    """打开（不创建）一个 LangChain Chroma 向量库实例。

    语料已由 indexer 建好（cosine、含 meta.chunk_id），这里只读不建：
    传 embed 构件是为了让 query 时自动用 BGE-M3 编码查询句。
    """
    if collection in _stores:
        return _stores[collection]
    with _stores_lock:
        if collection not in _stores:                  # double-check，同 get_model
            import chromadb
            raw = chromadb.PersistentClient(path=str(PERSIST_ROOT))
            try:
                raw.get_collection(collection)         # 建库前不许查询空库
            except Exception as e:
                raise RuntimeError(
                    f"collection {collection!r} 不存在——先跑 python -m app.retrieval.indexer 建库") from e
            _stores[collection] = Chroma(
                collection_name=collection,
                embedding_function=BGEEmbeddings(),
                persist_directory=str(PERSIST_ROOT),
            )
    return _stores[collection]


# ---------------- 向量路（LangChain 版） ----------------

def _vector_route(query: str, collection: str, top_k: int,
                  where: dict | None = None) -> list[tuple[str, int]]:
    """与手写版 _vector_route 同签名同语义：[(chunk_id, rank)]，含距离门槛重排。

    chunk_id 从 Document.metadata 里取（chunker 已把它塞进 meta 随行入库）；
    filter 直接透传给 Chroma 的 where，相等/$in 两种算子由 _validate_where 把关。
    """
    store = _lc_store(collection)
    pairs = store.similarity_search_with_score(query, k=top_k, filter=where or None)
    out: list[tuple[str, int]] = []
    for doc, dist in pairs:
        if dist > SEMANTIC_MAX_DIST:                   # D2：距离口径，越小越好
            continue
        cid = (doc.metadata or {}).get("chunk_id", "")
        if not cid:
            raise RuntimeError(f"召回块缺 meta.chunk_id（需重建索引）: {doc.metadata}")
        out.append((cid, len(out) + 1))                # 过滤后重排号，排名不带洞
    return out


# ---------------- 对外入口（与手写版同签名，graph/eval 无感切换） ----------------

def search(query: str, collection: str = "product_knowledge",
           top_k: int = TOP_K, where: dict | None = None) -> list[dict]:
    """LangChain 版混合检索。返回形状与手写版逐字段一致：
    [{"chunk_id", "text", "score", "meta"}]，融合段走共用 _fuse_and_format。"""
    _validate_where(where)
    vector_ranks = _vector_route(query, collection, top_k, where)
    bm25_ranks = _bm25_route(query, collection, top_k, where)
    return _fuse_and_format(query, collection, top_k, where, vector_ranks, bm25_ranks)


def search_with_product_focus(query: str, collection: str, product_id: str,
                              top_k: int = TOP_K) -> list[dict]:
    """话题商品锚点检索（重排+补充）。逻辑与手写版相同，只是底层 search 换本版。

    刻意不复用手写版函数——它内部写死了调用手写 search，混用会让"哪一半
    结果出自哪个实现"说不清。对照实验必须一个实现跑到底。
    """
    from .retriever import search as _hand_search
    if not product_id:
        return search(query, collection, top_k)
    base = search(query, collection, top_k)
    own = [h for h in base if (h["meta"].get("product_id") or "") == product_id]
    others = [h for h in base if (h["meta"].get("product_id") or "") != product_id]
    if base and len(own) * 2 >= top_k:                 # 已占住前排 → 纯重排
        return (own + others)[:top_k]
    extra = search(query, collection, top_k, where={"product_id": product_id})
    seen = {h["chunk_id"] for h in own}
    extra = [h for h in extra if h["chunk_id"] not in seen]
    return (own + extra + others)[:top_k]


# ---------------- 自测：python -m app.retrieval.lc_retriever ----------------
if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    from .retriever import search as hand_search
    from .retriever import _vector_route as hand_vector_route

    print("加载两版检索器（BGE-M3 首次加载十几秒）...")
    cases = [
        ("SKU-10001 现在什么价", "product_knowledge", None),   # BM25 主场
        ("通勤戴的降噪耳机，预算六百左右", "product_knowledge", None),  # 向量主场
        ("买了五天想退货，运费谁出", "policy_knowledge", None),
        ("这款耳机有什么缺点", "review_knowledge", "SKU-10001"),  # 焦点检索
        ("量子涨落对股市的影响", "product_knowledge", None),   # 必须双双返回 []
    ]

    # ---- 1. 双版对照：同 query 同库，结果必须逐项一致 ----
    from .retriever import search_with_product_focus as hand_focus_fn
    for q, coll, focus in cases:
        lc = (search_with_product_focus(q, coll, focus) if focus
              else search(q, coll))
        hand = (hand_focus_fn(q, coll, focus) if focus else hand_search(q, coll))
        ids_lc = [h["chunk_id"] for h in lc]
        ids_hand = [h["chunk_id"] for h in hand]
        assert ids_lc == ids_hand, f"排序分叉 {q!r}:\n  lc  ={ids_lc}\n  hand={ids_hand}"
        assert [h["score"] for h in lc] == [h["score"] for h in hand], q
        print(f"1 双版一致: [{coll}] {q!r} → {ids_lc}")

    # ---- 2. D2 钉死 score 语义：lc 的 score 必须等于 Chroma 原始 cosine 距离 ----
    q = "通勤戴的降噪耳机，预算六百左右"
    store = _lc_store("product_knowledge")
    lc_dists = sorted(d for _, d in store.similarity_search_with_score(q, k=5))
    coll = store._collection if hasattr(store, "_collection") else None
    emb = embed_texts([q])[0]
    raw = coll.query(query_embeddings=[emb], n_results=5, include=["distances"])
    raw_dists = sorted(raw["distances"][0])
    assert all(abs(a - b) < 1e-6 for a, b in zip(lc_dists, raw_dists)), \
        f"langchain_chroma 的 score 语义变了（与原始 cosine 距离对不上）: {lc_dists} vs {raw_dists}"
    print(f"2 距离口径: lc 分数与 Chroma 原始 cosine 距离逐项一致（min={lc_dists[0]:.4f}）")

    # ---- 3. 焦点版的"重排+补充"路径真的在走（锚定商品占住前排） ----
    lc_focus = search_with_product_focus("这款耳机有什么缺点", "review_knowledge", "SKU-10001")
    own_count = sum(1 for h in lc_focus if h["meta"].get("product_id") == "SKU-10001")
    assert lc_focus and own_count >= 1
    print(f"3 焦点检索: {own_count}/{len(lc_focus)} 块属于锚定商品 SKU-10001")

    # ---- 4. 坏 where 依旧早失败（D7 的校验对两版同样生效） ----
    try:
        search("随便", "product_knowledge", where={"brand": {"$ne": "x"}})
        raise AssertionError("坏算子应被 _validate_where 拦下")
    except ValueError:
        print("4 坏 where: $ne 被入口拦下（未进检索）")

    print("\n自测通过：LangChain 版与手写版在全部对照用例上逐项一致。")
    # （_hand_search 仅在类型层面提醒两版签名一致，未调用；保留 import 会被
    #  linter 删，因此这里实际不引用它。删掉上一行 import 亦可。）
