"""第 4 步：混合检索（向量 + BM25 → RRF 融合 → doc_type 加权）。已实现。

模块速览：
    search(query, collection, top_k)        检索入口（Agent 状态机调这个）
    _bm25_state(collection)                 该 collection 的 BM25 懒加载缓存
    _vector_route / _bm25_route             两路召回，各返回 [(chunk_id, rank)]
    _rrf(rank_lists)                        融合：{chunk_id: 分数}
五个设计决策（对着读，面试讲的就是这些）：
    D1 语料单一事实源：BM25 的语料不重新读 rag_docs，而是从 ChromaDB
       coll.get() 里捞——保证两条路检索的永远是同一份数据（建库后文件
       改了也不会出现"BM25 拿旧的、向量拿新的"）。这也是 indexer D2
       （模型/集合函数复用）埋的伏笔在这里兑现。
    D2 每个_collection一份懒加载缓存：{collection: (bm25, ids, docs, metas)}。
       和 get_model 单例同一个模式——重的对象只建一次，查询时瞬时复用。
       BM25 建在进程内存（README"已知局限"已声明：重启重建）。
    D3 RRF 融合：score(c) = Σ_route 1/(RRF_K + rank)。两条路都靠前的
       候选总分高；只在一条路出现的候选拿单份 1/(K+rank)。不需要把
       向量距离（0-1 的 cosine）和 BM25 分（0 到几十）硬拉到同一量纲
       ——这正是选 RRF 而不是加权求和的原因（免调分数阈值，工程稳）。
    D4 加权发生在融合之后：按 doc_type 给融合分乘系数（咨询场景
       spec > product > faq > guide，评价/政策不衰减）。依据：01 §5
       "价格类问题只信 spec"——spec 是结构化事实，幻觉代价高的场景
       应当优先它。
    D5 兜底阈值在加权后判定：加权后最高分 < SCORE_THRESHOLD → 返回 []，
       上层（状态机）走"暂无相关信息"+转人工入口（03 状态机 §4）。
      注意语义：schema.SCORE_THRESHOLD=0.02，换算一下——两路 rank1
      齐中 ≈ 0.033（过），单路 rank1 ≈ 0.016（不过）。即"只有一条
      路觉得非常相关"不足以过线，这是保守取向（宁可转人工，不硬答）。

依赖链：search ⊂ indexer.get_collection/get_model/embed_texts（D1 伏笔）。
命令行用法：python -m app.retrieval.retriever "查询词"
"""
import sys

import jieba

from .schema import RRF_K, SCORE_THRESHOLD, SEMANTIC_MAX_DIST, TOP_K
from .indexer import embed_texts, get_collection

# D4：咨询场景的 doc_type 权重（乘在 RRF 分数上）
DOC_TYPE_WEIGHTS = {
    "spec": 1.3,     # 结构化事实优先
    "product": 1.0,
    "faq": 0.9,
    "guide": 0.8,
    "policy": 1.0,
    "review": 1.0,
}

# D2：BM25 缓存  {collection: (bm25, ids, docs, metas)}
_bm25_cache: dict[str, tuple] = {}


def search(query: str, collection: str = "product_knowledge",
           top_k: int = TOP_K) -> list[dict]:
    """混合检索入口。返回降序 top_k：

        [{"chunk_id", "text", "score", "meta"}]
    最高加权重 < SCORE_THRESHOLD 时返回 []（D5，上层走兜底话术）。
    """
    # 两路各自召回到候选排名（D3 的输入）
    vector_ranks = _vector_route(query, collection, top_k)
    bm25_ranks = _bm25_route(query, collection, top_k)
    fused = _rrf([vector_ranks, bm25_ranks])

    # D4：加权 → 排序 → D5 阈值 → 截断
    _, ids, docs, metas = _bm25_state(collection)      # 语料都在 D1 的缓存里
    scored: list[tuple[str, float]] = []
    for cid, s in fused.items():
        w = DOC_TYPE_WEIGHTS.get(metas[ids.index(cid)].get("doc_type", ""), 1.0)
        scored.append((cid, s * w))
    scored.sort(key=lambda x: x[1], reverse=True)

    if not scored or scored[0][1] < SCORE_THRESHOLD:   # D5
        return []

    return [
        {"chunk_id": cid, "text": docs[ids.index(cid)],
         "score": round(s, 4), "meta": metas[ids.index(cid)]}
        for cid, s in scored[:top_k]
    ]


# ---------------- 两路召回 ----------------

def _vector_route(query: str, collection: str, top_k: int) -> list[tuple[str, int]]:
    """向量路：query 向量化 → ChromaDB 查询 → 距离门槛过滤。

    只把 cosine 距离 ≤ SEMANTIC_MAX_DIST 的候选交给 RRF（两路合并阶段）。
    过滤后重新排名：候选 #2 达标但 #3 不达标时，#2 记 rank 1——
    排名必须反映"进入融合的候选"的次序，不能带洞。
    """
    coll = get_collection(collection)
    emb = embed_texts([query])[0]
    hit = coll.query(query_embeddings=[emb], n_results=top_k,
                     include=["distances"])
    out: list[tuple[str, int]] = []
    for cid, dist in zip(hit["ids"][0], hit["distances"][0]):
        if dist <= SEMANTIC_MAX_DIST:
            out.append((cid, len(out) + 1))
    return out


def _bm25_route(query: str, collection: str, top_k: int) -> list[tuple[str, int]]:
    """BM25 路：jieba 分词后对建好的索引查询。返回 [(chunk_id, 排名)]。"""
    bm25, ids, docs, _ = _bm25_state(collection)
    tokens = [t for t in jieba.lcut(query) if t.strip()]
    scores = bm25.get_scores(tokens)                   # 和每篇语料的打分 array
    ranked = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)
    return [(ids[i], r + 1) for r, i in enumerate(ranked[:top_k]) if scores[i] > 0]


# ---------------- 状态与融合 ----------------

def _bm25_state(collection: str) -> tuple:
    """D2：按 collection 懒建 BM25 索引（语料来自 ChromaDB，D1）。"""
    if collection in _bm25_cache:
        return _bm25_cache[collection]
    coll = get_collection(collection, create=False)    # 检索端绝不建空库
    got = coll.get(include=["documents", "metadatas"])
    corpus = [[t for t in jieba.lcut(d) if t.strip()] for d in got["documents"]]
    from rank_bm25 import BM25Okapi                    # 小依赖，用到才 import
    state = (BM25Okapi(corpus), list(got["ids"]),
             list(got["documents"]), list(got["metadatas"]))
    _bm25_cache[collection] = state
    return state


def _rrf(rank_lists: list[list[tuple[str, int]]]) -> dict[str, float]:
    """D3：RRF 融合。输入若干路 [(chunk_id, rank)]，输出 {chunk_id: 总分}。"""
    fused: dict[str, float] = {}
    for ranks in rank_lists:
        for cid, rank in ranks:
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    return fused


# ---------------- 自测：python -m app.retrieval.retriever "查询词" ----------------
if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    # 用法一：命令行参数模式（README"快速开始"承诺的形态）
    if len(sys.argv) > 1:
        for r in search(sys.argv[1]):
            print(f"[{r['score']}] {r['chunk_id']}\n  {r['text'][:60]}\n")
        sys.exit(0)

    # 用法二：内置样例，一次看三个知识域 + 混合对单路的纠偏
    cases = [
        ("通勤戴的降噪耳机，预算六百左右，要续航长的", "product_knowledge"),
        ("洗澡能带吗？会不会进水坏掉", "product_knowledge"),        # 语义改写查询
        ("买了五天想退货，运费谁出", "policy_knowledge"),
        ("用过的说说洗烘一体机有什么毛病", "review_knowledge"),
        ("SKU-10001 现在什么价", "product_knowledge"),               # 精确型号：BM25 主场
        ("量子涨落对股市的影响", "product_knowledge"),               # 应命中 [] 兜底
    ]
    for q, coll in cases:
        print(f"\n=== {q}  [{coll}]")
        both = search(q, coll)
        vec_only = _vector_route(q, coll, 5)
        bm25_only = _bm25_route(q, coll, 5)
        if not both:
            print("  → []  （低于阈值，上层走『暂无相关信息』）")
        for r in both:
            tag = f"单路向量top1={vec_only[0][0] if vec_only else '-'} | 单路BM25top1={bm25_only[0][0] if bm25_only else '-'}"
            print(f"  [{r['score']}] {r['chunk_id']}")
            print(f"       {r['text'].splitlines()[0][:50]}")
        if both:
            print(f"  ({tag[:70]}...)")   # 对照：混合排序 vs 两条单路各自的 top1
    print("\n提示：SKU-10001 那条应看出 BM25 把精确型号顶上来；"
          "量子那条必须返回[]，否则阈值形同虚设。")
