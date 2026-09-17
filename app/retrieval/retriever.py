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
      注意语义（阈值已按评测结论调到 0.012）：单路 rank1 = 1/(60+1)
      ≈ 0.0164（过线），两路 rank1 齐中 ≈ 0.0328（过线），单路 rank3
      = 1/63 ≈ 0.0159（仍过），单路 rank4 起 ≈ 0.0156 以下就悬了。
      即"某一条路把候选排得很靠前"就足以入围，不再像 0.02 那样要求
      两条路同时认可——因为拦"巧合"的活已经交给 SEMANTIC_MAX_DIST
      （向量硬门槛）和 MIN_LEXICAL_OVERLAP（词法置信门）了，阈值不该
      一个人干三件事。

    D6 元数据过滤两路同口径：where 在 Agent 层由槽位拼出（03 文档 §3
       product_id），检索器只负责执行。向量路把 where 透传给 ChromaDB；
       BM25 路必须在打分后先筛元数据子集、再在子集内重排号——若只在
       最后按 meta 剪一刀，留下来的 rank 是全库排名，和已被约束的那一路
       不可比，RRF 就失去意义了。这与 D3（余弦距离门槛后重排号）是同一
       个原则：**排位必须反映"进入融合的候选"的次序**。
    D7 只支持两种最简算子（相等 / $in），其余直接抛错：不支持却静默放行
       等于退化成全库检索、给出看似有依据的错误答案，比报错更糟。
    D8 话题商品用"重排+补充"而非独占过滤（见 search_with_product_focus）：
       非商品文档的 product_id 是空串，独占过滤会把 faq/guide 整体排除，
       把一个原本答对的 FAQ 问题改成答错。宁可只在排序层做倾斜。

依赖链：search ⊂ indexer.get_collection/get_model/embed_texts（D1 伏笔）。
命令行用法：python -m app.retrieval.retriever "查询词"
"""
import os
import sys

import jieba

from .schema import RRF_K, SCORE_THRESHOLD, SEMANTIC_MAX_DIST, TOP_K
from .indexer import embed_texts, get_collection

# D4：咨询场景的 doc_type 权重（乘在 RRF 分数上）。
# 评测教训（app/retrieval/eval.py）：guide 0.8 / faq 0.9 的衰减把融合分
# 压到商品块之下，"预算三百以内"这类查询的指南文档被挤出 top5——
# 加权只该用在"幻觉代价不同"的场合（价格只信 spec），不该用来表达
# "哪个类型更重要"。只保留 spec 的 1.3，其余一律 1.0。
DOC_TYPE_WEIGHTS = {
    "spec": 1.3,     # 结构化事实优先
    "product": 1.0,
    "faq": 1.0,
    "guide": 1.0,
    "policy": 1.0,
    "review": 1.0,
}

# D2：BM25 缓存  {collection: (bm25, ids, docs, metas)}
_bm25_cache: dict[str, tuple] = {}

# BM25 两侧共用的停用词：不分词侧过滤会造成 query/语料词频口径不一致。
# 评测教训：'对/的' 这类高频虚词在语料里 IDF 低但架不住次数多，垃圾
# query（"量子涨落对股市的影响"）靠它们攒出 4.76 分混过融合阈值。
STOPWORDS = frozenset({
    "的", "了", "吗", "呢", "吧", "啊", "呀", "哦", "嗯", "是", "在", "有", "和",
    "与", "及", "或", "对", "对于", "关于", "把", "被", "让", "给", "用", "能",
    "会", "可以", "要", "想", "想买", "求", "怎么", "怎么样", "怎样", "什么",
    "多少", "几", "这个", "那个", "这种", "那种", "还有", "还是", "就是", "一下",
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "它们", "买", "卖",
})

# 仅 BM25 支持的候选（向量路零支持）必须与 query 有至少这么多个不同实词
# 相交——向量路没背书时，词法巧合是唯一的假命中来源，这道门专拦它。
MIN_LEXICAL_OVERLAP = 2

# 已用 where 锚定到具体商品时，这道门放宽到 1 个实词（D6 附则）：
# 这道门的逻辑是"只有 BM25 一路认可，不够可信"；而 where 是第二重独立
# 证据（这块资料确实属于用户正在聊的那件商品），两者叠加就够资格入围了。
# 代价要写实：只放宽带过滤的那次检索，完全不碰无过滤路径，所以不会让
# "量子涨落"这类无关 query 重新混进来。
LEXICAL_OVERLAP_WHEN_FILTERED = 1


def _tokenize(text: str) -> list[str]:
    """BM25 统一的分词口径（query 与语料都走这里）。"""
    return [t for t in jieba.lcut(text) if t.strip() and t not in STOPWORDS]


def search(query: str, collection: str = "product_knowledge",
           top_k: int = TOP_K, where: dict | None = None) -> list[dict]:
    """混合检索入口。返回降序 top_k：

        [{"chunk_id", "text", "score", "meta"}]
    最高加权重 < SCORE_THRESHOLD 时返回 []（D5，上层走兜底话术）。

    where 是 Agent 层拼出的元数据过滤条件（D6），支持两种最简形式：
        {"product_id": "SKU-10001"}        相等
        {"brand": {"$in": ["A", "B"]}}     属于
    不支持的算子抛 ValueError（D7）：静默退化成全库检索会让调用方拿到
    一个像模像样但越界的答案，报错至少让人立刻知道。
    """
    _validate_where(where)                              # D7：坏条件早失败

    # 两路各自召回到候选排名（D3 的输入），两路都要受同一份 where 约束（D6）
    vector_ranks = _vector_route(query, collection, top_k, where)
    bm25_ranks = _bm25_route(query, collection, top_k, where)
    return _fuse_and_format(query, collection, top_k, where, vector_ranks, bm25_ranks)


# 精排候选池：初检求"全"多召回一些（top 20），精排求"准"再挑 top_k。
# 太小精排没得挑，太大模型慢；20 在 50 条评测集上是精度/耗时的折中。
RERANK_CANDIDATES = 20


def search_reranked(query: str, collection: str = "product_knowledge",
                    top_k: int = TOP_K, where: dict | None = None) -> list[dict]:
    """两阶段检索：混合初检 top20 → cross-encoder 精排 → 取 top_k。新增。

    返回格式与 search() 一致（score 字段换成 rerank 分）。开关语义：
    环境变量 SHOPMATE_RERANK=1 才启用，默认走 search() 原路径——保留
    两条路才能跑第五路对照（eval.py），也方便线上出问题随时切回。
    无答案判定不在精排层（reranker.py D4）：初检融合结果为空就直接
    返回 []，阈值口径只有一套。
    """
    if os.environ.get("SHOPMATE_RERANK", "").strip() not in ("1", "true", "yes"):
        return search(query, collection, top_k, where)
    candidates = search(query, collection, RERANK_CANDIDATES, where)
    if not candidates:
        return []
    from .reranker import rerank
    return rerank(query, candidates)[:top_k]


def _fuse_and_format(query: str, collection: str, top_k: int, where: dict | None,
                     vector_ranks: list, bm25_ranks: list) -> list[dict]:
    """RRF 融合之后的公共段：词法置信门 → doc_type 加权 → 阈值兜底 → 截断。

    抽成独立函数是给 LangChain 版检索器（lc_retriever）复用的——它只换
    向量路的实现（langchain_chroma），融合语义必须与本版逐字节一致，否则
    评测对照失去意义。本函数不含任何检索动作，只做纯计算。
    """
    fused = _rrf([vector_ranks, bm25_ranks])

    # 词法置信门：向量路零支持的候选，词法巧合是唯一来源，要求 ≥2 个
    # 实词相交才保留（否则停用词级的弱匹配会污染兜底线之上的结果）。
    # 带 where 时放宽到 1 个——"属于这件商品"是第二重证据（见 LEXICAL_...
    # 常量处的说明）。评测里"打游戏延迟高不高"只有"延迟"一个实词和正解
    # 块相交，正是靠这条才捞回来的。
    if bm25_ranks and not vector_ranks:
        need = (LEXICAL_OVERLAP_WHEN_FILTERED if where else MIN_LEXICAL_OVERLAP)
        q_tokens = set(_tokenize(query))
        _, ids_all, docs_all, _ = _bm25_state(collection)
        pos_all = {cid: i for i, cid in enumerate(ids_all)}
        fused = {cid: s for cid, s in fused.items()
                 if len(q_tokens & set(_tokenize(docs_all[pos_all[cid]])))
                 >= need}

    # D4：加权 → 排序 → D5 阈值 → 截断
    _, ids, docs, metas = _bm25_state(collection)      # 语料都在 D1 的缓存里
    pos = {cid: i for i, cid in enumerate(ids)}        # 一次建索引，别反复 .index()
    scored: list[tuple[str, float]] = []
    for cid, s in fused.items():
        w = DOC_TYPE_WEIGHTS.get(metas[pos[cid]].get("doc_type", ""), 1.0)
        scored.append((cid, s * w))
    scored.sort(key=lambda x: x[1], reverse=True)

    if not scored or scored[0][1] < SCORE_THRESHOLD:   # D5
        return []

    return [
        {"chunk_id": cid, "text": docs[pos[cid]],
         "score": round(s, 4), "meta": metas[pos[cid]]}
        for cid, s in scored[:top_k]
    ]


# ---------------- 两路召回 ----------------

def _vector_route(query: str, collection: str, top_k: int,
                  where: dict | None = None) -> list[tuple[str, int]]:
    """向量路：query 向量化 → ChromaDB 查询 → 距离门槛过滤。

    where 直接交给 ChromaDB 在库内过滤（D6），命中集合天然是满足约束的
    候选，后面的重排号逻辑与无过滤时完全一致。
    """
    coll = get_collection(collection)
    emb = embed_texts([query])[0]
    kw: dict = {"query_embeddings": [emb], "n_results": top_k,
                "include": ["distances"]}
    if where:
        kw["where"] = where
    hit = coll.query(**kw)
    # 只把 cosine 距离 ≤ SEMANTIC_MAX_DIST 的候选交给 RRF（两路合并阶段）。
    # 过滤后重新排名：候选 #2 达标但 #3 不达标时，#2 记 rank 1——
    # 排名必须反映"进入融合的候选"的次序，不能带洞。
    out: list[tuple[str, int]] = []
    for cid, dist in zip(hit["ids"][0], hit["distances"][0]):
        if dist <= SEMANTIC_MAX_DIST:
            out.append((cid, len(out) + 1))
    return out


def _bm25_route(query: str, collection: str, top_k: int,
                where: dict | None = None) -> list[tuple[str, int]]:
    """BM25 路：统一口径分词后对建好的索引查询。返回 [(chunk_id, 排名)]。

    BM25 索引是整库建的（D2 按 collection 缓存），所以 where 在打分之后
    执行（D6）：先从全库得分里挑出元数据命中的候选，再在这个子集内部
    重新排 1..top_k。若图省事后置到形成结果时才剪，剩下的就是全库排名。
    """
    bm25, ids, docs, metas = _bm25_state(collection)
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)                   # 和每篇语料的打分 array
    cand = [i for i in range(len(ids))
            if scores[i] > 0 and _match_meta(metas[i] or {}, where or {})]
    ranked = sorted(cand, key=lambda i: scores[i], reverse=True)
    return [(ids[i], r + 1) for r, i in enumerate(ranked[:top_k])]


def search_with_product_focus(query: str, collection: str, product_id: str,
                              top_k: int = TOP_K) -> list[dict]:
    """已知话题商品时的检索：把它的资料顶到前面，但**不丢**其它候选。

    为什么不用独占式 where 过滤：faq / guide / policy 这些非商品文档的元数据
    里 product_id 是空串，独占过滤会把它们整体排除。可"耳机保修多长时间"的
    正解恰恰是 faq 文档——那样做会把已经答对的问题改成答错。所以这里做的是
    「重排 + 补充」，无过滤的全量结果永远保留：
      1. 先无过滤检索得到 base（召回上界与原来完全一致，不可能退化）
      2. base 里属于该商品的块整体提到最前（精度提升）
      3. base 里该商品的块没占住前排（不足一半）→ 再按 product_id 过滤
         检索一次补进来。"不足一半"是个粗但够用的信号：用户明显在问 X，
         可 X 的资料连一半席位都占不到，说明裸相似度把它排到了别的东西
         后面，这时候补一次检索是真的救命（评测里最后那条
         "打游戏延迟高不高"就是这么救回来的：base 里有 X 的 spec 块，
         但回答所需的 product 块被排到了 top5 之外）。
    代价：走到第 3 步才多一次检索，满足第 2 步时零额外开销。
    """
    if not product_id:
        return search(query, collection, top_k)
    base = search(query, collection, top_k)
    own = [h for h in base if (h["meta"].get("product_id") or "") == product_id]
    others = [h for h in base if (h["meta"].get("product_id") or "") != product_id]
    if base and len(own) * 2 >= top_k:          # 已经占住前排 → 纯重排，不再打扰
        return (own + others)[:top_k]
    # base 为空（连阈值都没过）或该商品没占住前排，都补一次带锚定的检索：
    # 带 where 的那次会放宽词法门，常能把无过滤时被阈值杀掉的资料救回来。
    extra = search(query, collection, top_k, where={"product_id": product_id})
    seen = {h["chunk_id"] for h in own}         # 补进来的不能和已有的重复
    extra = [h for h in extra if h["chunk_id"] not in seen]
    return (own + extra + others)[:top_k]


# ---------------- 元数据过滤（D6/D7） ----------------

def _validate_where(where: dict | None) -> None:
    """提前校验过滤条件，坏条件在检索开始前就炸，不要等到 RRF 里才发现。"""
    if not where:
        return
    for key, cond in where.items():
        if isinstance(cond, dict):
            if set(cond) != {"$in"}:
                raise ValueError(
                    f"不支持的过滤算子 {sorted(set(cond))}（字段 {key}）："
                    f"目前只实现对等的相等匹配与 $in")
            if not isinstance(cond["$in"], (list, tuple)):
                raise ValueError(f"字段 {key} 的 $in 必须是列表")


def _match_meta(meta: dict, where: dict) -> bool:
    """ChromaDB where 的最小实现：{"k": v} 相等 / {"k": {"$in": [...]}} 属于。

    故意不支持 $and/$or/$ne 等：用不上先不实现，且由 _validate_where 在
    入口挡住——半吊子实现比不支持更危险（D7）。
    """
    for key, cond in where.items():
        val = meta.get(key)
        if isinstance(cond, dict):
            if val not in cond["$in"]:
                return False
        elif val != cond:
            return False
    return True


# ---------------- 状态与融合 ----------------

def _bm25_state(collection: str) -> tuple:
    """D2：按 collection 懒建 BM25 索引（语料来自 ChromaDB，D1）。"""
    if collection in _bm25_cache:
        return _bm25_cache[collection]
    coll = get_collection(collection, create=False)    # 检索端绝不建空库
    got = coll.get(include=["documents", "metadatas"])
    corpus = [_tokenize(d) for d in got["documents"]]
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

    # 用法三：元数据过滤（D6）——"这款耳机"指谁由 Agent 层的槽位决定
    print("\n=== 元数据过滤对照（where product_id=SKU-10001）")
    q, coll = "这款耳机有什么缺点", "review_knowledge"
    for label, w in (("无过滤", None), ("过滤", {"product_id": "SKU-10001"})):
        rs = search(q, coll, where=w)
        head = " / ".join(r["chunk_id"].split("#")[0] for r in rs) or "[]"
        print(f"  {label:3s} → {head}")
    print("  过滤后应全部收敛到 review:SKU-10001；实际 Agent 层锚定哪个商品，"
          "看 intent 抽到的 product_id 槽位。")
