"""RAG 检索评测：01 文档 §7 的落地。已实现。

模块速览：
    EVAL_SET            50 条 (query, collection, 可接受doc_id集合)
    main()              三路对比跑分 + 未命中清单

四个设计决策：
    D1 标注"可接受文档集合"而不是单一 doc_id：一条真实问法往往 2~3 个
       文档都算命中（问续航，spec 和商品详情都是正确答案）。按集合判
       才公平，否则分数低估检索质量。
    D2 hit@5 判到文档级：chunk 命中后按 '#' 截断回 doc_id 再比对——
       评测问的是"找没找到对的资料"，不是"找没找到那一段"，段级
       命中率会随分块策略波动，文档级才反映用户可感知的质量。
    D3 多路对照共用同一批 query 同一个 k：只有变量是检索方式，差值才
       能归因给 RRF 融合与 doc_type 加权。纯向量/纯 BM25 直接复用
       retriever 的内部函数，保证对照的就是线上同一套代码。
    D4 评测集内嵌在代码里：50 条是人工整理的资产，diff 友好、可追溯
       每条标注的理由；量大后再挪 JSON/CSV 不迟（迁移只需改一个变量）。

第四路 focus 对照的是 Agent 层的 product_id 槽位过滤（QUERY_FOCUS 标注
该查询发出时用户正在聊的商品）。它的期望不是"越高越好"，而是**相对 hybrid
非负**：要是出现负数，说明锚点把原本正确的答案挤出了 top5。

用法：python -m app.retrieval.eval
产出：多路 hit@5 对比 + 每路的未命中 query 清单（调阈值/加权前先看这个）。
"""
from .retriever import TOP_K

# ---- 评测集（D1：可接受文档集合；顺序：query, collection, 期望 doc_id 集合） ----
EVAL_SET: list[tuple[str, str, set[str]]] = [
    # -- 商品咨询 / 参数 / FAQ / 指南（product_knowledge，25 条） --
    ("通勤坐地铁想买个降噪耳机", "product_knowledge", {"product:SKU-10001", "guide:guide_earphone"}),
    ("SKU-10001 续航能用多久", "product_knowledge", {"spec:SKU-10001", "product:SKU-10001"}),
    ("耳机戴着跑步出汗会坏吗", "product_knowledge", {"faq:faq_earphone", "product:SKU-10001"}),
    ("苹果手机能用这款耳机吗", "product_knowledge", {"faq:faq_earphone"}),
    ("支持 LDAC 吗 音质怎么样", "product_knowledge", {"faq:faq_earphone", "product:SKU-10001"}),
    ("充电盒能给手机反向充电吗", "product_knowledge", {"faq:faq_earphone"}),
    ("耳机保修多长时间", "product_knowledge", {"faq:faq_earphone"}),
    ("预算三百以内有什么入门耳机", "product_knowledge", {"guide:guide_earphone"}),
    ("六百到一千预算买哪个旗舰耳机", "product_knowledge",
     {"guide:guide_earphone", "product:SKU-20001"}),   # 该档位推荐商品本身就是答案
    ("Hi-Res 认证是不是代表音质好", "product_knowledge", {"guide:guide_earphone"}),
    ("冲锋衣防水指数静水压是多少", "product_knowledge", {"spec:SKU-30001"}),
    ("冲锋衣可以用柔顺剂洗吗", "product_knowledge", {"spec:SKU-30001"}),
    ("170 的身高穿 L 码胸围多少", "product_knowledge", {"spec:SKU-30001"}),
    ("洗衣机晚上洗噪音会不会吵到人", "product_knowledge", {"product:SKU-40001"}),
    ("10 公斤容量够几个人用", "product_knowledge", {"faq:faq_appliance"}),
    ("洗烘一体机和分开买哪个好", "product_knowledge", {"faq:faq_appliance"}),
    ("热泵烘干和普通烘干有什么区别", "product_knowledge", {"faq:faq_appliance"}),
    ("空气洗是什么 能去异味吗", "product_knowledge", {"faq:faq_appliance"}),
    ("羊毛衫能扔烘干机里烘吗", "product_knowledge", {"faq:faq_appliance"}),
    ("SKU-10001 和 SKU-20001 哪个好", "product_knowledge", {"product:SKU-10001", "product:SKU-20001", "faq:faq_earphone"}),
    ("这款耳机能用来专业录音监听吗", "product_knowledge", {"product:SKU-10001"}),
    ("冲锋衣本身有多重", "product_knowledge", {"spec:SKU-30001"}),
    ("打游戏延迟高不高", "product_knowledge", {"product:SKU-10001"}),
    ("洗衣机用的什么电机", "product_knowledge", {"product:SKU-40001", "spec:SKU-40001"}),
    ("冲锋衣有什么颜色和尺码", "product_knowledge", {"spec:SKU-30001"}),
    # -- 平台政策（policy_knowledge，13 条） --
    ("七天无理由退货要满足什么条件", "policy_knowledge", {"policy:return_policy"}),
    ("数码产品拆封了还能退吗", "policy_knowledge", {"policy:return_policy"}),
    ("退货的运费谁承担", "policy_knowledge", {"policy:return_policy"}),
    ("质量问题多久之内可以换货", "policy_knowledge", {"policy:return_policy"}),
    ("退款多久能到账微信", "policy_knowledge", {"policy:return_policy"}),
    ("买完降价了能补差价吗", "policy_knowledge", {"policy:return_policy"}),
    ("美妆拆封后还能退吗", "policy_knowledge", {"policy:return_policy"}),
    ("现货一般多久发货", "policy_knowledge", {"policy:shipping_policy"}),
    ("满多少钱包邮", "policy_knowledge", {"policy:shipping_policy"}),
    ("寄新疆西藏要加运费吗", "policy_knowledge", {"policy:shipping_policy"}),
    ("可以指定发顺丰快递吗", "policy_knowledge", {"policy:shipping_policy"}),
    ("港澳台能发货吗", "policy_knowledge", {"policy:shipping_policy"}),
    ("大促期间发货会不会变慢", "policy_knowledge", {"policy:shipping_policy"}),
    # -- 用户评价（review_knowledge，12 条） --
    ("SKU-10001 这个耳机口碑怎么样", "review_knowledge", {"review:SKU-10001"}),
    ("这款耳机有什么缺点", "review_knowledge", {"review:SKU-10001"}),
    ("降噪实际体验怎么样 坐地铁", "review_knowledge", {"review:SKU-10001"}),
    ("冲锋衣防水实测如何 真的能挡雨吗", "review_knowledge", {"review:SKU-30001"}),
    ("冲锋衣夏天穿会不会闷", "review_knowledge", {"review:SKU-30001"}),
    ("压胶洗完会起泡吗", "review_knowledge", {"review:SKU-30001"}),
    ("冲锋衣好买吗 会不会断码", "review_knowledge", {"review:SKU-30001"}),
    ("洗烘一体机用过的人怎么说", "review_knowledge", {"review:SKU-40002"}),
    ("这台洗衣机评价怎么样", "review_knowledge", {"review:SKU-40001"}),
    ("充电盒体积大吗 好携带吗", "review_knowledge", {"review:SKU-10001"}),
    ("耳机触控会不会误触", "review_knowledge", {"review:SKU-10001"}),
    ("冲锋衣版型穿起来好看吗", "review_knowledge", {"review:SKU-30001"}),
]


# ---- 话题商品标注：这一条查询发出时，用户"正在聊哪件商品" ----
# 语义等价于 Agent 层的 product_id 槽位（01 文档 §三的过滤输入）：真实对话里
# 它来自前一句的型号或会话锚点，评测里由人工标注给出。没列进来的表示这句
# 没有话题锚点（纯推荐/政策/跨商品 FAQ），只走普通混合检索。
# 标注原则：只有当"用户明显在追问某一件已知商品"时才标——把跨商品的 FAQ
# 问题硬安一个商品上去，是在给过滤刷分，不是在测它。
QUERY_FOCUS: dict[str, str] = {
    # product_knowledge
    "SKU-10001 续航能用多久": "SKU-10001",
    "冲锋衣防水指数静水压是多少": "SKU-30001",
    "冲锋衣可以用柔顺剂洗吗": "SKU-30001",
    "170 的身高穿 L 码胸围多少": "SKU-30001",
    "洗衣机晚上洗噪音会不会吵到人": "SKU-40001",
    "这款耳机能用来专业录音监听吗": "SKU-10001",
    "冲锋衣本身有多重": "SKU-30001",
    "打游戏延迟高不高": "SKU-10001",
    "洗衣机用的什么电机": "SKU-40001",
    "冲锋衣有什么颜色和尺码": "SKU-30001",
    # review_knowledge
    "SKU-10001 这个耳机口碑怎么样": "SKU-10001",
    "这款耳机有什么缺点": "SKU-10001",
    "降噪实际体验怎么样 坐地铁": "SKU-10001",
    "冲锋衣防水实测如何 真的能挡雨吗": "SKU-30001",
    "冲锋衣夏天穿会不会闷": "SKU-30001",
    "压胶洗完会起泡吗": "SKU-30001",
    "冲锋衣好买吗 会不会断码": "SKU-30001",
    "洗烘一体机用过的人怎么说": "SKU-40002",
    "这台洗衣机评价怎么样": "SKU-40001",
    "充电盒体积大吗 好携带吗": "SKU-10001",
    "耳机触控会不会误触": "SKU-10001",
    "冲锋衣版型穿起来好看吗": "SKU-30001",
}


def _hit(query: str, collection: str, allowed: set[str],
         route) -> tuple[bool, list[str]]:
    """route(query, coll, k) → [(chunk_id, rank)]；判文档级命中（D2）。"""
    ranked = route(query, collection, TOP_K)
    docs = [cid.split("#")[0] for cid, _ in ranked]
    return any(d in allowed for d in docs), docs


def main() -> None:
    import os
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    assert len(EVAL_SET) >= 50, f"评测集不足 50 条: {len(EVAL_SET)}"

    # 检索实现二选一（D3 双版对照的延伸）：SHOPMATE_RETRIEVER=lc 走 LangChain 版
    # （langchain_chroma 向量路），默认手写版。评测集是两版共同的回归网——
    # 换实现跑一遍，hit@5 不降且与另一版未命中清单一致，才算换得干净。
    if os.environ.get("SHOPMATE_RETRIEVER", "").lower() in ("lc", "langchain"):
        from . import lc_retriever as impl
        impl_name = "LangChain 版（langchain_chroma 向量路）"
    else:
        from . import retriever as impl
        impl_name = "手写版（chromadb 直连向量路）"

    # 预热两个懒加载资源（BM25 索引 / BGE-M3），否则首轮计时失真无妨、首轮报错难查
    modes: dict[str, list[tuple[bool, list[str]]]] = {"vector": [], "bm25": [], "hybrid": [], "focus": []}
    misses: dict[str, list[str]] = {"vector": [], "bm25": [], "hybrid": [], "focus": []}

    def hybrid_route(query, c, k):
        return [(r["chunk_id"], i) for i, r in enumerate(impl.search(query, c), 1)]

    def focus_route(query, c, k):
        """第四路：混合 + 话题商品倾斜（Agent 层 product_id 槽位的落地形态）。"""
        return [(r["chunk_id"], i)
                for i, r in enumerate(
                    impl.search_with_product_focus(query, c, QUERY_FOCUS.get(query, "")), 1)]

    for q, coll, allowed in EVAL_SET:
        h_v, _ = _hit(q, coll, allowed, impl._vector_route)
        h_b, _ = _hit(q, coll, allowed, impl._bm25_route)
        h_m, _ = _hit(q, coll, allowed, hybrid_route)
        h_f, _ = _hit(q, coll, allowed, focus_route)
        for name, h in (("vector", h_v), ("bm25", h_b), ("hybrid", h_m), ("focus", h_f)):
            modes[name].append((h, []))
            if not h:
                misses[name].append(f"[{coll.split('_')[0]}] {q}  期望:{sorted(allowed)}")

    n = len(EVAL_SET)
    print(f"评测集 {n} 条，hit@5 多路对比（01 文档 §7）——检索实现：{impl_name}")
    base = None
    hits_of: dict[str, int] = {}
    for name in ("vector", "bm25", "hybrid", "focus"):
        hits = sum(1 for h, _ in modes[name] if h)
        hits_of[name] = hits
        print(f"  {name:8s} {hits}/{n} = {hits / n:.0%}")
        if name == "bm25":
            base = hits / n
    print(f"\n混合 vs 纯 BM25 提升: {(hits_of['hybrid'] / n - base):+.0%}"
          f"（目标 ≥85% 且 +30%，见 01 文档 §一/§七）")
    print(f"话题倾斜 vs 混合: {hits_of['focus'] - hits_of['hybrid']:+d} 条"
          f"（应为正或 0；出现负数说明锚点挤掉了正确答案，必须回查）")

    for name in ("vector", "bm25", "hybrid", "focus"):
        if misses[name]:
            print(f"\n--- {name} 未命中 {len(misses[name])} 条 ---")
            for m in misses[name]:
                print("  ", m)


if __name__ == "__main__":
    main()
