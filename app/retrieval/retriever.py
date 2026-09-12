"""第 4 步：混合检索（向量 + BM25 + RRF 融合），01 文档 §5。

最后写。骨架和 loader/chunker 一样留白，indexer 完成后你再来。
"""


def search(query: str, collection: str = "product_knowledge", top_k: int = 5) -> list[dict]:
    """混合检索入口。

    流程（01 文档 §5 流程图）：
      1. 向量路：BGE-M3(query) → ChromaDB top_k
      2. BM25 路：同一 collection 的全量 chunk 建 BM25 索引（jieba 分词）→ top_k
      3. RRF 融合：score = Σ 1/(RRF_K + rank_i)
      4. 按 doc_type 加权（咨询意图：spec > product > faq > guide，系数放在本文件顶部常量）
      5. 最高融合分 < SCORE_THRESHOLD → 返回空列表（上层走"暂无相关信息"兜底）
    返回格式：[{"chunk_id", "text", "score", "meta"}]
    """
    raise NotImplementedError
