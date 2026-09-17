"""Rerank 精排：cross-encoder 对初检候选重排序（01 文档 §五两阶段检索）。新增。

模块速览：
    rerank()    对 search() 返回的候选打相关性分并重排

设计决策：
    D1 cross-encoder 与话题锚点重排是两回事：仓库里原有的"重排"
       （search_with_product_focus）是排序规则调整——把话题商品的块提
       到前排；本模块是模型精排——把"问题+候选段落"拼成一条过
       bge-reranker-v2-m3，直接输出相关性分数。它准得多但慢得多，
       不能对全库跑，只对初检回来的候选跑（两阶段检索的第二阶段）。
    D2 懒加载单例：与 indexer.get_model 同一模式。模型约 1.1GB，
       加载十几秒，rerank 开关没开就不该背上这份内存。
    D3 批量打分：FlagReranker.compute_score 一次吃全部候选对，
       内部自动批处理（RERANK_BATCH），比逐条调快一个量级。
    D4 分数与阈值语义隔离：rerank 分是 0~1 的 sigmoid，与融合分的
       阈值口径（0.012）不可比。是否"无答案"仍由初检段的融合阈值
       判定（search_reranked 里 fused 为空即兜底），rerank 只负责
       把最相关的排到前面——两阶段各管各的，不在精排层发明第二个阈值。

用法：
    SHOPMATE_RERANK=1 时 retriever.search_reranked 生效（默认关闭，保留原路径对照）。
自测：python -m app.retrieval.reranker（需要模型已在 models/ 下）。
"""
import os
import threading
from pathlib import Path

# 模型本地路径（snapshot_download 的 local_dir，避免运行时联网查 HF）
MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "bge-reranker-v2-m3"

# 一次喂给模型的候选对数：32 对约 2GB 峰值内存，50 条评测集上吞吐够用
BATCH = 32

_model = None
_lock = threading.Lock()


def get_model():
    """D2：懒加载单例。模型目录不存在时报错并给出补救命令。"""
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            if not MODEL_DIR.exists():
                raise RuntimeError(
                    f"reranker 模型不在 {MODEL_DIR}。下载（需能访问 hf-mirror）：\n"
                    '  HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 '
                    'python -c "from huggingface_hub import snapshot_download; '
                    "snapshot_download('BAAI/bge-reranker-v2-m3', "
                    "local_dir='models/bge-reranker-v2-m3')\"")
            from FlagEmbedding import FlagReranker
            _model = FlagReranker(str(MODEL_DIR), use_fp16=True)
    return _model


def rerank(query: str, hits: list[dict]) -> list[dict]:
    """对初检候选精排：返回按 rerank 分降序的新列表（不改动入参）。

    hits 是 search() 的产出格式 [{"chunk_id", "text", "score", "meta"}]。
    分数字段被替换成 rerank 分（D4），meta 里补 reranked=True 供前端/埋点区分。
    """
    if len(hits) <= 1:
        return list(hits)
    pairs = [[query, h["text"]] for h in hits]
    scores = get_model().compute_score(pairs, batch_size=BATCH,
                                       normalize=True)   # normalize → 0~1 sigmoid
    if isinstance(scores, float):                       # 单条时返回标量
        scores = [scores]
    ranked = sorted(
        ({**h, "score": float(s), "meta": {**h["meta"], "reranked": True}}
         for h, s in zip(hits, scores)),
        key=lambda h: h["score"], reverse=True)
    return ranked


# ---------------- 自测：python -m app.retrieval.reranker ----------------
if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    hits = [
        {"chunk_id": "policy:shipping_policy#01", "text": "现货商品 48 小时内发货，满 99 元包邮。",
         "score": 0.5, "meta": {"doc_type": "policy"}},
        {"chunk_id": "spec:SKU-10001#02", "text": "SKU-10001 降噪耳机续航：ANC 开启 30 小时，关闭 40 小时。",
         "score": 0.4, "meta": {"doc_type": "spec"}},
    ]
    out = rerank("耳机续航多久", hits)
    assert out[0]["chunk_id"].startswith("spec:SKU-10001"), \
        f"续航问题应把 spec 排前面: {[h['chunk_id'] for h in out]}"
    assert out[0]["meta"]["reranked"] is True
    print("1 精排序正确：", [(h["chunk_id"], round(h["score"], 3)) for h in out])
    print("2 入参未被改动：", hits[0]["score"] == 0.5)

    single = rerank("任意", hits[:1])
    assert len(single) == 1
    print("3 单条直通 OK")
    print("自测通过")
