"""chunk 策略对比：三种切法各建一套库，同一评测集跑分（01 文档 §四）。新增。

模块速览：
    main()   typed(750) / fixed512 / fixed256 三套库 × 三路检索 × 三指标

两个设计决策：
    D1 每种策略独立持久化目录（data/chroma_compare/<tag>）：索引按切法
       重建，绝不共用 collection——块边界变了 chunk_id 语义就变了，
       混在一个库里 upsert 覆盖会得到两种策略的杂交结果。
    D2 切换库的方式是运行时改 indexer.PERSIST_ROOT + 清 _bm25_cache：
       get_collection 每次现读 PERSIST_ROOT（不缓存 client），BM25 缓存
       键是 collection 名，不按库区分——不清缓存的话，换库后 BM25 还在
       用旧语料打分，向量路和词法路就在评两套不同的库。

用法：python -m app.retrieval.chunk_compare
产出：策略 × 路线 × (hit@5 / hit@1 / MRR) 对比表 + 块数/块长分布。
本地推理重跑 embedding（不花钱但费时间），全程不联网。
"""
import shutil
import sys
from pathlib import Path

from . import indexer, retriever
from .chunker import chunk_all
from .loader import load_all
from .eval import EVAL_SET, _rank_metrics

COMPARE_ROOT = Path(__file__).resolve().parents[2] / "data" / "chroma_compare"

STRATEGIES = ("typed", "fixed512", "fixed256")


def _use_lib(tag: str) -> None:
    """把检索端的库切到 data/chroma_compare/<tag>（D2）。"""
    indexer.PERSIST_ROOT = COMPARE_ROOT / tag
    retriever._bm25_cache.clear()


def _build(strategy: str) -> int:
    """按策略切块并建独立库，返回 chunk 总数。"""
    target = COMPARE_ROOT / strategy
    if target.exists():
        shutil.rmtree(target)                 # 重建，避免残留旧块
    chunks = chunk_all(load_all(), strategy)
    indexer.build_index(chunks, persist_dir=target)
    return len(chunks)


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    routes = {
        "vector": lambda q, c, k: retriever._vector_route(q, c, k),
        "bm25": lambda q, c, k: retriever._bm25_route(q, c, k),
        "hybrid": lambda q, c, k: [(r["chunk_id"], i)
                                   for i, r in enumerate(retriever.search(q, c), 1)],
    }

    results: dict[str, dict[str, tuple[float, float, float]]] = {}
    n = len(EVAL_SET)
    for s in STRATEGIES:
        n_chunks = _build(s)
        lens = []
        _use_lib(s)
        for coll_chunks in chunk_all(load_all(), s):
            lens.append(len(coll_chunks.text))
        avg = sum(lens) / len(lens)
        print(f"[{s}] 库已建：{n_chunks} 块，平均块长 {avg:.0f} 字符")

        row = {}
        for rname, fn in routes.items():
            h5 = h1 = rr = 0.0
            for q, coll, allowed in EVAL_SET:
                a, b, c, _ = _rank_metrics(q, coll, allowed, fn)
                h5 += a; h1 += b; rr += c
            row[rname] = (h5 / n, h1 / n, rr / n)
        results[s] = row

    print(f"\n===== chunk 策略对比（评测集 {n} 条）=====")
    print(f"  {'策略':10s} {'路线':8s} {'hit@5':>7s} {'hit@1':>7s} {'MRR':>7s}")
    for s in STRATEGIES:
        for rname, (h5, h1, mrr) in results[s].items():
            print(f"  {s:10s} {rname:8s} {h5:>7.0%} {h1:>7.0%} {mrr:>7.3f}")

    print("\n解读要点：fixed 策略若只掉 hit@1/MRR 不掉 hit@5，说明定长块")
    print("还能被捞进前五但排序变差——结构信息（标题归属/节边界）的价值")
    print("主要体现在排序质量上，这正是 hit@5 饱和后要看 MRR 的原因。")


if __name__ == "__main__":
    main()
