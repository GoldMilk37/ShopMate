"""检索埋点：把每一次 RAG 检索落盘成一行 JSONL。已实现。

模块速览：
    cite_overlap(chunk, reply)   单块与回复的字面重叠率
    cite_flags(hits, reply)      本轮全部块的重叠率（已扣掉本轮样板文字）
    preview(text, n)             截断成预览
    build_record(**kw)           组装一条完整记录
    log_retrieval(record)        追加一行到 data/logs/retrieval.jsonl

三个设计决策：
    D1 写在检索包、**调用方在 Agent 层**：真正要统计的那条线是"这一次用户
       提问，检索召回的东西有没有被用上"——而"回复"只有 Agent 层才有。
       更关键的是 `search_with_product_focus` 内部会再调一次 `search`
       （补一次带锚点的检索），把日志写在检索器里会给同一个 query 记出
       两行，命中率凭空翻倍。所以本模块只提供纯函数 + 一个写入器，
       由 `graph._answer_with_rag` 在**唯一一个**日志点调用。
       reply 是**参数**不是 import，检索包不反向依赖 Agent 层。
    D2 引用判定是**代理指标**，不是真答案：用字符 4-gram 重叠率当"这块资料
       有没有被用上"的信号。它便宜、确定、零额外 API 成本，但有两个已知
       盲区，写在这里免得日后被人当成准确率：
       （a）**漏报同义改写**——模型把"续航 30 小时"改写成"能撑一天多"，
            字面重叠几乎为零，会被判成没引用；
       （b）**看不见"该召回而没召回"**——它只回答"召回的块用没用上"，
            不回答"该召回的块召回了没有"。后者是 eval.py 用人工标注集
            算的 hit@5（当前 100%），两件事不许混着说。
       记录里带 `citation.method` 字段，就是让读日志的人知道这个数是怎么来的。
       补充一条**实测**（2026-09-13，真实 LLM 跑口碑轮）：回复同时用上了三四块
       评价，但只有 1 块过了 0.15 的线，另外几块在 0.09~0.12——它们确实被用了，
       只是被改写掉了字面。也就是说这个阈值**偏保守**，会低报。
       对策不是去调那个常数，而是：每块原始重叠率都落在记录里（hits[].cite_overlap），
       阈值只是个便利标记。要重新划线，拿已有日志离线重算即可，不必重跑 LLM。
    D3 扣掉"本轮样板文字"：chunker 给每个块头部拼了商品锚点
       （`【SKU-10001 无线降噪蓝牙耳机 Pro | SoundCore】正文…`），所以同一件
       商品的每个块都长着同一个开头。不扣掉的话，回复里只要提一句商品名，
       该商品的所有块都会被判成"被引用"——系统性虚高，而且正好虚高在
       最热门的商品上。判据：在**本轮一半以上**的块里都出现的 gram 视为
       样板，从分母里扣掉。只有 1 个块时不扣（没有"本轮共有"这回事）。

记录只追加、不轮转。演示规模下不值得引入日志轮转，真上量再说。
"""
import json
import os
import time
from collections import Counter
from math import ceil
from pathlib import Path

# 与 executor 的工具日志同目录（该目录已被 .gitignore 忽略）
LOG_DIR = Path(__file__).resolve().parents[2] / "data" / "logs"
LOG_NAME = "retrieval.jsonl"

CITE_NGRAM = 4          # 字符 n-gram 的 n
CITE_THRESHOLD = 0.15   # 重叠率达到多少算"用上了"
PREVIEW_CHARS = 120     # 日志里每块存多少字预览
BOILER_SHARE = 0.5      # 出现在多少比例的块里算样板文字（D3）

CITE_METHOD = "char4gram_overlap_minus_turn_common"


def _grams(text: str, n: int = CITE_NGRAM) -> set[str]:
    """取字符 n-gram 集合。先去掉所有空白，免得"续 航"和"续航"对不上。"""
    t = "".join((text or "").split())
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def cite_overlap(chunk_text: str, reply: str) -> float:
    """单块的重叠率，**不扣样板**。给外部/测试用，主路径走 cite_flags。"""
    g = _grams(chunk_text)
    return len(g & _grams(reply)) / len(g) if g else 0.0


def cite_flags(hits: list[dict], reply: str) -> list[float]:
    """D2+D3：本轮每块与回复的重叠率，已扣掉本轮样板 gram。

    返回与 hits 等长的 float 列表；`>= CITE_THRESHOLD` 即判为"这块被用上了"。
    """
    per_hit = [_grams(h.get("text", "")) for h in hits]
    if not per_hit:
        return []
    r = _grams(reply)
    # 出现在 >= 半数块里的 gram = 分块锚点/模板句（D3）
    cnt: Counter = Counter()
    for g in per_hit:
        cnt.update(g)
    need = max(2, ceil(len(per_hit) * BOILER_SHARE))
    boiler = {g for g, c in cnt.items() if c >= need}

    out = []
    for g in per_hit:
        keep = g - boiler
        # 整块都是样板（不可能是）→ 无从归属，记 0 而不是除零
        out.append(len(keep & r) / len(keep) if keep else 0.0)
    return out


def preview(text: str, n: int = PREVIEW_CHARS) -> str:
    """截断成预览。超长时补一个省略号，故最长 n+1 个字符。

    trace 和日志里都只存预览：全文在 Chroma 里躺着，凭 chunk_id 能捞回来，
    没必要在每条记录里复制一遍（前端面板和 st.session_state 快照都会变重）。
    """
    t = text or ""
    return t[:n] + ("…" if len(t) > n else "")


def _doc_id(chunk_id: str) -> str:
    """`spec:SKU-10001#01` → `spec:SKU-10001`（chunker 的分块编号约定）。"""
    return (chunk_id or "").split("#")[0]


def hit_rows(hits: list[dict], overlaps: list[float]) -> list[dict]:
    """把检索回来的原始 hit 整理成一行一条的展示/落盘形态。

    公开是为了让 graph 的 trace 和本模块的记录共用同一份构造逻辑——
    两处各写一遍，迟早有一边忘了截断预览或忘了把 `cited` 算上。
    """
    rows = []
    for i, h in enumerate(hits):
        meta = h.get("meta") or {}
        ov = overlaps[i] if i < len(overlaps) else 0.0
        body = h.get("text", "")
        rows.append({
            "rank": i + 1,
            "chunk_id": h.get("chunk_id", ""),
            "doc_id": _doc_id(h.get("chunk_id", "")),
            # 注意：这是 RRF 融合分×doc_type 权重，只能跟 SCORE_THRESHOLD 比，
            # 不是相似度（余弦/距离），别当"相关度 0.03 很低"去解读（见 retriever）
            "rrf_score": h.get("score", 0.0),
            "doc_type": meta.get("doc_type", ""),
            "product_id": meta.get("product_id", ""),
            "preview": preview(body),
            "len": len(body),
            "cite_overlap": round(ov, 4),
            "cited": ov >= CITE_THRESHOLD,
        })
    return rows


def build_record(*, sid: str, turn: int, intent: str, collection: str, query: str,
                 focus: str, outcome: str, hits: list[dict], overlaps: list[float],
                 retrieval_ms: int, reply: str = "") -> dict:
    """组装一条检索记录。outcome ∈ answered / no_info / llm_fail。

    outcome 是必须的：没有它，LLM 挂掉那轮会记成"召回了但一块都没被引用"，
    网络一抖就把引用率拉低，把基础设施故障算成检索质量问题。
    """
    rows = hit_rows(hits, overlaps)
    cited = [r["rank"] for r in rows if r["cited"]]
    return {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sid": sid,
        "turn": turn,
        "intent": intent,
        "collection": collection,
        "query": query,
        "focus": focus,                    # 本轮实际用上的商品锚点（空串=没锚定）
        "outcome": outcome,
        "n_hits": len(rows),
        "top_rrf_score": rows[0]["rrf_score"] if rows else 0.0,
        "retrieval_ms": retrieval_ms,
        "citation": {
            "method": CITE_METHOD,
            "threshold": CITE_THRESHOLD,
            "any_cited": bool(cited),
            "max_overlap": round(max(overlaps), 4) if overlaps else 0.0,
            "cited_ranks": cited,
        },
        "reply": reply,
        "hits": rows,
    }


def log_retrieval(record: dict, *, path: Path | None = None) -> bool:
    """追加一行 JSONL。返回是否写成功——**任何情况都不抛**。

    path 可注入是为了自测不污染真实日志文件（executor 的自测直接读真日志，
    那是个反面教材）。设 SHOPMATE_TELEMETRY=0 可整体静默。
    """
    if os.environ.get("SHOPMATE_TELEMETRY", "1") == "0":
        return False
    target = path or (LOG_DIR / LOG_NAME)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as f:
            # default=str：记录里混进非 JSON 原生类型（如 numpy 标量）时降级成
            # 字符串，也不要让一条诊断日志把用户的正常对话打断
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return True
    except OSError as e:
        print(f"[telemetry] 检索日志写入失败（不影响对话）: {type(e).__name__}: {e}")
        return False


# ---------------- 自测：python -m app.retrieval.telemetry ----------------
if __name__ == "__main__":
    import sys
    import tempfile
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    # 自测不能被环境开关误伤：7/8 节要验证"写得进去"，若外层恰好设了
    # SHOPMATE_TELEMETRY=0，那两节会失败在"本该 True 却拿到 False"上——
    # 报错信息还指向写入器，排查方向整个是错的。所以整套自测期间强制打开，
    # 9 节自己会临时关掉再恢复，跑完再把外层原值放回去。
    _ambient = os.environ.get("SHOPMATE_TELEMETRY")
    os.environ["SHOPMATE_TELEMETRY"] = "1"

    # ---- 1. 基础性质 ----
    assert cite_overlap("", "随便什么回复") == 0.0, "空块应得 0，不该除零"
    assert cite_overlap("续航三十小时", "") == 0.0, "空回复应得 0"
    assert cite_overlap("续航三十小时", "续航三十小时") > 0.9, "逐字回声应接近 1"
    print("1 基础: 空输入/空回复得 0，逐字回声得 %.2f"
          % cite_overlap("续航三十小时", "续航三十小时"))

    # ---- 2. 已知盲区写成断言，不是注释（D2a） ----
    # 同义改写必然漏报。这条断言的意义在于：如果哪天有人"优化"成会误报的
    # 形态，或者把它当成准确率汇报，自测会先拦一次。
    chunk = "续航时间约 30 小时，充电盒可再提供 20 小时。"
    paraphrase = "这块电池能撑一天多，配上盒子还能更久。"
    ov = cite_overlap(chunk, paraphrase)
    assert ov < CITE_THRESHOLD, f"同义改写本就不该被认出（现 {ov:.2f}）"
    print(f"2 盲区: 同义改写重叠率 {ov:.2f} < 阈值 {CITE_THRESHOLD}（已知漏报，非 bug）")

    # ---- 3. 锚点误报：本模块存在的头号理由（D3） ----
    anchor = "【SKU-10001 无线降噪蓝牙耳机 Pro | SoundCore】"
    hits = [
        {"chunk_id": "rev:SKU-10001#00", "text": anchor + "好评集中在降噪和续航。", "score": 0.03,
         "meta": {"doc_type": "review", "product_id": "SKU-10001"}},
        {"chunk_id": "rev:SKU-10001#01", "text": anchor + "差评说充电盒偏大。", "score": 0.02,
         "meta": {"doc_type": "review", "product_id": "SKU-10001"}},
        {"chunk_id": "rev:SKU-10001#02", "text": anchor + "触控容易误触。", "score": 0.01,
         "meta": {"doc_type": "review", "product_id": "SKU-10001"}},
    ]
    naive = [cite_overlap(h["text"], anchor) for h in hits]
    assert all(v > 0 for v in naive), "前提：不扣样板时这三块都会被误判成被引用"
    flags = cite_flags(hits, anchor)
    assert not any(v >= CITE_THRESHOLD for v in flags), \
        f"只回声商品锚点不该算引用，实际 {flags}"
    print(f"3 锚点误报: 不扣样板 {[round(v, 2) for v in naive]} 全误判 → "
          f"扣掉本轮样板后 {[round(v, 2) for v in flags]} 全部正确否决")

    # ---- 4. 真引用只认那一块 ----
    reply = "差评主要是充电盒偏大，介意便携的话要留意。"      # 命中第 2 块独有的内容
    flags = cite_flags(hits, reply)
    assert flags[1] >= CITE_THRESHOLD, f"第 2 块应判为被引用，实际 {flags[1]:.2f}"
    assert flags[0] < CITE_THRESHOLD and flags[2] < CITE_THRESHOLD, \
        f"另外两块不该被牵连，实际 {flags}"
    print(f"4 真引用: 只有第 2 块被判引用 {[round(v, 2) for v in flags]}")

    # ---- 5. 单块时不扣样板（没有"本轮共有"这回事） ----
    one = [hits[0]]
    assert len(cite_flags(one, anchor)) == 1
    assert cite_flags([], "随便") == [], "空 hits 应返回空列表"
    print("5 边界: 单块不扣样板，空 hits 返回空表")

    # ---- 6. preview ----
    assert preview("短") == "短"
    long = "字" * 200
    p = preview(long)
    assert len(p) == PREVIEW_CHARS + 1 and p.endswith("…"), f"预览长度异常: {len(p)}"
    assert len(long) == 200
    print(f"6 预览: 200 字 → {len(p)} 字（含省略号），原始长度 {len(long)} 另存")

    # ---- 7. 落盘：注入路径，不碰真实日志 ----
    rec = build_record(sid="s-test", turn=1, intent="review_consult",
                       collection="review_knowledge", query="SKU-10001 口碑",
                       focus="SKU-10001", outcome="answered", hits=hits,
                       overlaps=flags, retrieval_ms=123, reply=reply)
    need_keys = {"ts", "sid", "turn", "intent", "collection", "query", "focus",
                 "outcome", "n_hits", "top_rrf_score", "retrieval_ms", "citation",
                 "reply", "hits"}
    assert set(rec) == need_keys, f"记录字段缺失/多余: {set(rec) ^ need_keys}"
    assert rec["citation"]["method"] == CITE_METHOD
    assert rec["hits"][0]["chunk_id"] == "rev:SKU-10001#00"
    assert rec["hits"][0]["doc_id"] == "rev:SKU-10001"
    assert rec["hits"][0]["len"] == len(hits[0]["text"]), "len 应是未截断的原长"
    assert rec["hits"][0]["preview"] == preview(hits[0]["text"])
    assert rec["citation"]["cited_ranks"] == [2], rec["citation"]

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "sub" / "r.jsonl"          # 顺带验证会自动建父目录
        assert log_retrieval(rec, path=tmp) is True
        assert log_retrieval(rec, path=tmp) is True
        lines = tmp.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2, f"应恰好追加两行，实际 {len(lines)}"
        assert json.loads(lines[0])["sid"] == "s-test", "每行应是独立可解析的 JSON"
        print(f"7 落盘: 自动建目录 + 追加 {len(lines)} 行，每行独立可解析")

        # 写不进去时不抛，只返回 False（诊断坏了不能打断对话）
        bad = Path(d) / "r.jsonl"
        bad.mkdir()                                # 拿目录当文件用
        assert log_retrieval(rec, path=bad) is False
        print("8 容错: 路径不可写 → 返回 False，未抛异常")

        # 开关：SHOPMATE_TELEMETRY=0 整体静默
        off = Path(d) / "off.jsonl"
        os.environ["SHOPMATE_TELEMETRY"] = "0"
        try:
            assert log_retrieval(rec, path=off) is False
            assert not off.exists(), "关掉埋点后不该产生文件"
        finally:
            os.environ.pop("SHOPMATE_TELEMETRY", None)
        print("9 开关: SHOPMATE_TELEMETRY=0 时不落盘")

    if _ambient is None:
        os.environ.pop("SHOPMATE_TELEMETRY", None)
    else:
        os.environ["SHOPMATE_TELEMETRY"] = _ambient
    print("\n自测通过。注意 D2：本模块给出的是引用**代理指标**，不是 hit@5。")
