"""本轮轨迹的**契约与渲染**：键名在这里定义，展示逻辑在这里实现。已实现。

模块速览：
    TRACE_KEYS / STAGES / FALLBACKS   轨迹的合法键名与取值（graph 也 import 这个）
    summarize(trace)      → 标量摘要 dict（给人看的一行行）
    hit_rows(trace)       → 召回块表格数据
    tool_rows(trace)      → 工具调用表格数据
    fallback_label(trace) → 命中的兜底话术标签，没命中返回 None
    render_text(trace)    → 纯文本版（自测断言它，cli 的 /trace 也用它）

四个设计决策：
    D1 键名契约放在这里、由 graph import：轨迹是个自由 dict，两边各写一遍
       键名的话，一个拼写错误的表现是"前端那一栏永远是空的"——和"这轮本来
       就没有这个数据"长得一模一样，极难发现。放一处，graph 侧遇到不认识的
       键会当场告警。
    D2 不 import streamlit：本模块要能被自测和 CLI 用，而 streamlit 是个重依赖
       （且 import 时会做运行时检查）。格式化只产出 plain dict / str，由前端
       决定怎么画。
    D3 模块名不能叫 `trace.py`：`streamlit run` 会把脚本所在目录（app/agent/）
       塞进 sys.path 的最前面，一个叫 trace 的顶层模块会**遮蔽标准库的
       trace** 模块——那种 bug 的报错离现场十万八千里。
    D4 所有格式化函数必须容忍缺键、容忍半成品轨迹：确认门那一轮根本没跑意图
       识别（没有 intent），空召回那轮 hits 是空表，LLM 挂掉那轮只有降级话术。
       轨迹是"走到哪记到哪"，不是"填满了才成立"。自测专门钉这条性质。

术语：「兜底」= 主流程走不通时的备用出路（澄清追问/转人工/资料暂无），
本文件里的 FALLBACKS 就是这些出路的枚举。
"""

# ---- D1：契约。graph 侧写 trace 时用这里的名字 ----
TRACE_KEYS = (
    "sid", "turn", "user",            # 谁、第几轮、说了什么
    "stage",                          # 本轮走到哪个分支
    "intent", "confidence", "dissatisfied", "slots",   # 意图识别产物
    "collection", "query", "focus", "anchor",          # 检索侧
    "hits", "cited",                                   # 召回与引用判定
    "tool_calls", "gate",                              # 工具编排与确认门
    "fallback", "transfer_reason",                     # 兜底与转人工原因
    "elapsed_ms",                                      # 本轮总耗时
)

STAGES = ("confirmation", "classify", "clarify", "rag", "tool", "chitchat", "transfer")

FALLBACKS = (
    "low_confidence_clarify",     # 置信度 < 阈值 → 追问澄清
    "dissatisfied_transfer",      # 连续不满 → 主动转人工
    "empty_retrieval",            # 检索没召回/低于阈值 → "暂无相关信息"
    "llm_fail",                   # 模型没答上（重试后仍失败）→ 降级话术
    "tool_fail_transfer",         # 工具连续失败 → 转人工
    "tool_rounds_exhausted",      # 工具编排轮次用尽 → 兜底
)

STAGE_LABELS = {
    "confirmation": "确认门（待确认写操作）",
    "classify": "意图识别",
    "clarify": "追问澄清",
    "rag": "RAG 检索",
    "tool": "工具调用",
    "chitchat": "闲聊",
    "transfer": "转人工",
}

FALLBACK_LABELS = {
    "low_confidence_clarify": "置信度不足，追问澄清（没让模型硬猜）",
    "dissatisfied_transfer": "用户连续表达不满，主动转人工（附摘要）",
    "empty_retrieval": "检索没拿到可用资料，回答“暂无相关信息”",
    "llm_fail": "模型重试后仍失败，给降级话术（不是“查不到”）",
    "tool_fail_transfer": "工具连续失败，转人工",
    "tool_rounds_exhausted": "工具编排轮次用尽，收尾",
}

# 展示用的列（顺序即表格列序）。列名保留英文键，值才是中文——前端要按
# 键取值，改名会把表格改空。
HIT_COLUMNS = ("rank", "chunk_id", "rrf_score", "doc_type", "product_id",
               "cited", "cite_overlap", "len", "preview")
TOOL_COLUMNS = ("name", "status", "ok", "message")


def stage_label(trace: dict) -> str:
    """本轮分支的中文名。未知或缺失时回落到原文/占位，不抛。"""
    s = (trace or {}).get("stage", "")
    return STAGE_LABELS.get(s, s or "（未记录）")


def fallback_label(trace: dict) -> str | None:
    """命中的兜底标签；本轮没走兜底则返回 None（区别于"兜底了但名字不认识"）。"""
    f = (trace or {}).get("fallback")
    if not f:
        return None
    return FALLBACK_LABELS.get(f, f)


def summarize(trace: dict) -> dict:
    """D4：标量摘要，给前端画"这一轮发生了什么"。缺键一律容忍。"""
    t = trace or {}
    hits = t.get("hits") or []
    cited = t.get("cited") or []
    tools = t.get("tool_calls") or []
    conf = t.get("confidence")
    return {
        "会话": t.get("sid", ""),
        "轮次": t.get("turn", ""),
        "分支": stage_label(t),
        "意图": t.get("intent", "（未跑意图识别）"),
        "置信度": f"{conf:.2f}" if isinstance(conf, (int, float)) else "—",
        "用户不满": "是" if t.get("dissatisfied") else "否",
        "检索库": t.get("collection", "—"),
        "检索词": t.get("query", "—"),
        "商品锚点": t.get("focus") or "（未锚定）",
        "召回块数": len(hits),
        "判定被引用": sum(1 for c in cited if c),
        "工具调用": len(tools),
        "确认门": t.get("gate", "—"),
        "转人工原因": t.get("transfer_reason", "—"),
        "兜底": fallback_label(t) or "未触发",
        "耗时ms": t.get("elapsed_ms", "—"),
    }


def hit_rows(trace: dict) -> list[dict]:
    """召回块表格。D4：hits 缺失/为空返回空表，绝不抛。"""
    return [h for h in ((trace or {}).get("hits") or []) if isinstance(h, dict)]


def tool_rows(trace: dict) -> list[dict]:
    """工具调用表格。D4：同上。"""
    return [r for r in ((trace or {}).get("tool_calls") or []) if isinstance(r, dict)]


def render_text(trace: dict) -> str:
    """纯文本版轨迹。给 CLI 的 /trace 和自测用。"""
    t = trace or {}
    if not t:
        return "（本轮还没有轨迹——先问一句试试）"
    lines = [f"── 第 {t.get('turn', '?')} 轮 · {stage_label(t)} ──"]
    lines += [f"  {k}: {v}" for k, v in summarize(t).items()
              if k not in ("会话", "轮次", "分支")]
    fb = fallback_label(t)
    if fb:
        lines.append(f"  ⚠ 兜底: {fb}")
    if t.get("slots"):
        lines.append(f"  槽位: {t['slots']}")
    rows = hit_rows(t)
    if rows:
        lines.append("  召回块:")
        for h in rows:
            mark = "✓被引用" if h.get("cited") else " 未引用"
            lines.append(f"    [{h.get('rank')}] {mark} {h.get('rrf_score')} "
                         f"{h.get('doc_type')} {h.get('chunk_id')}")
    rows = tool_rows(t)
    if rows:
        lines.append("  工具调用:")
        for r in rows:
            lines.append(f"    {r.get('name')} → {r.get('status')} {r.get('message', '')}")
    return "\n".join(lines)


# ---------------- 自测：python -m app.agent.trace_view（不联网、秒级） ----------------
if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    # 1. 标签字典必须覆盖全部枚举值——漏一个前端就会显示英文原文
    assert set(STAGE_LABELS) == set(STAGES), set(STAGES) ^ set(STAGE_LABELS)
    assert set(FALLBACK_LABELS) == set(FALLBACKS), set(FALLBACKS) ^ set(FALLBACK_LABELS)
    print(f"1 契约: {len(STAGES)} 个分支 / {len(FALLBACKS)} 个兜底，标签全覆盖")

    # 2. D4：半成品轨迹不能抛——三种真实的不完整形态
    assert isinstance(render_text({}), str)                 # 还没问过
    assert isinstance(render_text(None), str)               # 显式 None
    partial = {"sid": "s1", "turn": 1, "stage": "confirmation", "gate": "reject"}
    assert isinstance(render_text(partial), str)            # 确认门：没有 intent
    empty_rag = {"sid": "s1", "turn": 2, "stage": "rag", "intent": "product_consult",
                 "collection": "product_knowledge", "hits": [], "cited": [],
                 "fallback": "empty_retrieval"}
    assert "暂无" in fallback_label(empty_rag)
    print("2 容错: 空/None/确认门(无意图)/空召回，四种半成品都能渲染")

    # 3. 摘要的缺键回落
    s = summarize(partial)
    assert s["意图"] == "（未跑意图识别）", s["意图"]
    assert s["置信度"] == "—" and s["召回块数"] == 0 and s["兜底"] == "未触发", s
    s2 = summarize(empty_rag)
    assert s2["分支"] == "RAG 检索" and s2["兜底"].startswith("检索没拿到"), s2
    print("3 摘要: 缺 key 时回落成占位符，不显示 None")

    # 4. 完整轨迹的渲染
    full = {"sid": "s1", "turn": 3, "user": "SKU-10001 口碑怎么样", "stage": "rag",
            "intent": "review_consult", "confidence": 0.95, "dissatisfied": False,
            "slots": {"product_id": "SKU-10001"},
            "collection": "review_knowledge", "query": "SKU-10001 口碑", "focus": "SKU-10001",
            "hits": [{"rank": 1, "chunk_id": "rev:SKU-10001#00", "doc_id": "rev:SKU-10001",
                      "rrf_score": 0.03, "doc_type": "review", "product_id": "SKU-10001",
                      "preview": "好评…", "len": 412, "cite_overlap": 0.41, "cited": True}],
            "cited": [True], "tool_calls": [], "elapsed_ms": 1234}
    txt = render_text(full)
    assert "第 3 轮" in txt and "RAG 检索" in txt and "✓被引用" in txt, txt
    assert summarize(full)["判定被引用"] == 1
    assert summarize(full)["置信度"] == "0.95"
    print("4 完整轨迹:")
    print("\n".join("   " + ln for ln in txt.splitlines()))

    # 5. 工具行只认 dict（轨迹是拼出来的，脏数据不该把前端搞崩）
    assert tool_rows({"tool_calls": ["坏数据", {"name": "query_price"}]}) == [
        {"name": "query_price"}]
    assert hit_rows({"hits": None}) == []
    print("5 脏数据: 非 dict 的行被过滤掉，不抛")

    print("\n自测通过。")
