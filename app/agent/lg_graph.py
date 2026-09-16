"""LangGraph 版状态机：graph.py 的同任务重写（03 文档 §一的另一种实现）。已实现。

用法：
    SHOPMATE_AGENT=lg python -m app.agent.cli        # 终端（环境变量切换到本实现）
    python -m app.agent.lg_graph                     # 离线自测（不联网）

与手写版 graph.py 的映射关系（"同一任务、两种实现"对比的对照表）：

    手写版                                LangGraph 版
    ──────────────────────────────       ─────────────────────────────────
    Session.pending_write + 关键词前置门   interrupt() / Command(resume=...)
                                          ——挂起态由 checkpointer 承载，
                                          进程重启后确认门仍然有效
    handle() 里的 if/elif 路由             conditional_edges + 路由函数
    _tool_flow 的 for 循环（≤3 轮）        tool_llm ⇄ tool_exec 带计数器的环
    Session（进程内存 dict）               checkpointer（SqliteSaver 落盘）
    Session.trace + try/finally 重置       state["trace"]，ingest 节点每轮重建
    store / cli / webui                    完全复用（handle 签名不变，
                                          轨迹/历史镜像回 Session 供前端展示）

四个设计决策：
    D1 确认门的"前置性"由暂停机制本身承载：手写版把确认门放在意图识别之前，
       是为了"用户回『算了』时不让 LLM 猜意图"；本版里图在 human_gate 暂停，
       adapter 把下一句话作为 resume 值送回 human_gate——那句话根本不经过
       ingest/classify，前置性等价成立，且不再需要手写 pending 路由分支。
    D2 interrupt() 只放在 human_gate 节点顶部（任何副作用之前）：LangGraph 的
       resume 会**从头重执行该节点**，若 interrupt 前已有工具执行，重放就是
       重复副作用。写操作一律在 tool_exec 里以 confirmed=False 探路、拿到
       提案文案后存入 state，真正的执行发生在 interrupt 返回之后。
    D3 "听不懂"不丢挂起：resume 回来仍是模糊话术时，节点提交"没听清"进历史，
       条件边路由回 human_gate 自身重新 interrupt——门还关着，用户可以继续
       说"确认"或"取消"，与手写版 pending_write 保留语义一致。
    D4 Session 降级为视图模型：逻辑状态（messages/计数/pending_write/trace）
       全部住进 graph state 并由 checkpointer 持久化；adapter 每轮把最终
       state 镜像回 SessionStore 的 Session（history 截 10 轮、trace 补耗时），
       cli 的 /history、/trace 和 webui 侧栏因此一行不改。

依赖关系：复用 graph.py 的常量与提示词（PERSONA/RAG_MODES/YES_WORDS…）、
intent.classify、retriever、executor、telemetry、llm.client——出网与业务
逻辑零改动，只换编排层。
"""
import json
import sqlite3
import time
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from ..llm import client as llm
from ..retrieval import telemetry
from ..retrieval.retriever import search as rag_search
from ..retrieval.retriever import search_with_product_focus
from ..tools.executor import execute as tool_execute
from ..tools.registry import get_schemas
from .graph import (CLARIFY_THRESHOLD, FOCUSABLE, LLM_FAIL_REPLY, MAX_DISCONTENT,
                    MAX_TOOL_FAILS, MAX_TOOL_ROUNDS, NO_INFO_REPLY, NO_WORDS,
                    PERSONA, RAG_MODES, RAG_PROMPT, YES_WORDS, _anchored_product)
from .intent import IntentResult, classify
from .session import MAX_ROUNDS, store
from .trace_view import TRACE_KEYS

# checkpointer 落盘位置（.gitignore 已忽略 data/ 下运行产物）
CHECKPOINT_DB = Path(__file__).resolve().parents[2] / "data" / "graph_checkpoints.sqlite3"

TOOL_SYSTEM_PROMPT = PERSONA + """
你可以调用业务工具查询实时数据（订单/物流/价格/库存）。规则：
- 用户没给订单号就直接查：不要自己填 user_id，系统会自动带上当前用户身份
- 一次回答最多调 2 个工具；查询结果原样转述，不要编造
- 退换货/维修是写操作：收集齐订单号/商品/原因后立刻调用工具，系统会
  自动向用户出示确认提示。你自己绝不在回复文本里征求确认（那会让
  系统错过确认流程），工具返回的提示原样转达即可"""


# ---------------- State：一轮对话的全部可变状态（住进 checkpointer） ----------------

class LGState(TypedDict, total=False):
    sid: str                 # 会话 id（thread_id 同源）
    user_text: str           # 本轮用户原话
    messages: list           # OpenAI 格式历史（ingest 截 10 轮）
    trace: dict              # 本轮轨迹（键名契约同 TRACE_KEYS）
    # 意图识别产物
    intent: str
    confidence: float
    dissatisfied: bool
    slots: dict
    discontent: int          # 连续不满计数（跨轮，checkpointer 持久化）
    focus: str               # 话题商品锚点（跨轮）
    # 确认门（HITL）
    pending_write: dict | None    # {name, arguments, message=提案文案}
    gate_hint: str                # "没听清"后再 interrupt 时给用户看的话
    # 工具编排
    pending_tool_calls: list      # 本轮 LLM 决定要调的工具 [{id, name, arguments}]
    tool_rounds: int              # 本轮已编排几轮（上限 MAX_TOOL_ROUNDS）
    fail_streak: int              # 工具连败计数（跨轮）
    transfer_ctx: dict | None     # 触发转人工的上下文 {reason, user_text, summary}
    last_reply: str               # 本轮最终回复（adapter 返回给用户）


def _gate_decision(text: str) -> str:
    """手写版确认门的关键词判定，原样搬来（先否后肯：『不要』必须落在否定）。"""
    t = (text or "").strip()
    if any(w in t for w in NO_WORDS):
        return "reject"
    if any(w in t for w in YES_WORDS):
        return "accept"
    return "unclear"


def _assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


# ---------------- 节点 ----------------

def _ingest(state: LGState) -> dict:
    """每轮入口：追加用户消息、重置本轮一次性字段、重建轨迹。"""
    text = state.get("user_text", "")
    msgs = list(state.get("messages", []))
    msgs.append({"role": "user", "content": text})
    msgs = msgs[-MAX_ROUNDS * 2:]                     # 与 append_round 同口径：10 轮成对截断
    turn = (len(msgs) - 1) // 2 + 1                   # 含本轮的轮次号
    return {
        "messages": msgs,
        "trace": {"sid": state.get("sid", ""), "turn": turn, "user": text[:200]},
        # 本轮一次性字段清零（跨轮字段 discontent/fail_streak/focus/pending_write 不动）
        "last_reply": "", "tool_rounds": 0,
        "pending_tool_calls": [], "gate_hint": "", "transfer_ctx": None,
    }


def _classify(state: LGState) -> dict:
    """意图识别 + 不满计数（兜底④的判定半步，路由半步在条件边）。"""
    ir = classify(state["user_text"], state.get("messages", [])[:-1])   # 历史不含本轮，与手写版同口径
    discontent = state.get("discontent", 0)
    discontent = discontent + 1 if ir.dissatisfied else 0
    tr = {**state.get("trace", {}), "intent": ir.intent, "confidence": ir.confidence,
          "dissatisfied": ir.dissatisfied, "slots": ir.slots}
    upd: dict = {"intent": ir.intent, "confidence": ir.confidence,
                 "dissatisfied": ir.dissatisfied, "slots": ir.slots,
                 "discontent": discontent, "trace": tr}
    if discontent >= MAX_DISCONTENT:
        upd["discontent"] = 0
        upd["transfer_ctx"] = {"reason": "user_dissatisfied", "user_text": state["user_text"],
                               "summary": "用户连续表达不满，主动转人工"}
        upd["trace"] = {**tr, "fallback": "dissatisfied_transfer"}
    return upd


def _route_after_classify(state: LGState) -> str:
    """8 意图 → 节点名；兜底①④优先于意图路由（与手写版判定顺序一致）。"""
    if state.get("transfer_ctx"):
        return "transfer"
    if state.get("confidence", 0.0) < CLARIFY_THRESHOLD:
        return "clarify"
    return {"human_transfer": "transfer",
            "product_consult": "rag", "review_consult": "rag",
            "param_compare": "rag", "recommendation": "rag",
            "order_query": "tool_llm", "after_sale": "tool_llm",
            "chitchat": "chitchat"}[state.get("intent", "chitchat")]


def _clarify(state: LGState) -> dict:
    """兜底①：低置信度追问。"""
    reply = ("抱歉，我想确认一下您的需求再回答，免得帮错忙：\n"
             "您是想咨询商品信息、查订单/物流，还是办理退换货呢？")
    return {"messages": state["messages"] + [_assistant(reply)], "last_reply": reply,
            "trace": {**state["trace"], "stage": "clarify",
                      "fallback": "low_confidence_clarify"}}


def _rag(state: LGState) -> dict:
    """RAG 支路：手写版 _answer_with_rag 的逐行移植（检索词拼装/锚点/grounding/埋点）。"""
    intent = state.get("intent", "product_consult")
    slots = state.get("slots") or {}
    collection, style = RAG_MODES[intent]
    query = state["user_text"]
    if intent == "param_compare" and slots.get("compare"):
        query = f"{' 与 '.join(slots['compare'])} 对比 参数"
    elif intent == "recommendation" and slots.get("need"):
        query = f"选购 {slots['need']} 推荐"

    focus = ""
    if intent in FOCUSABLE:
        focus = slots.get("product_id") or state.get("focus", "")

    t0 = time.perf_counter()
    hits = (search_with_product_focus(query, collection, focus)
            if focus else rag_search(query, collection))
    retrieval_ms = round((time.perf_counter() - t0) * 1000)
    tr = {**state["trace"], "stage": "rag", "collection": collection,
          "query": query, "focus": focus}
    new_focus = state.get("focus", "")

    if not hits:                                     # 兜底②（这轮也落埋点）
        reply, outcome, overlaps = NO_INFO_REPLY, "no_info", []
        tr = {**tr, "fallback": "empty_retrieval"}
    else:
        new_focus = _anchored_product(hits) or new_focus
        docs = "\n\n".join(
            f"【资料{i}】({h['meta'].get('doc_type', '')} | {h['meta'].get('product_id', '')})\n{h['text']}"
            for i, h in enumerate(hits, 1))
        messages = [{"role": "system", "content": f"{RAG_PROMPT}\n回答侧重：{style}"},
                    *state["messages"][:-1],
                    {"role": "user", "content": f"【资料】\n{docs}\n\n【用户问题】{state['user_text']}"}]
        raw = llm.safe_call(llm.chat_text, messages)
        if raw is None:                              # 兜底③：不是"资料没有"，别说错话
            reply, outcome = LLM_FAIL_REPLY, "llm_fail"
            tr = {**tr, "fallback": "llm_fail"}
        else:
            reply, outcome = (raw or NO_INFO_REPLY), "answered"
        overlaps = telemetry.cite_flags(hits, reply)

    rows = telemetry.hit_rows(hits, overlaps)
    tr = {**tr, "hits": rows, "cited": [r["cited"] for r in rows], "anchor": new_focus}
    telemetry.log_retrieval(telemetry.build_record(
        sid=tr.get("sid", ""), turn=tr.get("turn", 0), intent=intent,
        collection=collection, query=query, focus=focus, outcome=outcome,
        hits=hits, overlaps=overlaps, retrieval_ms=retrieval_ms, reply=reply))
    return {"messages": state["messages"] + [_assistant(reply)], "last_reply": reply,
            "focus": new_focus, "trace": tr}


def _tool_llm(state: LGState) -> dict:
    """工具编排的 LLM 半步：决定调什么工具；不调了就出终答。"""
    tr = {**state.get("trace", {}), "stage": "tool"}
    msgs = [{"role": "system", "content": TOOL_SYSTEM_PROMPT}, *state["messages"]]
    msg = llm.safe_call(llm.chat, msgs, tools=get_schemas())
    if msg is None:                                  # 编排断了≠工具失败，不记连败
        tr = {**tr, "fallback": "llm_fail"}
        return {"messages": state["messages"] + [_assistant(LLM_FAIL_REPLY)],
                "last_reply": LLM_FAIL_REPLY, "trace": tr}
    if not msg.tool_calls:                           # 终答
        reply = msg.content or NO_INFO_REPLY
        return {"messages": state["messages"] + [_assistant(reply)],
                "last_reply": reply, "trace": tr}
    calls = []
    for tc in msg.tool_calls:
        try:
            arguments = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}
        calls.append({"id": tc.id, "name": tc.function.name, "arguments": arguments})
    msgs.append({"role": "assistant", "content": msg.content or "",
                 "tool_calls": [{"id": c["id"], "type": "function",
                                 "function": {"name": c["name"],
                                              "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                                for c in calls]})
    return {"messages": msgs, "pending_tool_calls": calls, "trace": tr}


def _route_after_tool_llm(state: LGState) -> str:
    return END if state.get("last_reply") else "tool_exec"


def _tool_exec(state: LGState) -> dict:
    """工具编排的执行半步：全部过 executor；写操作 → 挂起进 human_gate。"""
    tr = dict(state.get("trace", {}))
    tool_rows = list(tr.get("tool_calls", []))
    msgs = list(state["messages"])
    streak, rounds = state.get("fail_streak", 0), state.get("tool_rounds", 0) + 1

    for tc in state.get("pending_tool_calls", []):
        result = tool_execute(tc["name"], tc["arguments"], user_id="u1001")
        tool_rows.append({"name": tc["name"], "arguments": tc["arguments"],
                          "status": result.status, "ok": result.ok,
                          "message": result.message})
        msgs.append({"role": "tool", "tool_call_id": tc["id"],
                     "content": json.dumps({"status": result.status, "data": result.data,
                                            "message": result.message}, ensure_ascii=False)})
        if result.status == "needs_confirmation":    # 写操作：不执行，交确认门
            tr["tool_calls"] = tool_rows
            return {"messages": msgs, "trace": tr, "tool_rounds": rounds,
                    "pending_write": {"name": tc["name"], "arguments": tc["arguments"],
                                      "message": result.message}}
        if result.status != "ok":                    # 兜底③：连败计数
            streak += 1
            if streak >= MAX_TOOL_FAILS:
                tr["tool_calls"] = tool_rows
                tr["fallback"] = "tool_fail_transfer"
                return {"messages": msgs, "trace": tr, "tool_rounds": rounds,
                        "fail_streak": 0,
                        "transfer_ctx": {"reason": "out_of_scope",
                                         "user_text": state["user_text"],
                                         "summary": f"工具连续失败，最后错误：{result.message}"}}
        else:
            streak = 0

    if rounds >= MAX_TOOL_ROUNDS:                    # 兜底：编排轮次用尽
        reply = "这个问题我尝试了几次都没查成功，直接帮您转人工处理吧。"
        tr["tool_calls"] = tool_rows
        tr["fallback"] = "tool_rounds_exhausted"
        return {"messages": msgs + [_assistant(reply)], "last_reply": reply,
                "trace": tr, "tool_rounds": rounds, "fail_streak": streak}
    tr["tool_calls"] = tool_rows
    return {"messages": msgs, "trace": tr, "tool_rounds": rounds, "fail_streak": streak}


def _route_after_tool_exec(state: LGState) -> str:
    if state.get("pending_write"):
        return "human_gate"
    if state.get("transfer_ctx"):
        return "transfer"
    return END if state.get("last_reply") else "tool_llm"


def _human_gate(state: LGState) -> dict:
    """确认门（HITL）：interrupt() 挂起等用户裁决（D2：先 interrupt 后执行）。"""
    pw = state.get("pending_write")
    tr = dict(state.get("trace", {}))
    if not pw:                                       # 防御：不该到达
        return {"last_reply": "(内部状态异常：没有待确认的写操作)", "pending_write": None}
    show = state.get("gate_hint") or pw.get("message", "")
    tr.update(stage="confirmation")

    ans = interrupt({"reply": show})                 # ★ 暂停点：进程重启后仍在此等
    decision = _gate_decision(ans if isinstance(ans, str) else "")

    if decision == "reject":
        reply = "好的，已为您取消本次申请。还有其他可以帮您的吗？"
        tr["gate"] = "reject"
        return {"messages": state["messages"] + [{"role": "user", "content": ans},
                                                 _assistant(reply)],
                "last_reply": reply, "pending_write": None, "gate_hint": "", "trace": tr}
    if decision == "accept":
        tr["gate"] = "accept"
        r = tool_execute(pw["name"], pw["arguments"], user_id="u1001", confirmed=True)
        tr.setdefault("tool_calls", []).append(
            {"name": pw["name"], "arguments": pw["arguments"],
             "status": r.status, "ok": r.ok, "message": r.message})
        if r.ok:
            reply = (f"已受理！工单号 {r.data.get('after_sale_id', '')}，"
                     f"{r.data.get('next_step', '')}")
        else:
            reply = f"提交没有成功：{r.message} 需要我帮您转人工处理吗？"
        return {"messages": state["messages"] + [{"role": "user", "content": ans},
                                                 _assistant(reply)],
                "last_reply": reply, "pending_write": None, "gate_hint": "", "trace": tr}
    # 听不懂：提交进历史、留在门里继续等（D3）
    reply = "没听清您的意思～ 请回复「确认」提交申请，或回复「取消」放弃。"
    tr["gate"] = "unclear"
    return {"messages": state["messages"] + [{"role": "user", "content": ans},
                                             _assistant(reply)],
            "last_reply": reply, "gate_hint": reply, "trace": tr}


def _route_after_gate(state: LGState) -> str:
    """还有待确认操作（unclear）→ 回门口继续 interrupt；已解决 → 结束。"""
    return "human_gate" if state.get("pending_write") else END


def _transfer(state: LGState) -> dict:
    """兜底汇聚点：三种触发源（点名/连败/不满）共用，附对话摘要。"""
    ctx = state.get("transfer_ctx") or {"reason": "user_request",
                                        "user_text": state.get("user_text", ""), "summary": ""}
    tr = dict(state.get("trace", {}))
    tr.update(stage="transfer", transfer_reason=ctx["reason"])
    recent = " / ".join(m["content"][:50] for m in state["messages"][-6:]
                        if m["role"] == "user")
    if ctx.get("summary"):
        recent = f"{recent}；{ctx['summary']}" if recent else ctx["summary"]
    r = tool_execute("transfer_to_human",
                     {"reason": ctx["reason"], "summary": recent, "urgency": "normal"},
                     user_id="u1001", confirmed=True)    # 系统发起自动确认
    tr.setdefault("tool_calls", []).append(
        {"name": "transfer_to_human", "arguments": {"reason": ctx["reason"]},
         "status": r.status, "ok": r.ok, "message": r.message})
    if r.ok:
        reply = (f"正在为您转接人工客服（工单 {r.data.get('ticket_id', '')}，"
                 f"当前排队第 {r.data.get('queue_position', '?')} 位，"
                 f"{r.data.get('estimated_wait', '')}）。已把您的问题摘要发给坐席，不用重复描述。")
    else:
        reply = f"转人工没有成功：{r.message} 您也可以稍后再试。"
    return {"messages": state["messages"] + [_assistant(reply)], "last_reply": reply,
            "transfer_ctx": None, "trace": tr}


def _chitchat(state: LGState) -> dict:
    tr = {**state.get("trace", {}), "stage": "chitchat"}
    reply = llm.safe_call(llm.chat_text,
                          [{"role": "system", "content": PERSONA}, *state["messages"]],
                          temperature=0.7) or LLM_FAIL_REPLY
    if reply == LLM_FAIL_REPLY:
        tr["fallback"] = "llm_fail"
    return {"messages": state["messages"] + [_assistant(reply)], "last_reply": reply,
            "trace": tr}


# ---------------- 图装配 ----------------

def build_graph(checkpointer):
    """装配 StateGraph。checkpointer 必传——interrupt() 依赖它持久化暂停点。"""
    g = StateGraph(LGState)
    for name, fn in (("ingest", _ingest), ("classify", _classify), ("clarify", _clarify),
                     ("rag", _rag), ("tool_llm", _tool_llm), ("tool_exec", _tool_exec),
                     ("human_gate", _human_gate), ("transfer", _transfer),
                     ("chitchat", _chitchat)):
        g.add_node(name, fn)
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "classify")
    g.add_conditional_edges("classify", _route_after_classify)
    g.add_edge("clarify", END)
    g.add_edge("rag", END)
    g.add_edge("chitchat", END)
    g.add_edge("transfer", END)
    g.add_conditional_edges("tool_llm", _route_after_tool_llm)
    g.add_conditional_edges("tool_exec", _route_after_tool_exec)
    g.add_conditional_edges("human_gate", _route_after_gate)
    return g.compile(checkpointer=checkpointer)


def get_lg_graph():
    """进程级默认图：SqliteSaver 落盘（断电/重启后确认门仍可续）。"""
    global _default_graph
    if _default_graph is None:
        CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
        _default_graph = build_graph(SqliteSaver(conn))
    return _default_graph


_default_graph = None


# ---------------- adapter：对 cli/webui 暴露与手写版一致的 handle() ----------------

class LgAgent:
    """与 graph.Agent 同签名的入口。handle(sid, text) → 回复文本（D4：镜像回 Session）。"""

    def __init__(self, graph) -> None:
        self.graph = graph

    def handle(self, session_id: str, text: str) -> str:
        cfg = {"configurable": {"thread_id": session_id}}
        paused = bool(self.graph.get_state(cfg).next)      # 暂停中=确认门在等回答（D1）
        t0 = time.perf_counter()
        if paused:
            result = self.graph.invoke(Command(resume=text), cfg)
        else:
            result = self.graph.invoke({"user_text": text, "sid": session_id}, cfg)

        reply = result.get("last_reply", "")
        intr = result.get("__interrupt__")
        if intr:                                           # 本轮以挂起收尾：回复取 interrupt 载荷
            val = getattr(intr[0], "value", intr[0])
            if isinstance(val, dict):
                reply = str(val.get("reply", "")) or reply

        # D4：镜像给前端/cli 的视图层（逻辑真身在 graph state）
        s = store.get(session_id)
        tr = dict(result.get("trace") or {})
        tr["elapsed_ms"] = round((time.perf_counter() - t0) * 1000)
        s.trace = tr
        s.history = list(result.get("messages", []))[-MAX_ROUNDS * 2:]
        s.current_product_id = result.get("focus", "")
        s.pending_write = result.get("pending_write")
        return reply


lg_agent = LgAgent(get_lg_graph())


# ---------------- 自测：python -m app.agent.lg_graph（不联网） ----------------
if __name__ == "__main__":
    import os
    import sys
    import tempfile
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.checkpoint.sqlite import SqliteSaver as _SqlSaver

    def _patch(**kw):
        """换掉本模块全局名，返回还原函数（以 __main__ 运行必须打在 globals 上）。"""
        saved = {k: globals()[k] for k in kw}
        globals().update(kw)
        return lambda: globals().update(saved)

    # ---- 1. 低置信澄清 ----
    restore = _patch(classify=lambda text, history=None:
                     IntentResult("product_consult", 0.2, False, {}))
    try:
        a = LgAgent(build_graph(MemorySaver()))
        r = a.handle("t1", "那个……")
    finally:
        restore()
    assert "想确认" in r, r
    assert a.graph.get_state({"configurable": {"thread_id": "t1"}}).values["trace"]["fallback"] \
        == "low_confidence_clarify"
    print("1 低置信澄清:", r[:20] + "…")

    # ---- 2. RAG 支路（桩掉意图/检索/生成，验轨迹+埋点） ----
    anchor = "【SKU-10001 无线降噪蓝牙耳机 Pro | SoundCore】"
    fake_hits = [
        {"chunk_id": "rev:SKU-10001#00", "text": anchor + "好评集中在降噪和续航。",
         "score": 0.031, "meta": {"doc_type": "review", "product_id": "SKU-10001"}},
        {"chunk_id": "rev:SKU-10001#01", "text": anchor + "差评说充电盒偏大，装兜里硌。",
         "score": 0.028, "meta": {"doc_type": "review", "product_id": "SKU-10001"}},
    ]
    fake_reply = "差评主要是充电盒偏大，装兜里会硌。"
    captured: list[dict] = []
    saved_log = telemetry.log_retrieval
    telemetry.log_retrieval = lambda rec, **k: captured.append(rec) or True
    restore = _patch(
        classify=lambda text, history=None:
            IntentResult("review_consult", 0.95, False, {"product_id": "SKU-10001"}),
        search_with_product_focus=lambda q, c, p, top_k=5: fake_hits,
        rag_search=lambda q, c, top_k=5: fake_hits,
    )
    saved_chat = llm.chat_text
    llm.chat_text = lambda *a, **k: fake_reply
    try:
        a2 = LgAgent(build_graph(MemorySaver()))
        r2 = a2.handle("t2", "SKU-10001 这耳机口碑怎么样")
    finally:
        restore()
        llm.chat_text = saved_chat
        telemetry.log_retrieval = saved_log
    assert r2 == fake_reply and len(captured) == 1 and captured[0]["outcome"] == "answered"
    tr2 = a2.graph.get_state({"configurable": {"thread_id": "t2"}}).values["trace"]
    assert tr2["stage"] == "rag" and tr2["collection"] == "review_knowledge"
    assert tr2["cited"] == [False, True] and tr2["anchor"] == "SKU-10001"
    assert set(tr2) <= set(TRACE_KEYS), f"未收录键: {set(tr2) - set(TRACE_KEYS)}"
    print(f"2 RAG 支路: 召回 {len(tr2['hits'])} 块，引用 {tr2['cited']}，埋点 1 条")

    # ---- 3. 确认门全流程：interrupt 挂起 → accept/reject/unclear ----
    class _F:                                              # 假 tool_calls 协议对象
        def __init__(self, i, name, args):
            self.id, self.content = f"call_{i}", ""
            class _Fn:
                pass
            self.function = _Fn()
            self.function.name, self.function.arguments = name, json.dumps(args, ensure_ascii=False)
    class _Msg:
        content, tool_calls = "", None

    def _to_order_flow(a: LgAgent, sid: str):
        """桩掉意图识别与 LLM 编排：直接决定调 apply_after_sale（真实 executor 会拦下要确认）。"""
        m = _Msg()
        m.tool_calls = [_F(1, "apply_after_sale",
                           {"order_id": "ORD-20260908-002", "product_id": "SKU-30001",
                            "type": "return", "reason": "尺码偏大"})]
        saved_chat, saved_cls = llm.chat, classify
        llm.chat = lambda *a, **k: m
        globals()["classify"] = lambda text, history=None: \
            IntentResult("after_sale", 0.95, False, {"order_id": "ORD-20260908-002"})
        try:
            return a.handle(sid, "ORD-20260908-002 的冲锋衣尺码偏大，我要退货")
        finally:
            llm.chat, globals()["classify"] = saved_chat, saved_cls

    a3 = LgAgent(build_graph(MemorySaver()))
    r3 = _to_order_flow(a3, "t3")
    assert "确认提交吗" in r3, r3                            # 首轮以 interrupt 挂起，回复=提案
    assert a3.graph.get_state({"configurable": {"thread_id": "t3"}}).next, "应在 human_gate 暂停"
    r3b = a3.handle("t3", "确认")                            # resume=确认 → 真执行
    assert "已受理" in r3b and "AS-" in r3b, r3b
    tr3 = a3.graph.get_state({"configurable": {"thread_id": "t3"}}).values["trace"]
    assert tr3["gate"] == "accept" and tr3["stage"] == "confirmation"
    assert not a3.graph.get_state({"configurable": {"thread_id": "t3"}}).next
    print(f"3a 确认门-accept: {r3b[:24]}…  gate={tr3['gate']}")

    a3r = LgAgent(build_graph(MemorySaver()))
    _to_order_flow(a3r, "t3r")
    r3r = a3r.handle("t3r", "算了不换了")
    assert "取消" in r3r, r3r
    print("3b 确认门-reject:", r3r[:20] + "…")

    a3u = LgAgent(build_graph(MemorySaver()))
    _to_order_flow(a3u, "t3u")
    r3u = a3u.handle("t3u", "啊？啥意思")                    # unclear：门不丢
    assert "没听清" in r3u and a3u.graph.get_state({"configurable": {"thread_id": "t3u"}}).next
    r3u2 = a3u.handle("t3u", "确认")                         # 还能继续确认
    assert "已受理" in r3u2, r3u2
    print("3c 确认门-unclear→accept: 门保持挂起，后续仍可确认")

    # ---- 4. ★ 断点续跑：SqliteSaver 落盘，换进程（换图实例）仍能续 ----
    # Windows 教训：sqlite 文件被打开时删不掉，临时目录清理会炸——
    # 连接用完必须 close；ignore_cleanup_errors 再兜一层。
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        db = Path(d) / "cp.sqlite3"
        con1 = sqlite3.connect(db, check_same_thread=False)
        g1 = build_graph(_SqlSaver(con1))
        a4 = LgAgent(g1)
        _to_order_flow(a4, "t4")
        assert g1.get_state({"configurable": {"thread_id": "t4"}}).next
        del g1, a4                                          # 模拟进程死掉
        con1.close()                                        # 文件句柄放手，目录才删得掉
        con2 = sqlite3.connect(db, check_same_thread=False)
        g2 = build_graph(_SqlSaver(con2))
        snap = g2.get_state({"configurable": {"thread_id": "t4"}})
        assert snap.next, "换实例后暂停点应仍在（checkpointer 持久化）"
        a4b = LgAgent(g2)
        r4 = a4b.handle("t4", "确认")                        # 新进程里 resume
        assert "已受理" in r4, r4
        con2.close()
    print("4 断点续跑: 进程重启后 interrupt 暂停点仍在，resume 成功（手写版做不到）")

    # ---- 5. 连续不满 → 主动转人工 ----
    calls = {"n": 0}

    def _grumpy(text, history=None):
        calls["n"] += 1
        return IntentResult("chitchat", 0.9, True, {})      # 每轮都判不满

    restore = _patch(classify=_grumpy)
    saved_chat = llm.chat_text
    llm.chat_text = lambda *a, **k: "非常抱歉给您添堵了！"
    try:
        a5 = LgAgent(build_graph(MemorySaver()))
        a5.handle("t5", "什么破东西")
        r5 = a5.handle("t5", "等了三天还没到，火大")
    finally:
        restore()
        llm.chat_text = saved_chat
    assert "转接人工" in r5, r5
    tr5 = a5.graph.get_state({"configurable": {"thread_id": "t5"}}).values["trace"]
    assert tr5["fallback"] == "dissatisfied_transfer"
    print("5 连续不满×2:", r5[:22] + "…")

    # ---- 6. 契约：history 只有 role/content；镜像是视图 ----
    for sid, ag in (("t2", a2), ("t3", a3)):
        msgs = ag.graph.get_state({"configurable": {"thread_id": sid}}).values["messages"]
        assert all(set(m) <= {"role", "content", "tool_calls", "tool_call_id", "id",
                              "type", "function"} for m in msgs), msgs
    s = store.get("t3")
    assert s.trace.get("gate") == "accept" and "elapsed_ms" in s.trace
    assert any(m["content"] == "确认" for m in s.history)
    print("6 契约: 轨迹键合法 / Session 镜像（trace+history）完整")

    # ---- 7. 有 key 才跑的联网冒烟 ----
    if os.environ.get("DEEPSEEK_API_KEY"):
        print("\n-- 联网冒烟（真实 LLM）--")
        live = LgAgent(build_graph(MemorySaver()))
        for q in ("SKU-10001 这个耳机现在多少钱",
                  "冲锋衣洗的时候有什么注意事项",
                  "你好呀"):
            print(f"\n用户: {q}\n小搭: {live.handle('smoke', q)}")
    else:
        print("\n联网冒烟: 跳过（无 DEEPSEEK_API_KEY）")
    print("\n自测全部通过")
