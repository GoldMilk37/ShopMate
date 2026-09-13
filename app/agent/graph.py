"""状态机主体：03 文档的七个节点 + 四条兜底，全部串起来。已实现。

模块速览：
    Agent                       状态机对象，cli/未来的 API 层只调它的 handle()
    agent                       模块级默认实例

状态机全景（03 文档 §一 的落地）：
    用户输入 → [确认门] → 意图识别 → 路由分发
      ├─ 商品咨询/口碑评价/参数对比/推荐 → RAG 检索 → 生成回复
      ├─ 订单查询/售后 → LLM 工具编排（function calling 循环）→ 生成回复
      ├─ 点名转人工 → transfer_to_human → END
      └─ 闲聊 → 直接生成回复
    四条兜底：意图置信度<0.6 追问 / 检索空结果话术 / 工具连败2次转人工 /
    连续2次不满转人工

九个设计决策（面试讲的就是这些）：
    D1 确认门在意图识别之前：pending_write 存在时，用户的下一句话只
       有两种合法含义（同意/拒绝），让 LLM 再去识别意图反而引入误判
       （"不换了"会被判成售后处理）。关键词判定 + 模糊话术回问，
       零 LLM 成本、零误路由。
    D2 工具编排放状态机不放 LLM 全权：LLM 决定"调什么、传什么参"
       （function calling），但执行必须过 executor（超时/确认门/日志
       一样不少）。写操作拿到 needs_confirmation 就中断本轮、挂起
       pending_write——「提案→确认→执行」的跨轮状态由 Session 承载。
    D3 系统发起的转人工自动确认：兜底触发的 transfer_to_human 以
       confirmed=True 直达——确认门保护的是"用户资产变更"（售后单），
       而兜底转人工本身就是保护用户的动作，再问一遍"确认转人工吗"
       是把死循环留给已经不满的用户。
    D4 param_compare/recommendation 复用 RAG 通路：对比和推荐的差异化
       全部放在生成端 prompt（"列对比表"/"按需求点推荐并说理由"），
       检索端不另建节点——03 文档把 param_compare 画成独立节点是
       预留扩展位（将来可能要先实体链接再逐商品检索），当前数据量
       下单路检索即可覆盖。
    D5 RAG 生成严格 grounding：system prompt 明令"只依据资料回答，
       资料没有就说不知道"。检索端已经把过关（阈值兜底），生成端
       不许把 85% 的召回率变成 100% 的幻觉率。
    D6 工具循环上限 3 轮：防 LLM 无限调工具烧钱。到顶就走"多次尝试
       未成功"的转人工话术，而不是硬答。
    D7 兜底动作收敛到一个 _transfer()：三种触发源（点名/连败/不满）
       复用同一段落地逻辑，summary 参数取最近对话，正好满足工具
       定义里"附最近对话摘要"的要求。
    D8 话题商品锚点驱动检索倾斜：意图抽出 product_id（或沿用 Session
       记住的上文锚点）后调用 retriever.search_with_product_focus，
       把该商品的资料重排到前面。刻意不用独占式过滤——见检索器 D8，
       那会把 product_id 为空的 faq/guide 整体排除，反而答错更多。
    D9 LLM 出网失败一律降级、不裸抛：连接类抖动（超时/断连/限流）重试
       一次，仍失败就返回 None 交给调用方，各支路给各自的兜底话术。
       不统一文案是因为"资料里没有"和"服务没连上"对用户是两回事——
       混成一句会让用户以为我们真的查不到这个商品的信息。认证/参数类
       错误不重试：重试一万次也是同样结果，只会让用户白等。

依赖链：handle ⊂ intent.classify + retrieval.search + tools.executor + llm.chat。
入口：python -m app.agent.cli（人机对话）；python -m app.agent.graph（自测）。
"""
import json
import time

from ..llm import client as llm
from ..retrieval.retriever import search as rag_search
from ..retrieval.retriever import search_with_product_focus
from ..tools.registry import get_schemas
from ..tools.executor import execute as tool_execute
from .intent import classify, IntentResult
from .session import Session, store

# 03 文档 §4 的阈值们
CLARIFY_THRESHOLD = 0.6      # 意图置信度低于此值 → 追问澄清
MAX_DISCONTENT = 2           # 连续表达不满次数 → 主动转人工
MAX_TOOL_FAILS = 2           # 工具连续失败次数 → 转人工
MAX_TOOL_ROUNDS = 3          # D6：一次用户输入里 LLM 最多编排几轮工具

# 人设（所有 LLM 调用共用的底色）
PERSONA = f"""你是电商智能客服"小搭"。今天是 {time.strftime("%Y-%m-%d")}。
语气：友好、简洁、说人话，不堆敬语。回答控制在 5 句以内，用户追问再展开。
不确定的事直说不确定，绝不编造价格、库存、政策条款。"""

# D5：RAG 生成的 grounding 约束
RAG_PROMPT = PERSONA + """

本次回答规则：只依据下方【资料】回答用户问题。资料里没有的信息就明说
"这一点我暂时没有准确信息"，并建议用户输入「转人工」由人工确认。
引用资料里的数字（价格快照、参数）要原样转述，不要四舍五入或换算。"""

# 检索空结果的兜底话术（03 文档 §4 + 01 文档 §5 兜底的合并落地）
NO_INFO_REPLY = ("这一点我暂时没有准确信息，为避免误导就不猜了。"
                 "您可以输入「转人工」，人工客服会马上帮您确认。")

# D9：LLM 本身没答上来（网络/服务问题）的兜底。刻意与 NO_INFO_REPLY 分开——
# 那句的意思是"资料里没有"，这句的意思是"我没连上"，对用户是两回事。
LLM_FAIL_REPLY = ("抱歉，我这边刚才没能连上后台服务，这个问题没能答上。"
                  "您可以再说一遍，或输入「转人工」由人工客服帮您处理。")

# 哪些异常值得重试：只认连接类抖动。不 import openai 的异常类，一是本模块
# 的离线自测至今不依赖 openai 装没装（client.py 是懒加载的），二是 httpx
# 的异常同样会冒上来，按类名一并兜住。
_RETRYABLE_HINTS = ("Timeout", "Connection", "RateLimit", "APIStatus",
                    "InternalServer", "ServiceUnavailable")


def _retryable(e: Exception) -> bool:
    """按异常类名判断是否值得重试（见 D9 的取舍说明）。"""
    name = type(e).__name__
    return any(h in name for h in _RETRYABLE_HINTS)


def _safe_llm(fn, *args, **kwargs):
    """D9：LLM 调用的统一兜底。抖动重试一次，仍失败返回 None。

    返回 None 而不是直接给话术，是因为三个调用点的降级文案不一样，
    一刀切会让用户看到文不对题的回复（见 LLM_FAIL_REPLY 的说明）。
    """
    last: Exception | None = None
    for attempt in range(2):
        try:
            return fn(*args, **kwargs)
        except Exception as e:                  # 兜底就是要兜住全部，包括 openai 的各类异常
            last = e
            if attempt == 0 and _retryable(e):
                time.sleep(0.8)                 # 只等一次，别把用户晾在这儿
                continue
            break
    print(f"[graph] LLM 调用失败，已降级: {type(last).__name__}: {last}")
    return None


# D1：确认门的关键词（判定顺序：先否后肯，"不要"必须落在否定上）
YES_WORDS = ("确认", "确定", "是的", "好的", "好", "嗯", "可以", "要", "对", "继续", "办", "申请")
NO_WORDS = ("不", "取消", "算了", "先不用", "再想想", "等等")

# D4：四个 RAG 类意图的差异化全在这张表
RAG_MODES = {
    "product_consult": ("product_knowledge", "针对用户问的具体点作答。"),
    "review_consult":  ("review_knowledge",
                        "用户在问口碑：好评和差评**两边都要讲**，不要只挑好听的。"
                        "有具体吐槽点就直说，最后给一句适用建议（什么人适合、什么人建议再想想）。"),
    "param_compare":   ("product_knowledge",
                        "用户在对比多个商品：先用资料给出逐项对比（表格或分条），再给一句倾向性建议；资料没覆盖的维度明说没有。"),
    "recommendation":  ("product_knowledge",
                        "用户在求推荐：按需求先给 1-2 个商品，每个用资料里的卖点说清'为什么适合他'，并主动问一个能缩小范围的问题。"),
}

# D8：只有「单商品」意图 + 带商品维度的库才做话题倾斜。
# 对比类天然是两个商品，政策/指南库没有 product_id，套上去只会帮倒忙。
# review_consult 也得进来：评价按 SKU 存、doc 带 product_id，是标准的
# 单商品场景。漏了它，"SKU-10001 口碑怎么样"就享受不到锚点重排。
FOCUSABLE = {"product_consult", "review_consult"}


def _anchored_product(hits: list[dict]) -> str:
    """从本轮命中里挑出话题商品，作为下一轮的锚点（下一句可以省主语）。

    只认出现 ≥2 次的商品ID：单块命中可能是"顺带提了另一个型号"的噪音，
    拿它当锚点会把后续对话引到错误商品上。取不到就返回空串——锚点宁可
    丢一次，不可锚错一次（锚错会一路把检索结果带偏）。
    """
    counts: dict[str, int] = {}
    for h in hits:
        pid = h["meta"].get("product_id") or ""
        if pid:
            counts[pid] = counts.get(pid, 0) + 1
    if not counts:
        return ""
    pid, n = max(counts.items(), key=lambda kv: kv[1])
    return pid if n >= 2 else ""


class Agent:
    """状态机。用法：agent.handle(session_id, 用户消息) → 回复文本。"""

    def __init__(self, sessions: store.__class__ = store):
        self.sessions = sessions

    # ---------------- 主入口（状态机主干） ----------------

    def handle(self, session_id: str, text: str) -> str:
        s = self.sessions.get(session_id)

        if s.pending_write:                              # D1 确认门
            return self._resolve_confirmation(s, text)

        ir = classify(text, s.history)

        if ir.dissatisfied:                              # 兜底4：连续不满
            s.dissatisfaction += 1
            if s.dissatisfaction >= MAX_DISCONTENT:
                s.dissatisfaction = 0
                return self._transfer(s, "user_dissatisfied", text,
                                      "用户连续表达不满，主动转人工")
        else:
            s.dissatisfaction = 0                        # "连续"不成立就清零

        if ir.confidence < CLARIFY_THRESHOLD:            # 兜底1：低置信度
            reply = ("抱歉，我想确认一下您的需求再回答，免得帮错忙：\n"
                     "您是想咨询商品信息、查订单/物流，还是办理退换货呢？")
            s.append_round(text, reply)
            return reply

        route = {
            "human_transfer": lambda: self._transfer(s, "user_request", text),
            "product_consult": lambda: self._answer_with_rag(s, text, ir),
            "review_consult": lambda: self._answer_with_rag(s, text, ir),
            "param_compare": lambda: self._answer_with_rag(s, text, ir),
            "recommendation": lambda: self._answer_with_rag(s, text, ir),
            "order_query": lambda: self._tool_flow(s, text),
            "after_sale": lambda: self._tool_flow(s, text),
            "chitchat": lambda: self._free_chat(s, text),
        }[ir.intent]
        return route()

    # ---------------- RAG 支路（D4/D5） ----------------

    def _answer_with_rag(self, s: Session, text: str, ir: IntentResult) -> str:
        collection, style = RAG_MODES[ir.intent]
        # 槽位进查询：对比拼 compare 列表，推荐拼 need，比裸用户话术召回准
        query = text
        if ir.intent == "param_compare" and ir.slots.get("compare"):
            query = f"{' 与 '.join(ir.slots['compare'])} 对比 参数"
        elif ir.intent == "recommendation" and ir.slots.get("need"):
            query = f"选购 {ir.slots['need']} 推荐"

        # D8 话题商品倾斜：本句抽到的型号优先，抽不到就用会话里记住的锚点
        focus = ""
        if ir.intent in FOCUSABLE:
            focus = ir.slots.get("product_id") or s.current_product_id
        hits = (search_with_product_focus(query, collection, focus)
                if focus else rag_search(query, collection))
        if not hits:                                     # 兜底2：检索空
            s.append_round(text, NO_INFO_REPLY)
            return NO_INFO_REPLY
        s.current_product_id = _anchored_product(hits) or s.current_product_id

        docs = "\n\n".join(
            f"【资料{i}】({h['meta'].get('doc_type', '')} | {h['meta'].get('product_id', '')})\n{h['text']}"
            for i, h in enumerate(hits, 1))
        messages = [
            {"role": "system", "content": f"{RAG_PROMPT}\n回答侧重：{style}"},
            *s.history,
            {"role": "user", "content": f"【资料】\n{docs}\n\n【用户问题】{text}"},
        ]
        raw = _safe_llm(llm.chat_text, messages)         # D9
        if raw is None:
            reply = LLM_FAIL_REPLY          # 压根没答上：不是"资料没有"，别说错话
        else:
            reply = raw or NO_INFO_REPLY    # 答了但是空的：按"资料没覆盖"处理
        s.append_round(text, reply)
        return reply

    # ---------------- 工具支路（D2/D6） ----------------

    def _tool_flow(self, s: Session, text: str) -> str:
        messages = [
            {"role": "system", "content": PERSONA + """
你可以调用业务工具查询实时数据（订单/物流/价格/库存）。规则：
- 用户没给订单号就直接查：不要自己填 user_id，系统会自动带上当前用户身份
- 一次回答最多调 2 个工具；查询结果原样转述，不要编造
- 退换货/维修是写操作：收集齐订单号/商品/原因后立刻调用工具，系统会
  自动向用户出示确认提示。你自己绝不在回复文本里征求确认（那会让
  系统错过确认流程），工具返回的提示原样转达即可"""},
            *s.history,
            {"role": "user", "content": text},
        ]
        for _ in range(MAX_TOOL_ROUNDS):                 # D6
            msg = _safe_llm(llm.chat, messages, tools=get_schemas())   # D9
            if msg is None:                              # 编排断了≠工具失败，不记连败
                s.append_round(text, LLM_FAIL_REPLY)
                return LLM_FAIL_REPLY
            if not msg.tool_calls:                       # 不再调工具 → 终答
                reply = msg.content or NO_INFO_REPLY
                s.append_round(text, reply)
                return reply

            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                result = tool_execute(tc.function.name, arguments,
                                      user_id="u1001")   # D2：一律过 executor
                if result.status == "needs_confirmation":    # 写操作：挂起待确认
                    s.pending_write = {"name": tc.function.name, "arguments": arguments}
                    s.append_round(text, result.message)
                    return result.message
                if result.status != "ok":                    # 兜底3：连败计数
                    s.tool_fail_streak += 1
                    if s.tool_fail_streak >= MAX_TOOL_FAILS:
                        s.tool_fail_streak = 0
                        return self._transfer(s, "out_of_scope", text,
                                              f"工具连续失败，最后错误：{result.message}")
                else:
                    s.tool_fail_streak = 0
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(
                                     {"status": result.status,
                                      "data": result.data, "message": result.message},
                                     ensure_ascii=False)})
        # 工具轮次用尽（D6）
        reply = "这个问题我尝试了几次都没查成功，直接帮您转人工处理吧。"
        s.append_round(text, reply)
        return reply

    # ---------------- 确认门（D1） ----------------

    def _resolve_confirmation(self, s: Session, text: str) -> str:
        t = text.strip()
        if any(w in t for w in NO_WORDS):
            s.pending_write = None
            reply = "好的，已为您取消本次申请。还有其他可以帮您的吗？"
        elif any(w in t for w in YES_WORDS):
            pw = s.pending_write
            s.pending_write = None
            r = tool_execute(pw["name"], pw["arguments"],
                             user_id="u1001", confirmed=True)
            if r.ok:                                     # 售后单落地信息直接模板化
                reply = (f"已受理！工单号 {r.data.get('after_sale_id', '')}，"
                         f"{r.data.get('next_step', '')}")
            else:
                reply = f"提交没有成功：{r.message} 需要我帮您转人工处理吗？"
        else:                                            # 听不懂 → 教用户怎么说
            reply = '没听清您的意思～ 请回复「确认」提交申请，或回复「取消」放弃。'
        s.append_round(text, reply)
        return reply

    # ---------------- 兜底汇聚点（D3/D7） ----------------

    def _transfer(self, s: Session, reason: str, user_text: str,
                  summary: str = "") -> str:
        """D7：三种触发源（点名/连败/不满）共用的落地动作。

        user_text 是本轮用户原话，summary 是系统发起时才有的原因说明。
        两个都要：user_text 用来写会话历史 + 拼摘要，summary 让坐席一眼
        看懂"为什么转"（连败次数、错误信息这类用户话里没有的上下文）。
        """
        # 摘要必须含本轮：本轮要等回复生成后才成对入 history，先算就等于
        # 把用户最后一句丢掉——而那句往往正是转人工的原因。
        pending = [*s.history, {"role": "user", "content": user_text}]
        recent = " / ".join(m["content"][:50] for m in pending[-6:]
                            if m["role"] == "user")
        if summary:                                     # 系统发起：原因一并带上
            recent = f"{recent}；{summary}" if recent else summary
        r = tool_execute("transfer_to_human",
                         {"reason": reason, "summary": recent, "urgency": "normal"},
                         user_id="u1001", confirmed=True)    # D3：系统发起自动确认
        if r.ok:
            reply = (f"正在为您转接人工客服（工单 {r.data.get('ticket_id', '')}，"
                     f"当前排队第 {r.data.get('queue_position', '?')} 位，"
                     f"{r.data.get('estimated_wait', '')}）。已把您的问题摘要发给坐席，不用重复描述。")
        else:
            reply = f"转人工没有成功：{r.message} 您也可以稍后再试。"
        # 转人工这一轮也必须进会话历史：之前漏了，/history 里看不到，
        # 下一轮拼上下文时用户最后一句凭空消失（P2-1）。
        s.append_round(user_text, reply)
        return reply

    # ---------------- 闲聊支路 ----------------

    def _free_chat(self, s: Session, text: str) -> str:
        messages = [{"role": "system", "content": PERSONA},
                    *s.history, {"role": "user", "content": text}]
        reply = _safe_llm(llm.chat_text, messages, temperature=0.7) or LLM_FAIL_REPLY
        s.append_round(text, reply)
        return reply


agent = Agent()


# ---------------- 自测：python -m app.agent.graph ----------------
if __name__ == "__main__":
    import os
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    # ---- 无 key 也能跑的部分：确认门（D1，纯关键词，不碰 LLM） ----
    from .session import SessionStore
    local_store = SessionStore()
    a = Agent(local_store)
    s = local_store.get("t")
    s.pending_write = {"name": "apply_after_sale",
                       "arguments": {"order_id": "ORD-20260908-002",
                                     "product_id": "SKU-30001",
                                     "type": "return", "reason": "尺码偏大"}}
    r = a.handle("t", "确认")
    assert "已受理" in r and s.pending_write is None, r
    print("确认门-同意:", r)
    s.pending_write = {"name": "apply_after_sale", "arguments": {
        "order_id": "ORD-20260908-002", "product_id": "SKU-30001",
        "type": "return", "reason": "尺码偏大"}}
    r = a.handle("t", "算了不换了")
    assert "取消" in r and s.pending_write is None, r
    print("确认门-拒绝:", r)
    s.pending_write = {"name": "apply_after_sale", "arguments": {
        "order_id": "ORD-20260908-002", "product_id": "SKU-30001",
        "type": "return", "reason": "尺码偏大"}}
    r = a.handle("t", "啊？啥意思")
    assert s.pending_write is not None, "听不懂不应消耗挂起操作"
    print("确认门-模糊:", r)

    # 系统转人工落地（D3/D7，只碰 mock 不碰 LLM）
    r = a._transfer(s, "user_request", "我要转人工", "测试摘要")
    assert "转接人工" in r
    assert s.history[-2:] == [{"role": "user", "content": "我要转人工"},
                              {"role": "assistant", "content": r}], \
        f"转人工这轮必须成对写进会话历史（P2-1），实际 {s.history[-2:]}"
    print("转人工落地 + 写历史:", r)

    # ---- LLM 故障降级（D9，不联网：用会抛异常的桩替掉真实调用） ----
    class _FakeConnTimeout(Exception):
        pass

    class _FakeAuthError(Exception):
        pass

    conn_calls: list[int] = []

    def _conn_fail(*a, **k):
        conn_calls.append(1)
        raise _FakeConnTimeout("连接超时")

    assert _safe_llm(_conn_fail) is None
    assert len(conn_calls) == 2, f"连接类异常应重试一次（共 2 次），实际 {len(conn_calls)}"

    auth_calls: list[int] = []

    def _auth_fail(*a, **k):
        auth_calls.append(1)
        raise _FakeAuthError("key 无效")

    assert _safe_llm(_auth_fail) is None
    assert len(auth_calls) == 1, f"认证类错误不该重试，实际调用 {len(auth_calls)} 次"
    print(f"LLM 降级: 抖动重试 {len(conn_calls)} 次 / 认证错误只试 1 次，均返回 None")

    # 端到端：整条闲聊支路在 LLM 挂掉时给降级话术，而不是把异常抛给用户
    orig_chat_text = llm.chat_text
    llm.chat_text = _conn_fail                            # 桩（_free_chat 在调用时才取属性）
    try:
        s2 = local_store.get("t2")
        r2 = a._free_chat(s2, "你好呀")
    finally:
        llm.chat_text = orig_chat_text
    assert r2 == LLM_FAIL_REPLY, r2
    assert s2.history[-1]["content"] == LLM_FAIL_REPLY, "降级回复也要写进历史"
    print("LLM 挂掉时的闲聊支路:", r2[:18] + "…")

    # ---- 有 key 才跑的部分：三条主流路全通 ----
    if os.environ.get("DEEPSEEK_API_KEY"):
        print("\n-- 联网冒烟（真实 LLM + 本地 RAG + mock 工具）--")
        live = Agent(SessionStore())
        for q in ("SKU-10001 这个耳机现在多少钱",           # 工具支路（价格实时）
                  "冲锋衣洗的时候有什么注意事项",             # RAG 支路（product_knowledge）
                  "SKU-10001 这耳机口碑怎么样 有什么缺点",     # RAG 支路（review_knowledge，D6）
                  "你好呀"):                                # 闲聊
            print(f"\n用户: {q}\n小搭: {live.handle('smoke', q)}")
    else:
        print("\n联网冒烟: 跳过（.env 里还没有 DEEPSEEK_API_KEY）")
    print("\n自测通过")
