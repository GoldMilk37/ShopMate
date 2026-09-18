"""ShopMate 服务层：把已有的 Agent 包成一个能被 curl / 任何程序调用的 HTTP 服务。已实现。

用法：
    python -m app.api.server --serve         # 起服务（127.0.0.1:8000，单 worker）
    python -m app.api.server --serve --port 8001
    python -m app.api.server                 # 离线自测（14 组断言，不联网、零成本）
    curl 127.0.0.1:8000/health
    # 交互式文档：浏览器打开 http://127.0.0.1:8000/docs

模块速览：
    app               FastAPI 实例（uvicorn 的目标是 app.api.server:app）
    get_agent_dep     Agent 依赖注入（自测用 dependency_overrides 换掉它）
    health / chat / get_session / get_trace / delete_session / list_tools
    _selftest()       离线自测，挂在 python -m 上

本层**不含业务逻辑**：所有分支判断仍在 graph.py / lg_graph.py 里，这里只做两件事——
把 HTTP 请求翻译成 agent.handle(sid, text) 调用，把结果翻译成 JSON。
验收标准：本文件（和整个 app/api/）出现业务 if/else 就说明逻辑漏到了错误的层。

五个设计决策：

    D1 端点一律写 `def`，**不许写 `async def`**。这是本文件最容易写错、也最容易被
       追问的一处。理由链：
         ① agent.handle 是同步阻塞的——内部是 httpx 同步出网 + BM25/向量编码的
            CPU 活，首个 RAG 轮实测 12.9 秒（webui.py D7 原话）。
         ② 写成 `async def` 意味着这个函数**跑在事件循环上**。事件循环被占住的
            12.9 秒里，连 /health 都不响应、新连接也 accept 不了——健康探针全红、
            看起来像服务挂了，**而它其实只是在正常工作**。
         ③ 写成普通 `def`，FastAPI 交给 run_in_threadpool（fastapi/routing.py 的
            `functools.partial(run_in_threadpool, func)`），事件循环空着，别的请求
            照常服务。实测：def 端点跑在 `AnyIO worker thread`，async 端点跑在
            `asyncio-portal-*` 线程（自测里没有断言它，但这是选它的全部理由）。
       这是"看起来更现代、实际更糟"的典型陷阱。
       **代价一起说**：线程池默认 40 并发 + SessionStore 无锁 = 同一个 sid 并发两个
       请求会互踩（dissatisfaction += 1 / history.append 是读-改-写）。CLI 和
       Streamlit 下几乎踩不到，加了 HTTP 之后它变成**能通过网络真踩到的竞态**。
       所以配两个约束：部署上单 worker，使用上同一个 sid 不要并发发两条。

    D2 查询类端点只用 store.peek()，**不许用 store.get()**。get() 走的是
       setdefault，也就是**读即写**——它保证"拿到的 Session 一定存在"，代价是
       "查一个不存在的 id 会当场把它造出来"。在 Agent 主流程里这是对的（用户一开口
       就该有会话），在 GET 端点上是个真陷阱：返回 200 + 一个空会话，**既撒谎又往
       内存里漏一个永不回收的空对象**（SessionStore 无 TTL、无上限）。
       自测第 3/4/5 组专钉这条：光看 404 不够，404 也可能是"先建后删"。

    D3 Agent 用 Depends 注入，不在端点里直接 get_agent()，也**不在模块级**
       `agent = get_agent()`。三个好处：
         a. 自测可替换：app.dependency_overrides[get_agent_dep] = lambda: _FakeAgent()
            就能离线跑通全部端点，不联网、不加载 BGE-M3、不花钱。
            **这条比想象的更重要**：实测 TestClient **不施加超时**（def 端点里
            sleep(7) 也正常返回 200，不走 httpx 默认的 5 秒）。意味着自测漏打桩时
            表现**不是报错，是一直挂着并且真的调用 DeepSeek 花钱**。用假 Agent 注入
            后，真 Agent 从头到尾没被构造过，这个失败模式从根上不存在。
         b. 诚实地承认"Agent 是可替换的"——runtime.py 用手写版/LG 版切换时已经在做。
         c. 不付无谓的 import 成本：模块级构造会让 `import app.api.server` 就拉起整条
            Agent import 链，/tools 这种纯读端点也被迫付。

    D4 轨迹字段用宽松的 dict，**不逐字段建模**。轨迹的键名契约在 trace_view.TRACE_KEYS
       **一处**（trace_view D1 的全部理由就是"两边各写一遍，拼错的表现是那一栏永远是
       空的"）。在 API 侧再钉一份 Pydantic 字段等于给同一个契约开第二个副本，
       将来加键要改两处。schemas.py 的 D3 同此。

    D5 "薄"没有护栏就不会薄。HTTP 一旦存在，SSE / 鉴权 / CORS / 多用户 / Redis 会
       一个接一个来，**而且每一个都有正当理由**（第一个尤其，因为首轮 13 秒确实难等）。
       所以别只把"薄"写在文档里，给它可检查的形式：3 个文件、6 个端点、
       不 import 任何业务模块，以及自测第 13 组的「路径数 == 6」——让"变厚"变成一次
       要改断言的有意识决定，而不是某次顺手。

依赖说明：自测用 fastapi.testclient.TestClient，它底层依赖 httpx（已在依赖树里，
由 openai 带入）。按 requirements.txt 文件头"只列真正 import 的包"的政策，
httpx 不单独列。
"""
import os
import sys
import time
import uuid
from pathlib import Path

# 与 webui.py D1 同款：把仓库根塞进 sys.path，让 `python -m app.api.server` 与
# `uvicorn app.api.server:app` 两种启动方式都能 import 到 app.*。
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import Depends, FastAPI, HTTPException                   # noqa: E402

from app.agent import trace_view                                      # noqa: E402
from app.agent.session import store                                   # noqa: E402
from app.api.schemas import (ChatRequest, ChatResponse, DeleteResponse,  # noqa: E402
                             HealthResponse, SessionView, ToolInfo,
                             ToolsResponse, TracePayload, TraceView)

app = FastAPI(
    title="ShopMate API",
    version="0.1.0",
    description=(
        "把 ShopMate 智能客服 Agent 包成 HTTP 接口的**薄服务层**。\n\n"
        "它最有价值的差异化资产是每轮内部决策轨迹（意图 / 召回块 / 兜底 / 工具调用），"
        "原来只在 Streamlit 侧栏里看得见，现在任何人可以 curl 到 JSON。\n\n"
        "**已知边界**：会话存进程内存，所以必须单 worker 运行（`--workers 1`），"
        "`/health` 的 pid + uptime_s 是这条约束的可检验证据；无鉴权、无限流、"
        "无 CORS、无 SSE。每个 `/chat` 是 1~4 次 DeepSeek 调用，"
        "所以默认只绑 127.0.0.1。"
    ),
)

# 进程启动时刻。**不能用 time.time() 相减**——系统时钟可能被 NTP 调整，
# 表现是 uptime 跳变或变成负数。perf_counter 单调，只回答"过去了多久"。
# 它的用处见 /health：--reload 下 pid 是恒定的（不换 worker 数），
# pid 这一项对"进程被重启过"完全看不见，只有 uptime 归零才能发现。
_STARTED_AT = time.perf_counter()


# ---------------------------------------------------------------- 依赖


def get_agent_dep():
    """D3：Agent 依赖注入。

    import 放在函数内而不是模块级：这是一个"用的时候才付钱"的开关，
    也让 `import app.api.server` 保持轻量（/tools、/docs 不需要 Agent）。
    """
    from app.agent.runtime import get_agent
    return get_agent()


def _is_lg() -> bool:
    """实际在跑的是不是 LangGraph 版。

    用**真实类名**判断，不看环境变量：环境变量填错时 runtime.get_agent() 会
    **静默**回落到手写版（runtime.py 的判据是 `in ("lg", "langgraph")`），
    照环境变量判断会得出相反结论——而"静默回落"正是需要被看见的那件事。
    实测 get_agent() 首次 0.16s、之后走模块缓存，且**不**加载 torch/chromadb。
    """
    return type(get_agent_dep()).__name__ == "LgAgent"


# ---------------------------------------------------------------- 轨迹翻译


def _trace_payload(trace: dict) -> dict:
    """把原始轨迹 dict 翻译成响应里的 trace 结构（D4：宽松 dict，不逐字段建模）。

    一切渲染都走 trace_view 的纯函数，**这里不做任何格式化**——CLI 的 /trace
    和本接口的输出因此逐字一致，两边不会各自漂移。trace_view D4 保证半成品轨迹
    （确认门那轮没有 intent、空召回那轮 hits 是空表）不会抛。
    """
    t = trace or {}
    return {
        "stage": t.get("stage", ""),
        "stage_label": trace_view.stage_label(t),
        "fallback": t.get("fallback") or None,
        "fallback_label": trace_view.fallback_label(t),
        "summary": trace_view.summarize(t),
        "hits": trace_view.hit_rows(t),
        "tools": trace_view.tool_rows(t),
        # 纯文本版，和 CLI 的 /trace 逐字一致。加它只花一行，但换来一个能直接
        # 贴给人看的证据——curl 完不用解释，对方自己看得懂。
        "text": trace_view.render_text(t),
    }


def _new_session_id() -> str:
    """服务端新建会话 id。客户端也可以自己给（见 ChatRequest.session_id）。"""
    return uuid.uuid4().hex[:8]


# ---------------------------------------------------------------- 6 个端点


@app.get("/health", response_model=HealthResponse, summary="存活 + 单 worker 证据")
def health() -> HealthResponse:
    """/health 的任务不是"保证"单 worker，而是**让违约变得可检验**。

    uvicorn 的 `--workers` 是部署参数，服务在代码里强制不了（`--workers 2` 照样跑，
    只是请求轮询到两个进程）。所以这里把违约做成现象：

      | 症状                        | 结论                                          |
      |-----------------------------|-----------------------------------------------|
      | 多次 /health 的 **pid 不同** | 起了多个 worker → 会话在进程间分裂            |
      | **uptime_s 被清零**         | 进程重启过 → 会话全丢（--reload 下只有它看得见）|
      | **sessions 只增不减**       | 无 TTL，且 HTTP 的 sid 由客户端给（新入口）     |
      | agent_env 是 "langgraphx"    | 环境变量拼错了，runtime 会静默回落到手写版      |

    ⚠ 这里**绝对不能做**的两件事（否则健康检查自己变成会失败的东西）：
      1. 不调 llm.get_client() —— 缺 key 时它 raise RuntimeError。
         只查 bool(os.environ.get("DEEPSEEK_API_KEY"))。
      2. 不调检索的 get_model() —— 那会在健康检查里加载 2GB 的 BGE-M3。
    """
    return HealthResponse(
        status="ok",
        pid=os.getpid(),
        uptime_s=round(time.perf_counter() - _STARTED_AT, 1),
        # 实际在跑的那版，不是环境变量说的那版——两者不一致正是要看见的事
        agent=type(get_agent_dep()).__name__,
        agent_env=os.environ.get("SHOPMATE_AGENT", ""),        # 原样回显，拼错一眼可见
        retriever=os.environ.get("SHOPMATE_RETRIEVER", ""),
        rerank=os.environ.get("SHOPMATE_RERANK", ""),
        telemetry=os.environ.get("SHOPMATE_TELEMETRY", ""),
        sessions=store.count(),
        llm_key_present=bool(os.environ.get("DEEPSEEK_API_KEY")),
        note="会话在进程内存：必须单 worker 运行（--workers 1）；"
             "--reload 会重启 worker 并清空会话，演示时别开。",
    )


@app.post("/chat", response_model=ChatResponse, summary="一轮对话（核心）")
def chat(req: ChatRequest, ag=Depends(get_agent_dep)) -> ChatResponse:
    """一轮对话。**必须是 def**，理由见模块 docstring 的 D1。

    ChatRequest.user_id 是预留字段，这里**故意不读它**（决策 D6）：身份一律以
    会话自身的身份为准，丢弃调用方自填的 user_id，防越权查他人订单。
    """
    # 调用方给了就用它的，没给就服务端新建并在响应里回显
    sid = req.session_id or _new_session_id()
    t0 = time.perf_counter()
    try:
        reply = ag.handle(sid, req.message)
    except Exception as e:
        # 走到这里说明状态机自己炸了（不是它内部兜住的那几种兜底）。
        # 把**已经走到哪一步的轨迹**一并带出去：刻意的，比一个干净的 500 有用。
        # 但绝不吐 traceback 原文——服务层不能把内部实现细节交给调用方。
        s = store.peek(sid)
        raise HTTPException(status_code=500, detail={
            "error": type(e).__name__,
            "message": str(e)[:300],
            "session_id": sid,
            "trace": _trace_payload((s.trace if s else {}) or {}),
        })

    # /health 的 elapsed_ms 由端点自己测（agent 的 trace 里那份同样字段是
    # 它内部测的），两者口径略有差别：这里含 HTTP 序列化开销，是调用方体感。
    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    s = store.peek(sid)
    return ChatResponse(
        session_id=sid,
        reply=reply,
        # 一问一答算一轮；handle 负责 append_round，所以这里数 history 即可
        turn=len(s.history) // 2 if s else 0,
        elapsed_ms=elapsed_ms,
        trace=TracePayload(**_trace_payload((s.trace if s else {}) or {})),
    )


@app.get("/sessions/{sid}", response_model=SessionView, summary="会话快照")
def get_session(sid: str) -> SessionView:
    """会话快照。**必须 404**，而且只能用 peek()——见模块 docstring 的 D2。"""
    s = store.peek(sid)                       # D2：只读不建，绝不是 store.get()
    if s is None:
        raise HTTPException(status_code=404, detail=(
            f"会话 {sid!r} 不存在。本服务不会为未知 id 自动创建会话——"
            "先 POST /chat 建一个（不传 session_id 会新建并在响应里回显）。"))

    # 诚实声明：lg 版下 Session 只是视图模型，只镜像 trace/history/
    # current_product_id/pending_write 四个字段（lg_graph.py D4），
    # 所以下面两个计数**恒为 0**。不许让调用方读成"用户从没不满过"。
    notes: list[str] = []
    if _is_lg():
        notes.append(
            "当前跑的是 LangGraph 版（SHOPMATE_AGENT=lg）：Session 在它下面是视图模型，"
            "dissatisfaction / tool_fail_streak 恒为 0，**不代表用户没不满过**"
            "（真实计数在 graph state 里，不进 Session）。")

    return SessionView(
        session_id=sid,
        turns=len(s.history) // 2,
        messages=len(s.history),
        pending_write=s.pending_write,
        current_product_id=s.current_product_id,
        dissatisfaction=s.dissatisfaction,
        tool_fail_streak=s.tool_fail_streak,
        notes=notes,
    )


@app.get("/sessions/{sid}/trace", response_model=TraceView, summary="最近一轮轨迹（不跑 LLM）")
def get_trace(sid: str) -> TraceView:
    """最近一轮轨迹。

    ⚠ 语义是"**最近一轮**"，不是审计日志——session.py D5 的既定取舍：轨迹是报告
    不是状态，不进 history。要看跨轮历史去 `data/logs/retrieval.jsonl`。
    本端点一次 LLM 都不调，纯读内存，是"演示内部决策"最便宜的一招。
    """
    s = store.peek(sid)                       # D2
    if s is None:
        raise HTTPException(status_code=404, detail=(
            f"会话 {sid!r} 不存在（没有轨迹可读）。"))
    t = dict(s.trace or {})
    return TraceView(
        session_id=sid,
        trace=t,
        summary=trace_view.summarize(t),
        hits=trace_view.hit_rows(t),
        tools=trace_view.tool_rows(t),
        text=trace_view.render_text(t),
    )


@app.delete("/sessions/{sid}", response_model=DeleteResponse, summary="结束会话")
def delete_session(sid: str) -> DeleteResponse:
    """结束会话，对齐 webui 侧栏的「结束会话」按钮。幂等：删不存在的也是 200。

    先 peek 再 drop 是为了回答"到底删掉了没有"——`dropped=false` 表示本来就没有。
    这里没用 store.drop() 的返回值（SessionStore.drop 现在的签名是 None），
    也没去改它：留着 Redis 版按 DEL 的语义返回删除数，那时这里可以简化。
    """
    dropped = store.peek(sid) is not None     # D2
    store.drop(sid)
    return DeleteResponse(session_id=sid, dropped=dropped)


@app.get("/tools", response_model=ToolsResponse, summary="工具清单 + 读写分级")
def list_tools() -> ToolsResponse:
    """6 个工具及其读写分级。纯读端点，不碰 Agent，也不联网。

    `is_write` 一律走 registry.is_write_op()，**不在这里重写一遍判定**——
    写操作清单是安全约束，必须由代码唯一裁决（registry D2），
    在这里复制一份等于给它开第二个副本。
    """
    from app.tools.registry import WRITE_OPS, is_write_op, load_definitions

    defs = load_definitions()
    return ToolsResponse(
        count=len(defs),
        write_ops=sorted(WRITE_OPS),
        tools=[ToolInfo(name=name,
                        description=d.get("description", ""),
                        parameters=d.get("parameters", {}),
                        returns=d.get("returns"),
                        is_write=is_write_op(name))
               for name, d in defs.items()],
    )


# ---------------------------------------------------------------- 离线自测

class _FakeAgent:
    """形状与 runtime.get_agent() 返回的一致（只有 handle 一个方法）。

    存在的全部理由见模块 docstring D3-a：真 Agent 从头到尾不会被构造，
    所以"忘了打桩 → 挂住 + 真花钱"这个失败模式从根上不存在。
    """

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail = fail

    def handle(self, session_id: str, text: str) -> str:
        self.calls.append((session_id, text))
        if self.fail:
            raise RuntimeError("模拟内部故障")
        s = store.get(session_id)              # 模仿真 Agent：会话由 handle 建
        s.trace = {"sid": session_id, "turn": len(s.history) // 2 + 1,
                   "stage": "rag", "intent": "product_consult", "confidence": 0.86,
                   "collection": "product_knowledge", "hits": [], "cited": [],
                   "tool_calls": [], "elapsed_ms": 1234}
        s.append_round(text, "（桩回复）")
        return "（桩回复）"


def _selftest() -> None:
    """离线自测：python -m app.api.server（不联网、秒级、零成本）。

    14 组断言，编号与 HANDOFF 第 8 节一一对应。先写"桩被调用过"的数量断言，
    再写其余的——否则桩没被调到的时候，后面所有断言都在验证一个没跑过的路径。
    """
    import json
    from fastapi.testclient import TestClient

    prefix = "lg-selftest-" if _is_lg() else "selftest-"
    fake = _FakeAgent()
    app.dependency_overrides[get_agent_dep] = lambda: fake
    client = TestClient(app)

    try:
        # ---- 1. /health：pid 是单 worker 声明的证据，得证明它真在 ----
        r = client.get("/health")
        assert r.status_code == 200, r.text
        h = r.json()
        assert h["pid"] == os.getpid(), h
        assert h["status"] == "ok"
        assert isinstance(h["uptime_s"], (int, float)) and h["uptime_s"] >= 0, h
        assert h["agent"] == ("LgAgent" if _is_lg() else "Agent"), h["agent"]
        assert h["sessions"] == store.count(), h
        assert isinstance(h["llm_key_present"], bool)
        print(f"1 /health: pid={h['pid']} uptime_s={h['uptime_s']} "
              f"agent={h['agent']} sessions={h['sessions']}")

        # ---- 2. /tools：复用 registry 自测那条 6 工具的口径 ----
        r = client.get("/tools")
        assert r.status_code == 200, r.text
        tj = r.json()
        from app.tools.registry import WRITE_OPS, is_write_op, load_definitions
        assert tj["count"] == 6, tj["count"]
        assert set(tj["write_ops"]) == set(WRITE_OPS), tj["write_ops"]
        names = {t["name"]: t for t in tj["tools"]}
        assert set(names) == set(load_definitions()), set(names)
        for n, t in names.items():
            assert t["is_write"] == is_write_op(n), (n, t["is_write"])
        for n in ("apply_after_sale", "transfer_to_human"):
            assert names[n]["is_write"] is True, n
        print(f"2 /tools: {tj['count']} 个工具，写操作 {tj['write_ops']}")

        # ---- 3/4/5. 读即写陷阱：三条一起钉 ----
        ghost = prefix + "从来不存在"
        assert store.peek(ghost) is None
        before = store.count()
        r = client.get(f"/sessions/{ghost}")
        assert r.status_code == 404, r.text                                   # 3
        # 光看 404 不够——404 也可能是"先建后删"（比如实现里先 get() 再 drop()）。
        # 下面这条才是真证据。
        assert store.peek(ghost) is None, "查询端点把会话建出来了（读即写陷阱）"  # 4
        assert store.count() == before, "查询端点漏了对象进内存"                 # 5
        assert client.get(f"/sessions/{ghost}/trace").status_code == 404
        print(f"3-5 读即写: GET 未知会话 404，且 store 里没有多出对象（{before} 个不变）")

        # ---- 6. POST /chat 不传 sid：请求 → handle → 响应，端到端 ----
        msg = "SKU-10001 续航多久"
        r = client.post("/chat", json={"message": msg})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["reply"] == "（桩回复）", body
        assert body["turn"] == 1, body
        sid1 = body["session_id"]
        assert sid1 and len(sid1) <= 64, sid1
        # 先钉"桩真被调过"，否则后面全是空验证
        assert fake.calls == [(sid1, msg)], fake.calls
        print(f"6 /chat: 新建 {sid1}，桩收到 {len(fake.calls)} 次调用，turn=1")

        # ---- 7. trace.stage_label 是中文且非空：证明 trace_view 真接进去了 ----
        assert body["trace"]["stage"] == "rag", body["trace"]
        assert body["trace"]["stage_label"] == "RAG 检索", body["trace"]
        assert body["trace"]["summary"]["分支"] == "RAG 检索", body["trace"]["summary"]
        assert "── 第 1 轮 · RAG 检索 ──" in body["trace"]["text"], body["trace"]["text"]
        print(f"7 trace: stage_label={body['trace']['stage_label']!r}，"
              f"text 与 CLI /trace 同源")

        # ---- 8. 同 sid 两次：会话复用，turn 累加 ----
        r2 = client.post("/chat", json={"message": "那它防水吗", "session_id": sid1})
        assert r2.status_code == 200, r2.text
        assert r2.json()["session_id"] == sid1, r2.json()
        assert r2.json()["turn"] == 2, r2.json()
        assert fake.calls[-1] == (sid1, "那它防水吗"), fake.calls
        print(f"8 复用: 同一个 sid 连问两次 → turn 累加到 2")

        # ---- 9. 不传 sid 两次：默认新建，两个 id 不同 ----
        a = client.post("/chat", json={"message": "甲"}).json()["session_id"]
        b = client.post("/chat", json={"message": "乙"}).json()["session_id"]
        assert a != b and a != sid1 and b != sid1, (a, b, sid1)
        print(f"9 新建: 两次不传 sid 拿到 {a} / {b}，互不相同")

        # ---- 10. 内部异常 → 500，且不吐 traceback ----
        app.dependency_overrides[get_agent_dep] = lambda: _FakeAgent(fail=True)
        r = client.post("/chat", json={"message": "会炸"})
        assert r.status_code == 500, r.status_code
        raw = r.text
        assert "Traceback (most recent call last)" not in raw, raw
        assert "RuntimeError" in raw, raw
        assert "模拟内部故障" in raw, raw
        assert "trace" in r.json()["detail"], r.json()
        print("10 500: 内部异常映射成 500，detail 里没有 traceback 原文")

        # ---- 11. 空 message → 422；校验在第一道关，没进到 Agent ----
        calls_before = len(fake.calls)
        assert client.post("/chat", json={"message": ""}).status_code == 422
        assert client.post("/chat", json={}).status_code == 422
        # session_id 的长度上限是外部可控的内存增长入口（decision：HTTP 的 sid
        # 由客户端给，而 SessionStore 无 TTL、无上限）。至少挡住最粗暴的一种。
        assert client.post("/chat", json={"message": "x",
                                          "session_id": "s" * 65}).status_code == 422
        assert client.post("/chat", json={"message": "x",
                                          "session_id": ""}).status_code == 422
        assert len(fake.calls) == calls_before, "非法请求不该走到 Agent"
        print("11 422: 空 message / 缺字段 / sid 超 64 字符 / sid 空串，全部挡在第一道关")

        # ---- 12. JSON-safe 承诺在 HTTP 层再钉一次 ----
        # DELETE 的序列化检查用一个临时会话，**不能拿 sid1 试**——那会把它删掉，
        # 后面第 14 组的"第一次 dropped=true"就变成了 false（踩过一次）。
        app.dependency_overrides[get_agent_dep] = lambda: fake
        throwaway = client.post("/chat", json={"message": "用来试 DELETE"}).json()["session_id"]
        for path in ("/health", "/tools", f"/sessions/{sid1}",
                     f"/sessions/{sid1}/trace", "/openapi.json"):
            json.dumps(client.get(path).json(), ensure_ascii=False)
        for resp in (client.post("/chat", json={"message": "再问一次",
                                                "session_id": sid1}),
                     client.delete(f"/sessions/{throwaway}")):
            json.dumps(resp.json(), ensure_ascii=False)
        print("12 JSON: 6 个端点的响应体都能 ensure_ascii=False 序列化")

        # ---- 13. 「薄」的护栏：操作数 == 6 ----
        # 注意 openapi 的 paths 是按**路径**分组的，不是按端点：
        # /sessions/{sid} 上的 GET + DELETE 只占一条路径。所以要数 method 不能数路径
        # （HANDOFF 第 8 节这里写的是 `len(paths) == 6`，那是把端点数当成路径数了）。
        paths = client.get("/openapi.json").json()["paths"]
        ops = {f"{m.upper()} {p}" for p, item in paths.items() for m in item
               if m in ("get", "post", "delete", "put", "patch")}
        assert len(ops) == 6, sorted(ops)
        assert set(ops) == {"GET /health", "POST /chat", "GET /sessions/{sid}",
                            "GET /sessions/{sid}/trace", "DELETE /sessions/{sid}",
                            "GET /tools"}, sorted(ops)
        assert len(paths) == 5, sorted(paths)
        print(f"13 薄: {len(ops)} 个操作 / {len(paths)} 条路径（加端点必须改这条断言）")

        # ---- 14. DELETE 幂等 ----
        r = client.delete(f"/sessions/{sid1}")
        assert r.status_code == 200, r.text
        assert r.json() == {"session_id": sid1, "dropped": True}, r.json()
        assert client.get(f"/sessions/{sid1}").status_code == 404
        r = client.delete(f"/sessions/{sid1}")
        assert r.json()["dropped"] is False, r.json()
        print("14 DELETE: 第一次 dropped=true 且随后 GET 404，第二次 dropped=false（幂等）")

        # ---- 附加：lg 版下 notes 必须在，不许让调用方误读恒为 0 的计数 ----
        s2 = client.post("/chat", json={"message": "建个会话看快照"}).json()["session_id"]
        v = client.get(f"/sessions/{s2}").json()
        assert v["turns"] == 1 and v["messages"] == 2, v
        assert v["pending_write"] is None and v["current_product_id"] == "", v
        if _is_lg():
            assert v["notes"], "lg 版下必须声明 dissatisfaction/tool_fail_streak 恒为 0"
            assert v["dissatisfaction"] == 0 and v["tool_fail_streak"] == 0, v
            print(f"15 lg 声明: notes 已带上 → {v['notes'][0][:40]}…")
        else:
            assert v["notes"] == [], "手写版没有额外前提，notes 应为空"
            print("15 快照: 手写版 notes 为空（没有额外前提）")
        client.delete(f"/sessions/{s2}")
    finally:
        app.dependency_overrides.clear()
        # 清掉自测造出来的会话，别把计数留给下一次 /health
        for sid, _ in list(fake.calls):
            store.drop(sid)

    print("\n自测通过：6 个端点 / 读即写陷阱三面钉死 / 500 不吐内部细节 / "
          "薄层护栏 6 个操作 / DELETE 幂等")


# ---------------- 入口：python -m app.api.server [--serve] [--port N] ----------------
if __name__ == "__main__":
    if "--serve" in sys.argv:
        import uvicorn
        port = 8000
        if "--port" in sys.argv:                 # 8000 常被占用，所以留一个口子
            port = int(sys.argv[sys.argv.index("--port") + 1])
        # workers=1 与 host=127.0.0.1 都写死在代码里，让"单 worker + 不对外"
        # 至少有一个入口是默认正确的。
        # host 写死 127.0.0.1 是**安全考虑**不是偷懒：每个 /chat 是 1~4 次
        # DeepSeek 调用（意图 + 生成 + 工具循环 ≤3 轮）。绑 0.0.0.0 还开着端口
        # 演示，等于把你的 key 交给同一网络里的任何人。绑 127.0.0.1 是唯一一个
        # "不需要写文档就自动成立"的约束。
        uvicorn.run("app.api.server:app", host="127.0.0.1", port=port, workers=1)
    else:
        if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8")
        _selftest()
        print("起服务：python -m app.api.server --serve    （--port 8001 换端口）")
