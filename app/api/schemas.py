"""服务层的请求/响应模型（Pydantic）。已实现。

模块速览：
    ChatRequest / ChatResponse    一轮对话
    TracePayload                  轨迹（由 trace_view 的纯函数组装）
    HealthResponse                存活 + 单 worker 证据
    SessionView / TraceView       读会话状态与轨迹
    ToolInfo / ToolsResponse      工具清单
    DeleteResponse                结束会话

三个设计决策：
    D1 响应模型全部显式声明字段，不用裸 dict：FastAPI 会照它生成 /docs 交互文档，
       而这份文档就是"这个服务能干什么"的说明书。裸 dict 生成出来的文档是空的。
    D2 description 一律写中文，且**写清楚没做到的部分**。比如 ChatRequest.user_id
       是预留字段、当前不生效——这件事必须写在字段说明里，而不是藏在文档角落。
       调用方看 /docs 时看到的就是这一行，它是最不容易被漏掉的位置。
    D3 轨迹相关的字段一律用宽松类型（dict / list[dict] / str | None），不钉死结构：
       轨迹是"走到哪记到哪"，确认门那一轮根本没有 intent，空召回那轮 hits 是空表
       （见 trace_view D4）。用严格模型会在半成品轨迹上直接 500——把"这轮没有这个
       数据"误报成"服务坏了"。
"""
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------- 对话


class ChatRequest(BaseModel):
    message: str = Field(description="用户这一轮说的话", min_length=1)
    session_id: str | None = Field(
        default=None, min_length=1, max_length=64,
        description="会话 ID。不传则新建一个并在响应里回传；之后带上它即可延续上下文。"
                    "**上限 64 字符不是装饰**：CLI 时代的 sid 是进程自己生成的，"
                    "而 HTTP 的 sid 由客户端给——这是新出现的、外部可控的内存增长入口"
                    "（SessionStore 无 TTL、无上限）。只挡长度不挡数量，"
                    "但至少挡住最粗暴的一种")
    user_id: str | None = Field(
        default=None,
        description="**预留字段，当前不生效**——传了也会被忽略。身份一律以会话身份覆盖"
                    "（决策 D6：丢弃模型/调用方自填的 user_id，防越权查他人订单）")


class TracePayload(BaseModel):
    """本轮内部决策轨迹。这是本项目区别于"套壳 ChatGPT"的那部分。"""
    stage: str = Field(description="本轮走到哪个分支（英文枚举值，见 trace_view.STAGES）")
    stage_label: str = Field(description="分支的中文名")
    fallback: str | None = Field(default=None, description="命中的兜底枚举值；本轮没走兜底则为 null")
    fallback_label: str | None = Field(default=None, description="兜底的中文说明，含为什么触发")
    summary: dict[str, Any] = Field(
        description="标量摘要（中文键）。缺数据的项回落成占位符（如「（未跑意图识别）」）而不是 null")
    hits: list[dict[str, Any]] = Field(
        default_factory=list,
        description="本轮召回的块，preview 已截断。**全文与跨轮历史在 data/logs/retrieval.jsonl**")
    tools: list[dict[str, Any]] = Field(
        default_factory=list, description="本轮的工具调用（名称/状态/是否成功/消息）")
    text: str = Field(
        description="纯文本版轨迹，与 CLI 的 `/trace` 命令**逐字一致**"
                    "（同为 trace_view.render_text 的输出）。加它只花一行，"
                    "但换来一个能直接贴给人看的证据")


class ChatResponse(BaseModel):
    session_id: str = Field(description="本次会话 ID；请求没带就是新建的，请自行保管")
    reply: str = Field(description="Agent 的回复全文")
    turn: int = Field(description="这是该会话的第几轮（一问一答算一轮）")
    elapsed_ms: int = Field(
        description="本轮耗时（毫秒）。首次检索要加载向量模型，实测万级；之后百毫秒级")
    trace: TracePayload


# ---------------------------------------------------------------- 健康检查


class HealthResponse(BaseModel):
    """存活 + **单 worker 的证据链**。

    单 worker 是部署参数，服务在代码里强制不了（`--workers 2` 照样跑，只是请求
    轮询到两个进程）。所以 /health 的任务不是"保证"，而是**让违约变得可检验**：
    每一条证据都对应一个可操作的判据（见 server.py 的 health() docstring）。
    """
    status: str = Field(default="ok")
    pid: int = Field(
        description="当前进程号。**这是「必须单 worker」的核心证据**：多次请求拿到"
                    "不同的 pid 就说明起了多个 worker，同一会话被分到不同进程、"
                    "各拿一个空会话（表现是用户「聊着聊着记忆没了」）")
    uptime_s: float = Field(
        description="进程已运行秒数。**只有它能发现 --reload 造成的重启**——"
                    "reload 不换 worker 数（pid 恒定），pid 这一项对它完全看不见，"
                    "但它会把内存里的会话全部清空。uptime 被清零 = 会话全丢")
    agent: str = Field(
        description="**实际在跑**的那版实现的类名（Agent / LgAgent），"
                    "取自 type(get_agent()).__name__，不是读环境变量")
    agent_env: str = Field(
        description="SHOPMATE_AGENT 的**原值，原样回显**（空=用手写版 graph.py）。"
                    "拼成 langgraphx 之类时 runtime 会**静默**回落到手写版"
                    "（runtime.py 只认 lg / langgraph）——原样回显才看得见。"
                    "与上面 agent 字段不一致就说明环境变量拼错了")
    retriever: str = Field(description="SHOPMATE_RETRIEVER 环境变量的原值（空=用手写版 retriever.py）")
    rerank: str = Field(description="SHOPMATE_RERANK 环境变量的原值（1/true/yes 才启用重排）")
    telemetry: str = Field(description="SHOPMATE_TELEMETRY 环境变量的原值（0 时检索埋点整体静默）")
    sessions: int = Field(
        description="进程内的会话数。**只增不减 = 客户端在造会话**（SessionStore 无 TTL、"
                    "无上限，且 HTTP 让 sid 由客户端给）。这是唯一的观测口")
    llm_key_present: bool = Field(
        description="**只查环境变量，绝不建客户端**——llm.get_client() 缺 key 时会 "
                    "raise RuntimeError，那会让健康检查自己变成会失败的东西")
    note: str = Field(default="", description="部署约束的原文声明")


# ---------------------------------------------------------------- 会话


class SessionView(BaseModel):
    session_id: str
    turns: int = Field(description="记忆里的轮数。**上限 10 轮**（MAX_ROUNDS），超出成对裁掉最老的")
    messages: int = Field(description="history 里的消息条数 = 轮数 × 2（一问一答）")
    pending_write: dict[str, Any] | None = Field(
        default=None, description="待用户确认的写操作。确认门的状态挂在这里，非 null 表示卡在确认门")
    current_product_id: str = Field(default="", description="当前话题商品锚点；空串表示还没锚定")
    dissatisfaction: int = Field(default=0, description="连续表达不满的次数（到 2 转人工）")
    tool_fail_streak: int = Field(default=0, description="工具连续失败次数")
    notes: list[str] = Field(
        default_factory=list, description="读上面这些字段时要知道的前提；空表示没有额外前提")


class TraceView(BaseModel):
    session_id: str
    trace: dict[str, Any] = Field(
        default_factory=dict,
        description="原始轨迹 dict（键名契约见 trace_view.TRACE_KEYS）。还没聊过时是空 dict")
    summary: dict[str, Any] = Field(default_factory=dict)
    hits: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    text: str = Field(description="纯文本版，与 CLI 的 /trace 命令输出一致")


# ---------------------------------------------------------------- 工具


class ToolInfo(BaseModel):
    name: str
    description: str = Field(default="")
    parameters: dict[str, Any] = Field(default_factory=dict, description="JSON Schema 形式的入参定义")
    returns: dict[str, Any] | None = Field(default=None, description="返回结构说明；给模型的那份会剥掉它")
    is_write: bool = Field(description="写操作（不可逆，必须过确认门）；未知工具按最坏情况算 True")


class ToolsResponse(BaseModel):
    count: int
    write_ops: list[str] = Field(description="需要确认门的写操作清单（决策 D2：硬编码在代码里，不读 JSON）")
    tools: list[ToolInfo]


class DeleteResponse(BaseModel):
    session_id: str
    dropped: bool = Field(description="true=确实删掉了一个会话；false=本来就没有（DELETE 是幂等的）")
