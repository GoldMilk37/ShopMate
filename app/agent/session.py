"""会话记忆：按 session_id 存最近 10 轮对话 + 跨轮状态。已实现。

模块速览：
    Session            一个会话的全部状态（历史/不满计数/待确认写操作/工具连败）
    SessionStore       {session_id: Session}，进程内版
    store              模块级默认实例（graph 直接用）

三个设计决策：
    D1 接口照 Redis 的 key 设计切（04 文档 chat:ctx:{session_id}）：
       get(sid) / append(sid, user, assistant)，将来接 Redis 只换
       这两个方法的内部实现，Agent 代码一行不动。TTL 30min 暂不做
       ——进程内 dict 会话本身活不过进程重启，等价于无限期。
    D2 10 轮截断在"轮"不在"条"：一问一答算一轮，裁的时候成对裁，
       不会把孤儿 user 消息留在开头让 LLM 以为用户自言自语。
    D3 跨轮状态放 Session 而不是全局：不满计数（03 文档"连续 2 次"）、
       写操作待确认（executor 的 needs_confirmation 跨轮兑现）、工具
       连败计数，都是"这一个用户这一次会话"的事，换会话自动归零。
    D4 话题商品锚点 current_product_id：跨记住"我们现在在聊哪件商品"，
       这样下一句省主语的追问（"那它防水吗"）也能做 product_id 槽位
       过滤。它属于"这一次会话"，寄存在全局会让两个用户串件。
    D5 trace 是"报告"不是"状态"：它记录本轮走到了哪个分支、意图置信度、
       召回了哪些块，供前端展示与检索埋点使用。三条纪律：
       一是**不进 prompt**——history 仍是唯一喂给 classify 和 messages 的
       东西，混进去会让 LLM 看到自己的诊断信息；二是**不进将来的 Redis
       chat:ctx 序列化**（04 文档），它是可再生的观测数据，没必要跟着会话
       持久化；三是每轮由 Agent.handle **重新绑定**而非原地 clear——前端
       上一轮抓着的引用要还能指向那份完整的旧轨迹。字段本身用 dict 而不是
       数据类：加一个键不该要改 N 处，且前端要的本来就是可序列化的 dict。

历史格式就是 OpenAI messages（{"role","content"}），可直接拼进请求。
"""
from dataclasses import dataclass, field

# 04 文档：短期记忆最近 10 轮
MAX_ROUNDS = 10


@dataclass
class Session:
    """一个会话的全部可变状态。"""
    history: list[dict] = field(default_factory=list)   # OpenAI messages 格式
    dissatisfaction: int = 0       # 连续表达不满的次数（03 文档 §4）
    pending_write: dict | None = None   # 待用户确认的写操作 {name, arguments}
    tool_fail_streak: int = 0     # 工具连续失败次数（03 文档 §4）
    current_product_id: str = ""  # 当前话题商品（槽位过滤的锚点，见 D4）
    trace: dict = field(default_factory=dict)   # 本轮诊断轨迹（D5：报告，非跨轮状态）

    def append_round(self, user: str, assistant: str) -> None:
        """D2：成对追加并按轮截断。"""
        self.history.append({"role": "user", "content": user})
        self.history.append({"role": "assistant", "content": assistant})
        if len(self.history) > MAX_ROUNDS * 2:
            self.history = self.history[-MAX_ROUNDS * 2:]


class SessionStore:
    """D1：进程内实现，方法签名按将来的 Redis 版设计。"""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def get(self, session_id: str) -> Session:
        return self._sessions.setdefault(session_id, Session())

    def drop(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


# graph 默认用的实例
store = SessionStore()


# ---------------- 自测：python -m app.agent.session ----------------
if __name__ == "__main__":
    s = store.get("t1")
    assert store.get("t1") is s, "同 id 应拿到同一实例"

    for i in range(15):                                  # 灌 15 轮
        s.append_round(f"问{i}", f"答{i}")
    assert len(s.history) == 20, f"应截到 10 轮 20 条，实际 {len(s.history)}"
    assert s.history[0]["content"] == "问5", "截断应成对裁掉最老的 5 轮"
    assert all(m["role"] in ("user", "assistant") for m in s.history)

    store.drop("t1")
    assert store.get("t1").history == [], "drop 后是全新会话"

    # D5：trace 是每会话独立的报告，且不参与历史
    assert Session().trace == {}, "新会话的 trace 应是空 dict"
    a, b = store.get("ta"), store.get("tb")
    a.trace["stage"] = "rag"
    assert b.trace == {}, "两个会话的 trace 必须互不影响"
    assert all(set(m) == {"role", "content"} for m in a.history), \
        "trace 不得混进 history（history 只喂 prompt）"
    print("自测通过：同实例复用 / 10 轮成对截断 / drop 重置 / trace 与会话一一对应")
