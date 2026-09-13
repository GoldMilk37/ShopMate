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
    print("自测通过：同实例复用 / 10 轮成对截断 / drop 重置")
