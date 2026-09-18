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

    def peek(self, session_id: str) -> Session | None:
        """只读不建。**查询类调用一律用这个，不要用 get()。**

        为什么必须分开：`get()` 走的是 `setdefault`，也就是**读即写**——
        它保证"拿到的 Session 一定存在"，代价是"查一个不存在的 id 会当场把它造出来"。
        这在 Agent 主流程里是想要的（用户一开口就该有会话），但在只读端点上是个陷阱：
        `GET /sessions/{sid}` 调 `get()` 的话，查询一个拼错的 id 会返回 200 + 一个空会话，
        既撒谎又往内存里漏一个永不回收的空对象。

        两者的区别一句话：get() 是"给我这个会话，没有就新建"，
        peek() 是"如果存在就给我，否则明确告诉我没有"。
        """
        return self._sessions.get(session_id)

    def drop(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def count(self) -> int:
        """当前会话数。**只读端点要观测"客户端在造会话"时必须走这里，别碰 _sessions。**

        为什么加这个方法：加了 HTTP 之后，sid **由客户端给**（CLI 时代是进程自己生成的），
        而本类无 TTL、无上限——这是新出现的外部可控的内存增长入口，需要一个观测口。
        D1 说"接口照 Redis 的 key 设计切"，Redis 侧的对应物就是 DBSIZE；
        直接读 self._sessions 是 dict 专有写法，接 Redis 时那一处会静默失效。
        """
        return len(self._sessions)


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

    # peek：只读不建。这三条是 GET 类端点的地基，不能只测 get()
    assert store.peek("t1") is s, "已存在的会话，peek 应拿到同一个实例"
    assert store.peek("从没出现过的 id") is None, "peek 对未知 id 返回 None"
    assert "从没出现过的 id" not in store._sessions, "peek 绝不能把会话建出来（读即写陷阱）"

    store.drop("t1")
    assert store.peek("t1") is None, "drop 之后 peek 应查不到"
    assert store.get("t1").history == [], "drop 后是全新会话"
    assert store.peek("t1") is not None, "get() 之后 peek 才查得到（两者语义不同）"

    # count：/health 的 sessions 计数用它（D1：按 Redis 的 DBSIZE 设计）
    n0 = store.count()
    store.get("tc")
    assert store.count() == n0 + 1, "get 之后计数应 +1"
    store.drop("tc")
    assert store.count() == n0, "drop 之后计数应回落"

    # D5：trace 是每会话独立的报告，且不参与历史
    assert Session().trace == {}, "新会话的 trace 应是空 dict"
    a, b = store.get("ta"), store.get("tb")
    a.trace["stage"] = "rag"
    assert b.trace == {}, "两个会话的 trace 必须互不影响"
    assert all(set(m) == {"role", "content"} for m in a.history), \
        "trace 不得混进 history（history 只喂 prompt）"
    print("自测通过：同实例复用 / 10 轮成对截断 / peek 只读不建 / drop 重置 / "
          "count 计数 / trace 与会话一一对应")
