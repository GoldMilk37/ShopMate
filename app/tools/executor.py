"""工具层第 3 步：统一执行入口（02 文档"调用原则"的落地）。已实现。

模块速览：
    execute(name, arguments, user_id, confirmed)   Agent 状态机只调这个
    ToolResult                                      每次调用的统一结果

六个设计决策（02 文档四条调用原则怎么变成代码）：
    D1 写操作确认门：is_write_op 且未 confirmed → 不执行，返回
       status="needs_confirmation" + 给用户看的提案文案。状态机拿它去问
       用户，用户点头后带 confirmed=True 重调（「提案→确认→执行」）。
       只读工具直接放行。
    D2 3s 超时：mock 都是瞬回，真实业务系统会挂。线程池 + future.result
       (timeout=3)，超时不杀线程（Python 做不到）但结果作废，返回
       status="timeout" 的降级话术——上层不必区分超时和失败。
    D3 结果三态：ok / error（业务查不到，ToolInputError）/ timeout。
       error 和 timeout 也算一次"完成"的调用，结构永远完整，LLM 侧
       拿到的不会是裸异常。连续 2 次非 ok 由状态机数（03 文档 §4）。
    D4 日志落 data/logs/tool_calls.jsonl：每行一条 JSON（02 文档要求
       的用户ID/工具名/参数/结果/耗时 + 时间戳）。jsonl 追加写天然
       并发友好，也是将来灌 MySQL tool_call_logs 表的现成格式。
    D5 只读工具 5min 结果缓存（04 文档 tool:cache:{tool}:{hash}）：
       Redis 还没接，先用进程内 dict + 过期时间戳顶上，key 结构照
       文档设计，接 Redis 时只换 _cache_get/_cache_set 两个函数。
       写操作永不缓存——副作用重放是事故。哈希串里必须含 user_id：
       按用户维度的只读工具（"我的订单列表"）参数可能就是空表，只哈希
       arguments 会让两个用户命中同一条缓存，把别人的订单发出去。
    D6 身份以会话为准，模型说的不算：execute 收到的 user_id 覆盖掉
       arguments 里模型自己填的任何 user_id。不这么做就有两个问题——
       一是越权（用户说"查一下 u2002 的订单"，模型照填就能读到别人的
       数据），二是缓存 key 里的 user_id 变成死分量（handler 拿到的
       仍是模型给的，隔离了个寂寞）。这也是 P3 接多用户时唯一的
       身份注入点：把这里的来源从硬编码换成登录态即可。

用法：from app.tools.executor import execute
命令行：python -m app.tools.executor（跑内置自测）
"""
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout   # Py3.10 的 future 超时异常（3.11 起与内建合并）
from dataclasses import dataclass, field
from pathlib import Path

from .registry import load_definitions, is_write_op
from .mock import HANDLERS, ToolInputError

# 超时秒数（02 文档：统一 3s）
TOOL_TIMEOUT_S = 3.0

# 日志目录：项目根/data/logs
LOG_DIR = Path(__file__).resolve().parents[2] / "data" / "logs"

# 超时/异常时的降级话术（02 文档：超时降级为话术回复）
TIMEOUT_MSG = "查询超时了，请您稍后再试，或直接输入「转人工」由人工客服为您查询。"
ERROR_PREFIX = "抱歉，查询没有成功："

# 专门的线程池：mock 阶段够小；真实业务换 HTTP 客户端时一并调大
_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")


@dataclass
class ToolResult:
    """一次工具调用的统一结果（D3 三态 + 原始数据）。"""
    status: str                 # ok / error / timeout / needs_confirmation
    data: dict = field(default_factory=dict)     # 成功时的工具返回
    message: str = ""           # 给用户的可直接展示的话术（含降级话术）

    @property
    def ok(self) -> bool:
        return self.status == "ok"


# ---------------- 5min 结果缓存（D5，只缓存只读 ok 结果） ----------------

_CACHE_TTL = 300.0
_cache: dict[str, tuple[float, ToolResult]] = {}
_cache_lock = threading.Lock()


def _cache_key(tool: str, arguments: dict, user_id: str) -> str:
    """04 文档的 tool:cache:{tool}:{hash}：用户 + 参数转稳定 JSON 再哈希。

    user_id 进哈希而不是另起一段（key 形状保持文档里的三段式）。少了它，
    "我的订单列表"这类参数为空的用户维度工具会串户——u1001 查过之后，
    u2002 用同样的空参数直接命中 u1001 的缓存。
    """
    canon = json.dumps({"user_id": user_id, "arguments": arguments},
                       sort_keys=True, ensure_ascii=False)
    return f"tool:cache:{tool}:{hashlib.md5(canon.encode('utf-8')).hexdigest()}"


def _cache_get(key: str) -> ToolResult | None:
    with _cache_lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < _CACHE_TTL:
            return hit[1]
        _cache.pop(key, None)      # 过期顺手清掉，避免缓存无限涨
        return None


def _cache_set(key: str, r: ToolResult) -> None:
    with _cache_lock:
        _cache[key] = (time.time(), r)


def clear_cache() -> None:
    """测试/运维用：清空结果缓存。"""
    with _cache_lock:
        _cache.clear()


# ---------------- 执行入口 ----------------

def execute(name: str, arguments: dict | None = None, *,
            user_id: str = "", confirmed: bool = False) -> ToolResult:
    """执行一个工具。参数不合法/未知工具 → error；写操作未确认 →
    needs_confirmation（D1）；超时 → timeout（D2）。永远不向上抛异常。"""
    arguments = dict(arguments or {})
    t0 = time.perf_counter()

    # ---- D6：身份以会话为准 ----
    # 先无条件剥掉模型填的 user_id（哪怕这次没登录态，也不能让模型替用户
    # 挑身份），再由本函数注入会话身份。handler 因此总能拿到可信的 user_id，
    # 缓存 key 里那个分量也才是活的。
    arguments.pop("user_id", None)
    if user_id:
        arguments["user_id"] = user_id

    # ---- 前置检查（不进线程池，快得很） ----
    if name not in load_definitions() or name not in HANDLERS:
        r = ToolResult("error", message=f"{ERROR_PREFIX}未知工具 {name}")
        _log(user_id, name, arguments, r, time.perf_counter() - t0)
        return r

    if is_write_op(name) and not confirmed:            # D1 确认门
        r = ToolResult("needs_confirmation", message=_proposal(name, arguments))
        _log(user_id, name, arguments, r, time.perf_counter() - t0)
        return r

    # ---- 只读走缓存（D5） ----
    key = _cache_key(name, arguments, user_id)
    if not is_write_op(name):
        cached = _cache_get(key)
        if cached is not None:
            return cached

    # ---- 真执行：线程池 + 超时（D2） ----
    try:
        future = _pool.submit(HANDLERS[name], **arguments)
        data = future.result(timeout=TOOL_TIMEOUT_S)
        r = ToolResult("ok", data=data)
    except ToolInputError as e:                        # 业务性查不到（D3）
        r = ToolResult("error", message=f"{ERROR_PREFIX}{e}")
    except TimeoutError:                               # Py3.11+：future 超时抛的就是它
        r = ToolResult("timeout", message=TIMEOUT_MSG)
    except FuturesTimeout:                             # Py3.10 及以下抛这个（Exception 子类，必须单列）
        r = ToolResult("timeout", message=TIMEOUT_MSG)
    except Exception as e:                             # mock 自身的 bug 也不裸抛
        r = ToolResult("error", message=f"{ERROR_PREFIX}工具内部错误 ({type(e).__name__})")

    elapsed = time.perf_counter() - t0
    if r.ok and not is_write_op(name):
        _cache_set(key, r)
    _log(user_id, name, arguments, r, elapsed)
    return r


# ---------------- 内部辅助 ----------------

def _proposal(name: str, arguments: dict) -> str:
    """写操作确认门给用户看的提案（D1）。"""
    if name == "apply_after_sale":
        t = {"return": "退货", "exchange": "换货", "repair": "维修"}.get(
            arguments.get("type"), arguments.get("type", "售后"))
        return (f"即将为您提交{t}申请：订单 {arguments.get('order_id', '?')} 中的商品 "
                f"{arguments.get('product_id', '?')}，原因：{arguments.get('reason', '未填写')}。"
                f"确认提交吗？")
    return "即将为您转接人工客服，请确认。"


def _log(user_id: str, name: str, arguments: dict, r: ToolResult, elapsed: float) -> None:
    """D4：追加写一行 JSON 到 data/logs/tool_calls.jsonl。写失败只 print，不影响主流程。"""
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "user_id": user_id,
           "tool": name, "arguments": arguments,
           "result": {"status": r.status, "data": r.data, "message": r.message},
           "elapsed_ms": round(elapsed * 1000)}
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with (LOG_DIR / "tool_calls.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[tool_log] 写日志失败: {e}")


# ---------------- 自测：python -m app.tools.executor ----------------
if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    # 1. 只读直接执行
    r = execute("query_stock", {"product_id": "SKU-10001"}, user_id="u1001")
    assert r.ok and r.data["stock"] == 42, r
    print("1 只读工具:", r.status, r.data)

    # 2. 缓存生效：同参二连调，日志里只多一条（第一条在 execute 里已写）
    r2 = execute("query_stock", {"product_id": "SKU-10001"}, user_id="u1001")
    assert r2.ok and r2.data == r.data
    print("2 缓存命中: OK（同参同果，未重复执行）")

    # 3. 写操作确认门：未确认 → needs_confirmation，且不产生副作用
    args = {"order_id": "ORD-20260908-002", "product_id": "SKU-30001",
            "type": "return", "reason": "尺码偏大"}
    r3 = execute("apply_after_sale", args, user_id="u1001")
    assert r3.status == "needs_confirmation", r3.status
    print("3 确认门:", r3.status, "|", r3.message)

    # 4. 确认后真执行
    r4 = execute("apply_after_sale", args, user_id="u1001", confirmed=True)
    assert r4.ok and r4.data["after_sale_id"].startswith("AS-"), r4
    print("4 确认执行:", r4.data)

    # 5. 业务错误 & 未知工具不抛异常
    r5 = execute("query_order", {"order_id": "ORD-XXX"})
    assert r5.status == "error" and r5.message
    r6 = execute("不存在的工具", {})
    assert r6.status == "error"
    print("5 错误三防: error 不裸抛 /", r5.message)

    # 6. 超时路径：mock 没有慢工具，用一个垫片验证超时分支本身。
    #    注意 runpy：本文件以 __main__ 运行时，import app.tools.executor
    #    会得到另一个模块实例，补丁必须打在自己（globals）身上才生效。
    HANDLERS["__slow"] = lambda: time.sleep(5)
    orig_ld = load_definitions
    load_definitions = lambda: {**orig_ld(), "__slow": {   # 临时注入定义，让前置检查放行
        "name": "__slow", "description": "", "parameters": {"type": "object", "properties": {}}}}
    try:
        t0 = time.perf_counter()
        r7 = execute("__slow", {}, confirmed=True)   # is_write_op 对未知名偏保守，直接放行
    finally:
        load_definitions = orig_ld                        # 无论成败都还原
        del HANDLERS["__slow"]
    assert r7.status == "timeout" and time.perf_counter() - t0 < 4, (r7, time.perf_counter() - t0)
    print("6 超时降级:", r7.status, "|", r7.message)

    # 7. 日志落盘校验
    lines = (LOG_DIR / "tool_calls.jsonl").read_text(encoding="utf-8").strip().splitlines()
    last = json.loads(lines[-1])
    assert {"ts", "user_id", "tool", "arguments", "result", "elapsed_ms"} <= set(last)
    print(f"7 日志: 共 {len(lines)} 条，末条字段完整（tool={last['tool']}）")

    # 8. 缓存按用户隔离（D5）：同参数不同用户不得命中同一条缓存。
    #    query_price 的优惠券只发给 u1001，正好能验出"有没有拿到别人的结果"。
    clear_cache()
    ra = execute("query_price", {"product_id": "SKU-10001"}, user_id="u1001")
    rb = execute("query_price", {"product_id": "SKU-10001"}, user_id="u2002")
    assert ra.ok and rb.ok
    assert ra.data["coupons"] and not rb.data["coupons"], (ra.data, rb.data)
    print("8 缓存按用户隔离: u1001 有券 / u2002 无券")

    # 9. 越权防护（D6）：模型在参数里塞的 user_id 必须无效。
    #    必须用"没有会话身份"来测——有身份时注入那一步会覆盖掉模型的值，
    #    测出来是绿的但 pop 那行有没有起作用根本看不出来（假通过）。
    clear_cache()
    r9 = execute("query_price", {"product_id": "SKU-10001", "user_id": "u1001"})
    assert not r9.data["coupons"], "无会话身份时模型塞的 user_id 生效了，越权没堵住"
    print("9 越权防护: 无会话身份时，模型塞的 user_id 被丢弃")

    print("\n自测全部通过")
