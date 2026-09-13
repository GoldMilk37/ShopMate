"""工具层第 2 步：6 个工具的模拟（mock）业务实现。已实现。

模块速览：
    HANDLERS            {工具名: 函数}，executor 按名字分发
    其余 _ 开头的       各工具的 mock 实现 / 数据源

四个设计决策：
    D1 商品数据不另造一份：价格从 data/rag_docs/specs/*.json 读（发布时
       快照），库存是快照里没有的实时数据，由本模块的 _STOCK 表模拟——
       正好对应 01 文档 §6"价格/库存不进 RAG、走 Function Calling"的边界。
    D2 mock 返回的结构严格贴合 JSON 定义里的 returns：字段名对不上，
       LLM 拿到的就是垃圾。写新 mock 时对照 data/tools/*.json 抄字段。
    D3 写操作有真实副作用：apply_after_sale 会往内存售后表插记录、
       递增工单号，transfer_to_human 生成排队工单——让状态机的
       "提案→确认→执行"流程有东西可测，而不是空转返回 ok。
    D4 查不到就抛 ToolInputError（本文件定义），由 executor 统一转成
       结构化错误结果。mock 不返回半吊子的 {"error": ...} 混进正常数据。

用法：只读 HANDLERS，不直接调函数——超时/日志/确认都在 executor。
"""
import json
import random
import time
from pathlib import Path

# 商品快照：价格/名称从 specs 读（D1）
SPECS_ROOT = Path(__file__).resolve().parents[2] / "data" / "rag_docs" / "specs"


class ToolInputError(Exception):
    """参数或业务对象查不到（如订单号不存在）。executor 转结构化错误。"""


# ---------------- 内存数据库（重启即失，mock 阶段够用） ----------------

def _load_specs() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(SPECS_ROOT.glob("*.json")):
        out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    return out

_SPECS = _load_specs()

# 库存：快照里没有的"实时"数据，这里造（D1）。负数=刻意断货便于测兜底。
_STOCK = {
    "SKU-10001": {"stock": 42, "restock_date": ""},
    "SKU-10002": {"stock": 7,  "restock_date": ""},
    "SKU-10003": {"stock": 0,  "restock_date": "2026-09-20"},   # 断货+有补货计划
    "SKU-20001": {"stock": 15, "restock_date": ""},
    "SKU-30001": {"stock": 60, "restock_date": ""},
    "SKU-30002": {"stock": 3,  "restock_date": ""},
    "SKU-40001": {"stock": 28, "restock_date": ""},
    "SKU-40002": {"stock": 0,  "restock_date": ""},             # 断货+无补货计划
}

# 订单：u1001 有两笔（一笔在途可查物流，一笔已完成可走售后）
_ORDERS = {
    "ORD-20260901-001": {
        "order_id": "ORD-20260901-001", "user_id": "u1001", "status": "shipped",
        "items": [{"product_id": "SKU-10001", "name": _SPECS["SKU-10001"]["name"],
                   "qty": 1, "price": 599.0}],
        "total_amount": 599.0, "created_at": "2026-09-01 10:23:00",
        "tracking_no": "SF1388001122334",
    },
    "ORD-20260908-002": {
        "order_id": "ORD-20260908-002", "user_id": "u1001", "status": "completed",
        "items": [{"product_id": "SKU-30001", "name": _SPECS["SKU-30001"]["name"],
                   "qty": 1, "price": 159.0}],
        "total_amount": 159.0, "created_at": "2026-09-08 19:05:00",
        "tracking_no": "YT9900112233445",
    },
}

# 售后工单池（apply_after_sale 的副作用落这里，D3）
_AFTER_SALES: list[dict] = []


# ---------------- 内部辅助 ----------------

def _find_order(kwargs: dict) -> dict:
    """按 order_id 精确查；没有就按 user_id 取最近一笔（JSON 定义的行为）。"""
    oid = kwargs.get("order_id")
    if oid:
        if oid not in _ORDERS:
            raise ToolInputError(f"订单不存在: {oid}")
        return _ORDERS[oid]
    uid = kwargs.get("user_id")
    if not uid:
        raise ToolInputError("order_id 和 user_id 至少提供一个")
    mine = [o for o in _ORDERS.values() if o["user_id"] == uid]
    if not mine:
        raise ToolInputError(f"用户 {uid} 没有订单")
    return max(mine, key=lambda o: o["created_at"])      # 字典序=时间序（ISO 格式）


def _find_product_id(kwargs: dict) -> str:
    """product_id 直取；否则用 product_name 在 specs 的 name 里模糊匹配。"""
    pid = kwargs.get("product_id")
    if pid:
        if pid not in _SPECS:
            raise ToolInputError(f"商品不存在: {pid}")
        return pid
    kw = kwargs.get("product_name", "")
    if not kw:
        raise ToolInputError("product_id 和 product_name 至少提供一个")
    hits = [p for p, d in _SPECS.items() if kw in str(d.get("name", ""))]
    if not hits:
        raise ToolInputError(f"没有匹配 '{kw}' 的商品")
    if len(hits) > 1:
        raise ToolInputError(f"'{kw}' 匹配到多个商品: {hits}，请提供精确 ID")
    return hits[0]


# ---------------- 六个工具的 mock 实现 ----------------

def _query_order(**kwargs) -> dict:
    order = _find_order(kwargs)
    flt = kwargs.get("status_filter", "all")
    if flt != "all" and order["status"] != flt:
        raise ToolInputError(f"订单 {order['order_id']} 状态是 {order['status']}，不匹配筛选 {flt}")
    return {"order_id": order["order_id"], "status": order["status"],
            "items": order["items"], "total_amount": order["total_amount"],
            "created_at": order["created_at"]}


def _query_stock(**kwargs) -> dict:
    pid = _find_product_id(kwargs)
    row = _STOCK[pid]
    return {"product_id": pid, "stock": row["stock"],
            "in_stock": row["stock"] > 0, "restock_date": row["restock_date"]}


def _query_price(**kwargs) -> dict:
    pid = _find_product_id(kwargs)
    price = _SPECS[pid].get("price", {})
    return {"product_id": pid,
            "original_price": price.get("original"),
            "current_price": price.get("current"),
            "promotions": price.get("promotions", []),
            "coupons": ["新人立减 20 元"] if kwargs.get("user_id") == "u1001" else []}


def _track_logistics(**kwargs) -> dict:
    order = _find_order(kwargs)
    tn = kwargs.get("tracking_no") or order.get("tracking_no", "")
    if not tn:
        raise ToolInputError(f"订单 {order['order_id']} 还没有物流单号（尚未发货）")
    if order["status"] == "completed":
        status, eta = "已签收", ""
        traces = [
            {"time": "2026-09-03 08:11:00", "desc": "【深圳市】已发出"},
            {"time": "2026-09-04 14:36:00", "desc": "【杭州市】到达派送网点"},
            {"time": "2026-09-04 18:02:00", "desc": "已签收，感谢使用"},
        ]
    else:
        status, eta = "运输中", "2026-09-14 18:00 前"
        traces = [
            {"time": "2026-09-11 21:40:00", "desc": "【深圳市】顺丰速运已揽收"},
            {"time": "2026-09-12 09:15:00", "desc": "【东莞转运中心】已发出"},
            {"time": "2026-09-13 07:02:00", "desc": "【杭州转运中心】已到达"},
        ]
    return {"carrier": "顺丰速运" if tn.startswith("SF") else "圆通速递",
            "tracking_no": tn, "status": status, "traces": traces,
            "estimated_delivery": eta}


def _apply_after_sale(**kwargs) -> dict:
    for req in ("order_id", "product_id", "type", "reason"):
        if not kwargs.get(req):
            raise ToolInputError(f"缺少必填参数: {req}")
    if kwargs["type"] not in ("return", "exchange", "repair"):
        raise ToolInputError(f"售后类型不合法: {kwargs['type']}")
    order = _find_order({"order_id": kwargs["order_id"]})
    if not any(i["product_id"] == kwargs["product_id"] for i in order["items"]):
        raise ToolInputError(f"订单 {order['order_id']} 里没有商品 {kwargs['product_id']}")
    # D3：真实副作用——插记录、递增工单号
    aid = f"AS-{20260900 + len(_AFTER_SALES) + 1}"
    _AFTER_SALES.append({"after_sale_id": aid, **kwargs,
                         "status": "待审核", "created_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    return {"after_sale_id": aid, "status": "待审核",
            "next_step": "客服将在 24 小时内审核，请保持电话畅通"}


def _transfer_to_human(**kwargs) -> dict:
    if not kwargs.get("reason"):
        raise ToolInputError("缺少必填参数: reason")
    ticket = f"HR-{random.randint(1000, 9999)}"
    pos = random.randint(1, 3)
    return {"ticket_id": ticket, "queue_position": pos,
            "estimated_wait": f"约 {pos * 2} 分钟"}


HANDLERS = {
    "query_order": _query_order,
    "query_stock": _query_stock,
    "query_price": _query_price,
    "track_logistics": _track_logistics,
    "apply_after_sale": _apply_after_sale,
    "transfer_to_human": _transfer_to_human,
}


# ---------------- 自测：python -m app.tools.mock ----------------
if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    from .registry import load_definitions
    defs = load_definitions()
    assert set(HANDLERS) == set(defs), f"实现与定义不齐: {set(HANDLERS) ^ set(defs)}"

    print("查订单(user_id 兜底取最近):", _query_order(user_id="u1001")["order_id"])
    print("查库存(名称模糊):", _query_stock(product_name="洗烘一体机"))
    print("查价格:", _query_price(product_id="SKU-10001", user_id="u1001"))
    print("查物流:", _track_logistics(order_id="ORD-20260901-001")["status"], "→ 预计", end=" ")
    r = _track_logistics(order_id="ORD-20260901-001")
    print(r["estimated_delivery"])
    print("售后申请:", _apply_after_sale(order_id="ORD-20260908-002",
                                         product_id="SKU-30001", type="return",
                                         reason="尺码偏大")["after_sale_id"])
    print("转人工:", _transfer_to_human(reason="user_request")["ticket_id"])

    # 错误路径也要通（D4：抛 ToolInputError 而不是返回半吊子 dict）
    for bad in (lambda: _query_order(order_id="ORD-XXX"),
                lambda: _query_stock(product_name="不存在的东西"),
                lambda: _apply_after_sale(order_id="ORD-20260908-002",
                                          product_id="SKU-10001", type="return", reason="x")):
        try:
            bad()
            raise AssertionError("应抛 ToolInputError")
        except ToolInputError as e:
            print("  预期内报错:", e)
    print("自测通过")
