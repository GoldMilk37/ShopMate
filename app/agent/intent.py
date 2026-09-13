"""意图识别节点：用户一句话 → (意图, 置信度, 槽位, 情绪)。已实现。

模块速览：
    classify(text, history)   唯一入口，graph 调这个
    IntentResult              结果数据类
    product_catalog()         SKU → 商品名的目录（给实体链接与上层复用）

七个设计决策：
    D1 意图集合 = 03 文档六类 + human_transfer：文档表里没有"用户点名
       转人工"，但 transfer_to_human 工具定义的第一条触发条件就是它，
       且这类请求必须零延迟直达——不过状态机、不让 LLM 编排，识别出
       即转。补第七类是对文档的修订而非偏离。
    D2 一次调用全带走：意图、置信度、槽位（商品名/订单号/对比对象/
       需求描述）、情绪（用户是否在表达不满）让同一次 LLM 调用输出。
       拆成多次调用会引入不一致（同一句话情绪判不满、意图却判闲聊）。
    D3 解析失败按最保守处理：JSON 坏了 → intent=chitchat、confidence=0，
       状态机的置信度兜底会自然接管（追问澄清）。宁可多问一句，
       不可错路由一次。
    D4 历史只喂最近 4 条 user 消息：意图识别要历史是为了消解指代
       （"那它防水吗"的"它"），assistant 消息和更久远的轮次是噪音，
       白花钱还稀释注意力。
    D5 商品ID两步走：正则优先、LLM 兜底。用户话里出现 SKU-xxxx 时
       直接正则锁定——确定性的东西不该让模型掷骰子，零成本零幻觉；
       没报型号时才把目录（SKU + 商品名）塞进 prompt 让 LLM 做实体
       链接（"这款耳机"→SKU-10001）。LLM 给的 PID 必须在本目录里，
       查不到就当没抽取——宁可不过滤，也别用一个幻觉出来的 ID 去把
       检索结果筛成空集。
    D6 评价域单列第八个意图 review_consult：01 文档 §二 给评价单独建了
       review_knowledge 库，但意图集原先只有三个 RAG 意图、全指向
       product_knowledge——库建好了没人查，"XX 口碑怎么样"会拿商品详情
       作答。补这一类是对 03 文档意图表的修订（性质同 D1 补 human_transfer）。
       不并进 product_consult 是因为两者查的不是一个库：并进去就得在节点
       内部塞"口碑/缺点/评价"关键词做二次判定，规则难维护，而且会让
       product_consult 的置信度失去统一语义（一半来自模型、一半来自规则）。
    D7 出网失败与解析失败分开处理，且都留痕：原先一个 try 包住了"调模型"
       和"解析 JSON"两件事，两种失败在输出上完全一样（都是 confidence=0），
       所以自测 9 条全判 chitchat 那次看起来像分类器坏了——实际是 TLS 握手
       被中间设备掐断（见 llm.client D4）。重试本身交给 llm.safe_call
       （策略收口在 LLM 层，不在这里再养一份）；本层负责的是：每一处降级
       都打印**具体是哪一步、坏在什么值上**。代价是一段代码分成两截，
       收益是下次网络出事能在 10 秒内定位，而不是从"分类器坏了"倒着查。

置信度阈值不在这里判——CLARIFY_THRESHOLD 的裁决权在状态机（它还
要结合其他信号），本模块只负责"测出"置信度。
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..llm import client as llm

# 商品目录来源：与 RAG 知识库同源（01 文档 §2 的 specs/）
SPECS_ROOT = Path(__file__).resolve().parents[2] / "data" / "rag_docs" / "specs"

# D5 第一步：型号字面量直接命中，格式如 SKU-10001
_SKU_RE = re.compile(r"sku-(\d+)", re.IGNORECASE)

# 意图集合（D1）。值与 graph 的路由表一一对应。
INTENTS = (
    "product_consult",    # 商品咨询：客观信息（功能/材质/参数/场景）
    "review_consult",     # 口碑评价：主观体验（优缺点/值不值得买）（D6）
    "param_compare",      # 参数对比：两个及以上商品对比
    "recommendation",     # 个性化推荐：有需求没明确商品
    "order_query",        # 订单查询：订单号/物流/价格/库存
    "after_sale",         # 售后处理：退换货/维修/投诉
    "human_transfer",     # 点名转人工（工具定义的触发条件 1）
    "chitchat",           # 闲聊/其他
)

SYSTEM_PROMPT = """你是电商客服的意图分类器。把用户消息分类并抽取信息，输出严格的 json 对象（不要输出其他内容）：
{"intent": "product_consult|review_consult|param_compare|recommendation|order_query|after_sale|human_transfer|chitchat",
 "confidence": 0到1的小数,
 "dissatisfied": true|false,
 "slots": {"product_name": "", "product_id": "", "order_id": "", "compare": [], "need": ""}}
在售商品目录（product_id 只能从这里选）：
{CATALOG}
判定要点：
- product_consult：问**客观信息**——功能、材质、参数、规格、适用场景；price/stock 的实时数字属于 order_query
- review_consult：问**主观体验**——口碑、评价、优缺点、"值不值得买"、真实使用感受、别人怎么说。
  同一件商品："多久充一次电"是 product_consult（客观参数），"续航够用吗"是 review_consult（使用体验）。
  拿不准时看问的是"它是什么"还是"它好不好用"
- param_compare：一句话里出现两个及以上要比的商品
- recommendation：表达购买需求但没点名商品（"预算600求推荐"）
- order_query：订单状态、物流进度、实时价格、实时库存
- after_sale：退、换、修、投诉、赔偿
- human_transfer：用户明确说转人工/找真人（只是抱怨不算，抱怨把 dissatisfied 置 true）
- dissatisfied：用户表达不满/生气/失望（"什么破东西""等了三天还没到，火大"）
- compare 填要对比的商品名/型号列表；need 填推荐场景描述；没有的槽位留空串/空表
- product_id：用户指的**具体商品**在目录里的 ID。只在能确定时填：消息里报了型号，
  或结合近期对话能确定（用户刚问完耳机再说"它续航多久"）。拿不准就留空串——
  填错会让检索被过滤到别的商品上，比不过滤更糟
confidence 给你自己的判断把握，拿不准就给低分，不要硬凑高。"""


def product_catalog() -> dict[str, str]:
    """返回 {SKU: 商品名}。首次调用读 specs/，之后复用（同 get_model 的懒加载套路）。"""
    global _catalog
    if _catalog is None:
        cat: dict[str, str] = {}
        for p in sorted(SPECS_ROOT.glob("*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            cat[str(d["product_id"])] = str(d.get("name", ""))
        _catalog = cat
    return _catalog


_catalog: dict[str, str] | None = None


def _pid_from_text(text: str) -> str:
    """D5 第一步：从字面抽出 SKU-xxxx。抽不到返回空串。

    型号是确定性的字面证据，走正则胜过让 LLM 复述一遍可能抄错的编号。
    """
    m = _SKU_RE.search(text or "")
    return f"SKU-{m.group(1)}" if m else ""


def _system_prompt() -> str:
    """把商品目录塞进 SYSTEM_PROMPT 的占位符——LLM 做实体链接得先知道在卖什么。"""
    catalog = product_catalog()
    lines = "\n".join(f"  - {pid} {name}" for pid, name in catalog.items())
    return SYSTEM_PROMPT.replace("{CATALOG}", lines or "  （目录为空）")


@dataclass
class IntentResult:
    """classify 的输出。confidence=0 视为"不知道"（D3）。"""
    intent: str = "chitchat"
    confidence: float = 0.0
    dissatisfied: bool = False
    slots: dict = field(default_factory=dict)


def _resolve_product_id(text: str, slots: dict) -> str:
    """D5：定商品ID。正则优先 → LLM 抽的兜底 → 不在目录里一律丢弃。

    最后那道校验是必须的：一个幻觉出来的 product_id 会被下游拿去当
    where 过滤条件，等于把正确答案筛成空结果，比不过滤更糟。
    """
    pid = _pid_from_text(text) or str(slots.get("product_id", "") or "").strip()
    return pid if pid in product_catalog() else ""


def _parse(raw: str, text: str) -> IntentResult:
    """D7：把模型吐的 JSON 变成 IntentResult。每一处降级各留各的痕。

    和"调模型"分开的理由见 D7——混在一起时，"网络断了"和"JSON 坏了"
    对调用方长得一模一样。
    """
    try:
        d = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"[intent] JSON 解析失败，按未知处理: {type(e).__name__}: {e}；"
              f"原文={raw[:120]!r}")
        return IntentResult()
    if not isinstance(d, dict):
        print(f"[intent] 模型返回的不是对象（{type(d).__name__}），按未知处理")
        return IntentResult()

    intent = d.get("intent", "")
    if intent not in INTENTS:
        print(f"[intent] 模型给了意图集外的名字 {intent!r}，按未知处理")
        return IntentResult()

    try:
        conf = float(d.get("confidence", 0.0))
    except (TypeError, ValueError):
        # 意图认得，但把握给成了非数字。保守取 0：状态机会再追问一句，
        # 总比拿一个编出来的把握去路由强。
        print(f"[intent] confidence 不是数字（{d.get('confidence')!r}），按下限 0 处理")
        conf = 0.0

    slots = d.get("slots")
    if not isinstance(slots, dict):
        if slots is not None:
            print(f"[intent] slots 不是对象（{type(slots).__name__}），按空槽位处理")
        slots = {}
    slots["product_id"] = _resolve_product_id(text, slots)
    return IntentResult(intent, max(0.0, min(1.0, conf)),
                        bool(d.get("dissatisfied", False)), slots)


def classify(text: str, history: list[dict] | None = None) -> IntentResult:
    """识别意图。LLM/解析任何环节出问题都返回 confidence=0（D3），
    不向上抛异常——状态机的兜底分支就是为这种时刻准备的。

    D7：调模型（含重试，走 llm.safe_call）与解析分成两截，各报各的错。
    """
    # D4：只带最近 4 条 user 消息作指代消解的上下文
    recent = [m["content"] for m in (history or [])[-8:] if m["role"] == "user"][-4:]
    context = ("\n近期对话（仅供消解指代，如\"它/这个\"指什么）：\n"
               + "\n".join(f"- {c}" for c in recent)) if recent else ""
    messages = [
        {"role": "system", "content": _system_prompt()},   # 带商品目录
        {"role": "user", "content": f"用户消息：{text}{context}"},
    ]

    msg = llm.safe_call(llm.chat, messages, temperature=0.0, json_mode=True)
    if msg is None:                     # 重试过仍失败；真实异常 safe_call 已打印
        return IntentResult()

    try:
        return _parse(msg.content or "", text)
    except Exception as e:              # D3 的契约：绝不向上抛
        # 走到这儿说明坏在解析路径的意料之外处（例如 specs/ 里的 JSON 读不动）。
        # 契约不变，但照样留痕——否则又是一次"全变 chitchat"的无头案。
        print(f"[intent] 解析兜底触发，按未知处理: {type(e).__name__}: {e}")
        return IntentResult()


# ---------------- 自测：python -m app.agent.intent（需 .env 里的 key） ----------------
if __name__ == "__main__":
    import os
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    # ---- 不联网也能验的部分（D5 里确定性的那几步） ----
    assert _pid_from_text("SKU-10001 续航多久") == "SKU-10001"
    assert _pid_from_text("这款耳机怎么样") == ""
    assert _resolve_product_id("SKU-99999 怎么样", {}) == "", "目录外的ID必须丢弃"
    assert _resolve_product_id("下单一件", {"product_id": "SKU-10001"}) == "SKU-10001"
    print(f"型号抽取（无需 key）: 目录 {len(product_catalog())} 个 SKU，"
          "正则命中 / 目录外ID丢弃 全部 OK\n")

    # ---- D7：解析降级各走各的路径（不联网，直接喂坏 JSON） ----
    # 这五条是"出网正常但模型不老实"的现场。原先它们和网络故障共用一个
    # except，测了也分不清谁是谁——现在每条都能单独断言。
    assert _parse("这不是 json", "随便问问").confidence == 0.0
    assert _parse('["数组，不是对象"]', "随便问问").confidence == 0.0
    assert _parse('{"intent": "free_refund", "confidence": 0.9}', "随便问问").confidence == 0.0
    _r = _parse('{"intent": "product_consult", "confidence": "很高"}', "SKU-10001 续航多久")
    assert _r.intent == "product_consult" and _r.confidence == 0.0, _r
    _r = _parse('{"intent": "product_consult", "confidence": 0.9, "slots": "无"}',
                "SKU-10001 续航多久")
    assert _r.confidence == 0.9 and _r.slots["product_id"] == "SKU-10001", _r
    print("解析降级（无需 key）: 坏JSON / 非对象 / 意图集外 / confidence非数字 / "
          "slots非对象 —— 五条路径均按预期降级\n")

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("跳过：.env 里没有 DEEPSEEK_API_KEY（client.py 已自测过缺 key 路径）")
        sys.exit(0)

    cases = [   # (话术, 期望意图, 期望槽位非空)
        # D6 的边界要成对测：只测"能认出评价"不够，还得测"没把客观咨询
        # 误判成评价"。"续航"这个词两边都会出现，正好当探针。
        ("SKU-10001 这个耳机续航多久", "product_consult", "product_name"),
        ("SKU-10001 这个耳机口碑怎么样 有什么缺点", "review_consult", None),
        ("冲锋衣防水实测真的能挡雨吗", "review_consult", None),
        ("10001 和 20001 哪个降噪好", "param_compare", "compare"),
        ("预算六百，通勤用，求推荐个耳机", "recommendation", "need"),
        ("我的订单 ORD-20260901-001 到哪了", "order_query", "order_id"),
        ("冲锋衣尺码偏大想换个小的", "after_sale", None),
        ("给我转人工", "human_transfer", None),
        ("今天天气不错", "chitchat", None),
    ]
    ok = 0
    for text, want, slot in cases:
        r = classify(text)
        hit = r.intent == want and r.confidence >= 0.6
        if hit and slot:
            hit = bool(r.slots.get(slot))
        ok += hit
        print(f"{'✓' if hit else '✗'} [{r.confidence:.2f}] {r.intent:16s} {r.slots}  ← {text}")
    # 情绪 + 指代消解（"它"指上文的耳机）
    r = classify("它洗澡能带吗", [{"role": "user", "content": "看看那个降噪耳机"},
                                 {"role": "assistant", "content": "好的"}])
    print(f"指代消解: {r.intent} (期望 product_consult)")
    r2 = classify("等了三天还没发货，火大", [])
    print(f"情绪识别: dissatisfied={r2.dissatisfied} (期望 True)")
    r3 = classify("它有什么缺点", [{"role": "user", "content": "SKU-10001 这个耳机怎么样"},
                                  {"role": "assistant", "content": "挺好的"}])
    print(f"商品ID（上下文指代）: {r3.slots.get('product_id')!r} (期望 SKU-10001)")
    print(f"\n{ok}/{len(cases)} 通过")
