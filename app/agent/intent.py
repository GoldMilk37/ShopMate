"""意图识别节点：用户一句话 → (意图, 置信度, 槽位, 情绪)。已实现。

模块速览：
    classify(text, history)   唯一入口，graph 调这个
    IntentResult              结果数据类

四个设计决策：
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

置信度阈值不在这里判——CLARIFY_THRESHOLD 的裁决权在状态机（它还
要结合其他信号），本模块只负责"测出"置信度。
"""
import json
from dataclasses import dataclass, field

from ..llm import client as llm

# 意图集合（D1）。值与 graph 的路由表一一对应。
INTENTS = (
    "product_consult",    # 商品咨询：功能/材质/适用场景
    "param_compare",      # 参数对比：两个及以上商品对比
    "recommendation",     # 个性化推荐：有需求没明确商品
    "order_query",        # 订单查询：订单号/物流/价格/库存
    "after_sale",         # 售后处理：退换货/维修/投诉
    "human_transfer",     # 点名转人工（工具定义的触发条件 1）
    "chitchat",           # 闲聊/其他
)

SYSTEM_PROMPT = """你是电商客服的意图分类器。把用户消息分类并抽取信息，输出严格的 json 对象（不要输出其他内容）：
{"intent": "product_consult|param_compare|recommendation|order_query|after_sale|human_transfer|chitchat",
 "confidence": 0到1的小数,
 "dissatisfied": true|false,
 "slots": {"product_name": "", "order_id": "", "compare": [], "need": ""}}
判定要点：
- product_consult：问具体商品的功能、材质、参数、适用场景；price/stock 的实时数字属于 order_query
- param_compare：一句话里出现两个及以上要比的商品
- recommendation：表达购买需求但没点名商品（"预算600求推荐"）
- order_query：订单状态、物流进度、实时价格、实时库存
- after_sale：退、换、修、投诉、赔偿
- human_transfer：用户明确说转人工/找真人（只是抱怨不算，抱怨把 dissatisfied 置 true）
- dissatisfied：用户表达不满/生气/失望（"什么破东西""等了三天还没到，火大"）
- compare 填要对比的商品名/型号列表；need 填推荐场景描述；没有的槽位留空串/空表
confidence 给你自己的判断把握，拿不准就给低分，不要硬凑高。"""


@dataclass
class IntentResult:
    """classify 的输出。confidence=0 视为"不知道"（D3）。"""
    intent: str = "chitchat"
    confidence: float = 0.0
    dissatisfied: bool = False
    slots: dict = field(default_factory=dict)


def classify(text: str, history: list[dict] | None = None) -> IntentResult:
    """识别意图。LLM/解析任何环节出问题都返回 confidence=0（D3），
    不向上抛异常——状态机的兜底分支就是为这种时刻准备的。"""
    # D4：只带最近 4 条 user 消息作指代消解的上下文
    recent = [m["content"] for m in (history or [])[-8:] if m["role"] == "user"][-4:]
    context = ("\n近期对话（仅供消解指代，如\"它/这个\"指什么）：\n"
               + "\n".join(f"- {c}" for c in recent)) if recent else ""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"用户消息：{text}{context}"},
    ]
    try:
        raw = llm.chat(messages, temperature=0.0, json_mode=True).content or ""
        d = json.loads(raw)
        intent = d.get("intent", "")
        if intent not in INTENTS:                       # 幻觉出的意图名按未知处理
            return IntentResult()
        conf = float(d.get("confidence", 0.0))
        slots = d.get("slots") if isinstance(d.get("slots"), dict) else {}
        return IntentResult(intent, max(0.0, min(1.0, conf)),
                            bool(d.get("dissatisfied", False)), slots)
    except Exception:                                   # 网络/JSON/类型，全按 D3
        return IntentResult()


# ---------------- 自测：python -m app.agent.intent（需 .env 里的 key） ----------------
if __name__ == "__main__":
    import os
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("跳过：.env 里没有 DEEPSEEK_API_KEY（client.py 已自测过缺 key 路径）")
        sys.exit(0)

    cases = [   # (话术, 期望意图, 期望槽位非空)
        ("SKU-10001 这个耳机续航多久", "product_consult", "product_name"),
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
    print(f"\n{ok}/{len(cases)} 通过")
