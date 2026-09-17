"""RAGAS 生成侧评测：01 文档 §7 检索指标之外的"生成质量"三件套。新增。

模块速览：
    build_samples()    跑真实检索 + 真实 RAG 生成，攒出评测样本
    load_ground_truth() 读人工标注的标准答案（没有就跳过依赖它的指标）
    main()             组装 EvaluationDataset → RAGAS evaluate → 打表

三个指标各测什么（面试要讲的）：
    faithfulness        答案有没有忠于检索到的资料（防幻觉的量化证据）
    answer_relevancy    答案有没有答到点子上
    context_precision   检索回来的资料里，相关的排得够不够靠前（与 MRR 互证）

四个设计决策：
    D1 裁判模型与生成模型必须不同族：DeepSeek 评 DeepSeek 有自评偏好，
       分数虚高。有 DASHSCOPE_API_KEY 就用 Qwen（dashscope 兼容端点）；
       没有 key 时回退 DeepSeek——但会打警告横幅，此时的分数只能用来
       验证流程跑通，不能当质量结论。
    D2 embedding 复用本地 BGE-M3（indexer.embed_texts 包一层 langchain
       接口）：answer_relevancy 要算问题与答案的向量相似度，RAGAS 默认
       走 OpenAI embedding（联网 + 花钱），没必要——线上检索用的就是
       BGE-M3，评测口径和线上一致反而更对。
    D3 标准答案外置 JSON：ground_truth 必须人工写（HANDOFF 明确），
       脚本只提供模板生成（--dump-template）和读取。缺标注时只跑
       faithfulness / answer_relevancy，依赖标准答案的指标自动跳过，
       而不是拿空串硬评。
    D4 样本量分两档：RAGAS_N 环境变量控制条数，默认 15——先用子集
       跑通（裁判调用按条计费），确认数字合理再上全量 50。

用法：
    python -m app.retrieval.ragas_eval                   # 跑评测
    python -m app.retrieval.ragas_eval --dump-template   # 生成标注模板
依赖：.env 里 DEEPSEEK_API_KEY（生成答案用）；DASHSCOPE_API_KEY（裁判，可选）。
联网脚本，记得带 HTTPS_PROXY（见 HANDOFF §9 的 TLS 坑）。
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# BGE-M3 加载时 transformers 会联网查 chat template（HANDOFF §9 的坑），
# ragas 在事件循环里触发这个加载会一直挂到超时。评测前模型已在缓存里，
# 直接断网模式加载，transformers 报错也不影响（chat template 与 embedding 无关）。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from .retriever import TOP_K, search
from app.llm import client as llm

# 标注文件与产物落盘位置（与 data/ 平级，git 忽略与否见 .gitignore）
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "eval"
GT_FILE = DATA_DIR / "ragas_ground_truth.json"
RESULT_FILE = DATA_DIR / "ragas_results.json"

# 裁判模型：Qwen 优先（D1）。两个端点都是 OpenAI 兼容的。
JUDGE_DASHSCOPE = {
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "model": "qwen-plus",
}


def _pick_judge() -> tuple[str, str, str | None]:
    """返回 (base_url, model, key)。Qwen 优先，DeepSeek 兜底并要求 caller 打警告。"""
    ds_key = os.environ.get("DASHSCOPE_API_KEY")
    if ds_key:
        return JUDGE_DASHSCOPE["base_url"], JUDGE_DASHSCOPE["model"], ds_key
    return llm.BASE_URL, llm.MODEL, os.environ.get("DEEPSEEK_API_KEY")


class BgeM3Embeddings:
    """D2：把 indexer.embed_texts 包成 RAGAS 认的 langchain Embeddings 接口。

    不继承 langchain 基类——RAGAS 只按鸭子类型调 embed_documents/embed_query。
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from .indexer import embed_texts
        return embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        from .indexer import embed_texts
        return embed_texts([text])[0]


class _NoMultiGen:
    """裁判包装：剥掉每次调用里的 n>1 参数。

    RAGAS 的 answer_relevancy 会请裁判一次生成多个答案变体（反解问题，
    strictness 默认 3），再对它们与原问题的向量余弦相似度取平均。
    DeepSeek 端点收 n>1 会 400，所以必须剥。

    **注意别指望对 Qwen 放行 n 就能拿回那个平均。** 2026-09-17 试过：
    不剥 n 之后日志里照样是 `LLM returned 1 generations instead of requested 3`
    ——langchain 这条路径根本没把 n 传到请求体，剥与不剥**实测等价**（同一批
    样本两次打分逐位相同）。所以 answer_relevancy 在这里**拿不到抽样平均这层
    保护**，它的方差只能靠多跑几轮来观察（见 docs/01 §七的实测记录）。
    想要那层平均得绕开 langchain 直接用 ragas 的 llm_factory，是另一件事。

    必须继承 BaseChatModel 而不是鸭子类型包装：RAGAS 只对它认识的
    chat model 做 LangchainLLMWrapper 自动包装，自造对象会走错调用路径。
    """

    def __init__(self, **kwargs):
        from langchain_openai import ChatOpenAI
        from app.llm.client import build_http_client, build_async_http_client
        self._cls = type("_Judge", (ChatOpenAI,), {
            # 生成入口统一在这两个钩子，n>1 在进入请求体前剥掉
            "_generate": _strip_n(ChatOpenAI._generate),
            "_agenerate": _strip_n(ChatOpenAI._agenerate),
        })
        # 裁判也得走 D5 的代理策略：ChatOpenAI 不传 http_client 就自建默认客户端
        # （trust_env=True），于是又去读 Windows 注册表里那个 scheme 写错的系统代理，
        # 报出来的还是那句和网络无关的 EOF。—— 两个客户端都要给，RAGAS 打分走 async。
        self._inner = self._cls(http_client=build_http_client(),
                                http_async_client=build_async_http_client(), **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _strip_n(fn):
    def wrapper(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs.pop("n", None)
        return fn(self, messages, stop=stop, run_manager=run_manager, **kwargs)
    return wrapper


def build_samples(n: int) -> list[dict]:
    """对评测集前 n 条跑"真实检索 + 真实 RAG 生成"，攒 RAGAS 样本。

    生成侧刻意不走完整 Agent（工具编排/转人工与本任务无关），但 prompt、
    检索方式（混合检索）、资料拼接格式都从 app.agent.graph 原样复用——
    评的就是线上那套生成路径，而不是另写一个评不对位的简化版。
    """
    from .eval import EVAL_SET
    from app.agent.graph import RAG_PROMPT

    samples = []
    for q, coll, allowed in EVAL_SET[:n]:
        hits = search(q, coll)
        contexts = [h["text"] for h in hits]
        docs = "\n\n".join(
            f"【资料{i}】({h['meta'].get('doc_type', '')} | {h['meta'].get('product_id', '')})\n{h['text']}"
            for i, h in enumerate(hits, 1))
        reply = None
        for attempt in (1, 2):
            reply = llm.safe_call(llm.chat_text, [
                {"role": "system", "content": RAG_PROMPT},
                {"role": "user", "content": f"【资料】\n{docs}\n\n【用户问题】{q}"},
            ])
            # None = 没调通（safe_call 给的），"" = 调通了但模型没吐字——client.chat_text
            # 刻意区分这两件事，graph.py 也分开处理（空串按"资料没覆盖"兜底）。
            # 但**评测**不能这么兜：那两行代码是在替用户回话，而 RAGAS 是在量模型的
            # 输出质量。往样本里塞一条空回复，等于让三个指标去给空字符串打分，出来的
            # 数不属于任何东西。所以这里既不兜底也不采信，重试一次、再空就跳过。
            #
            # 注意措辞的边界：这条分支是**防御性**的，不是"实测到大量空回复"——
            # 我一度以为 2026-09-17 那 15 条里有 5 条空回复，那是读错了
            # ragas_results.json（它压根不存 response 字段，`.get('response','')`
            # 读的是默认值）。真正的 4 个 0.00 是 answer_relevancy 的抽样方差，
            # 与空回复无关，见 _NoMultiGen。别把这两件事混起来。
            if reply and reply.strip():
                break
            if attempt == 1:
                print(f"[ragas] 空回复，重试一次: {q}")
        if reply is None or not reply.strip():
            why = "未调通" if reply is None else "空回复"
            print(f"[ragas] 生成失败（{why}），跳过: {q}")
            continue
        samples.append({
            "user_input": q,
            "collection": coll,
            "retrieved_contexts": contexts,
            "response": reply,
        })
    return samples


def load_ground_truth() -> dict[str, str]:
    """D3：读人工标注。文件不存在/条目没标 → 空串，依赖它的指标跳过。

    每条的值是 `{collection, retrieved_contexts, ground_truth}` 三字段的**字典**
    （见 dump_template），所以**不能对 v 直接 .strip()**——那会
    `AttributeError: 'dict' object has no attribute 'strip'`。这里按字段取。
    同时兼容"直接写成字符串"的简写形态，免得填法被实现绑死。
    """
    if not GT_FILE.exists():
        return {}
    data = json.loads(GT_FILE.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for k, v in data.items():
        text = v.get("ground_truth", "") if isinstance(v, dict) else v
        if isinstance(text, str) and text.strip():
            out[k] = text
    return out


def dump_template() -> None:
    """生成标注模板：question + 检索上下文 + 空 ground_truth，等人填。"""
    from .eval import EVAL_SET
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    template = {}
    for q, coll, _allowed in EVAL_SET:
        hits = search(q, coll)
        template[q] = {
            "collection": coll,
            "retrieved_contexts": [h["text"] for h in hits],
            "ground_truth": "",       # ← 人工填这里
        }
    GT_FILE.write_text(json.dumps(template, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print(f"模板已写入 {GT_FILE}（{len(template)} 条），填 ground_truth 字段后重跑。")

    # 写完立刻用 load_ground_truth 读回来过一遍。模板是嵌套字典、读取端要的是字符串，
    # 这两边一旦对不上，症状是**填完标注、跑分时才**AttributeError（且冒烟测试查不出来：
    # 那会儿文件还不存在，load_ground_truth 在 exists() 那行就返回了）。
    # 空模板读回来必须是 0 条且不抛——这一行就是那个契约的可执行版本。
    back = load_ground_truth()
    assert back == {}, f"模板回读异常：期望 0 条（都还没填），实得 {len(back)} 条"
    print("回读自检通过：模板能过 load_ground_truth，空标注返回 0 条。")


async def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    if "--dump-template" in sys.argv:
        dump_template()
        return

    n = int(os.environ.get("RAGAS_N", "15"))

    base_url, judge_model, key = _pick_judge()
    if not key:
        print("缺少裁判模型 key（DASHSCOPE_API_KEY 或 DEEPSEEK_API_KEY），终止。")
        return
    same_family = base_url == llm.BASE_URL
    if same_family:
        print("=" * 62)
        print("⚠ 裁判 = DeepSeek（与生成同族），自评偏好会虚高。")
        print("⚠ 以下分数只证明流程跑通，不是质量结论；正式跑分请配 Qwen。")
        print("=" * 62)
    else:
        print(f"裁判模型: {judge_model}（不同族，分数可用）")

    print(f"生成评测样本（{n} 条，检索 + DeepSeek 生成）……")
    samples = build_samples(n)
    if not samples:
        # 必须非零退出：这行曾经是 `return`，于是**全军覆没也报 exit 0**——
        # 2026-09-17 网络挂掉那轮，15 条一条没生成，脚本干干净净地"成功"了。
        # 终端里看得见，但任何按退出码判断的调用方（我自己的重跑、将来的 CI）
        # 都会把它当成跑过了。
        print("没有可用样本（生成全失败？），终止。")
        raise SystemExit(1)

    gt = load_ground_truth()
    for s in samples:
        s["reference"] = gt.get(s["user_input"], "")
    n_gt = sum(1 for s in samples if s["reference"])
    print(f"标准答案已标注: {n_gt}/{len(samples)} 条")

    # ---- RAGAS 组装（懒 import：--dump-template 不该背上这套依赖） ----
    from ragas import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    # context_precision 这两个名字原先漏了 import：它们只在 `if n_gt:` 分支里用到，
    # 而标注文件空着时那个分支永远不执行——所以"没标注"和"有标注"两条路上的
    # 脚手架各只跑通一半，合起来才是完整的。补 import 时注意 collections 那个路径
    # 在 0.4.3 下是个 module、没有 single_turn_ascore，跟下面的调用方式不兼容，
    # 别照 DeprecationWarning 的字面去改。
    from ragas.metrics import (LLMContextPrecisionWithoutReference, answer_relevancy,
                               context_precision, faithfulness)

    dataset = [SingleTurnSample(
        user_input=s["user_input"],
        response=s["response"],
        retrieved_contexts=s["retrieved_contexts"],
        reference=s["reference"] or None,
    ) for s in samples]

    # 评测前把 BGE-M3 预热到位：ragas 的事件循环里首次加载模型会卡在
    # transformers 的联网检查上（见模块头注释），预热后循环内只算向量。
    print("预热 BGE-M3 embedding……")
    BgeM3Embeddings().embed_query("预热")

    judge = _NoMultiGen(model=judge_model, api_key=key, base_url=base_url,
                        temperature=0.0)
    wrapped = LangchainLLMWrapper(judge)
    embeddings = BgeM3Embeddings()

    metrics = [faithfulness, answer_relevancy]
    if n_gt:
        if n_gt == len(samples):
            metrics.append(context_precision)
        else:
            metrics.append(LLMContextPrecisionWithoutReference())
    else:
        print("（无标准答案：context_precision 跳过；填好标注文件后重跑可得三项）")

    for m in metrics:
        m.llm = wrapped
        if hasattr(m, "embeddings"):
            m.embeddings = embeddings

    # 不走 ragas.evaluate()：它的作业线程吞异常只留 AttributeError，
    # 逐条调 single_turn_ascore 能看到每个样本的真实报错，也方便重试。
    rows = []
    for i, sample in enumerate(dataset):
        row = {"user_input": sample.user_input}
        for m in metrics:
            try:
                row[str(m.name)] = await m.single_turn_ascore(sample)
            except Exception as e:
                print(f"[ragas] {m.name} 打分失败（{i + 1}）: {type(e).__name__}: {e}")
                row[str(m.name)] = None
        rows.append(row)
        done = ", ".join(f"{k}={v:.2f}" if v is not None else f"{k}=失败"
                         for k, v in row.items() if k != "user_input")
        print(f"  [{i + 1}/{len(dataset)}] {done}")

    print(f"\n===== RAGAS 生成侧评测（{len(samples)} 条）=====")
    for m in metrics:
        vals = [r[str(m.name)] for r in rows if r[str(m.name)] is not None]
        label = "（同族裁判，仅验流程）" if same_family else ""
        print(f"  {m.name:28s} {sum(vals) / len(vals):.3f}  (n={len(vals)}) {label}"
              if vals else f"  {m.name:28s} 全部失败")

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"逐条明细: {RESULT_FILE}")
    print("\n解读提醒：faithfulness 低 = 编造；answer_relevancy 低 = 答非所问；")
    print("context_precision 与 eval.py 的 MRR 互为印证（一个 LLM 判、一个位置算）。")


if __name__ == "__main__":
    asyncio.run(main())
