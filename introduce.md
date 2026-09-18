# ShopMate 项目全讲解（introduce）

> 这份文档写给**第一次接触本项目的人**：不要求你读过任何一行代码，
> 读完后应该能回答四个问题——这是什么项目？它由哪些部分组成、互相怎么衔接？
> 每个文件是干什么的？它是怎么一步步做出来的、现在到了哪一步？
>
> 更深入的内容都有出处：设计依据在 `docs/01~04`，怎么运行在 `README.md`，
> 怎么演示在 `docs/05`。本文是它们的"导览 + 串讲"。

---

## 一、这是什么项目

**ShopMate** 是一个 AI 电商客服与导购系统（对话机器人名叫「小搭」），用一个大模型
模拟电商平台的智能客服，覆盖这些场景：

- **商品咨询**："这耳机续航多久？"、"冲锋衣能用柔顺剂洗吗？"
- **口碑评价**："这款耳机有什么缺点？"（好评差评都要讲，不许只挑好听的）
- **参数对比**："SKU-10001 和 SKU-20001 哪个好？"
- **个性化推荐**："预算六百，通勤用，求推荐个耳机"
- **订单/物流/价格/库存查询**：实时数据，走业务系统接口
- **售后办理**：退货/换货/维修（写操作，必须用户二次确认）
- **转人工**：用户点名、或系统判断撑不住时，附对话摘要转人工

项目性质是**学习/求职向的完整工程演示**（README 里有"面试可讲"章节），所以它有一个
鲜明特点：**不追求"能跑就行"，而是把工程上的关键取舍都做实并写明理由**——
每个模块的 docstring 都写着"设计决策 D1/D2/…"，每篇设计文档都记录了实测推翻了
哪些假设，README 的"已知局限"一节甚至专门列出"哪些数字不能吹"。

技术栈一句话（**每一样是什么、为什么用它，见第四节 4.1 的总表**）：
**Python 3.10+ · LangGraph（Agent 编排版）· LangChain（向量路集成）· ChromaDB（向量库）·
BGE-M3（中文向量模型）· bge-reranker-v2-m3（精排）· BM25 + jieba（关键词检索）·
DeepSeek API（大模型，OpenAI 兼容协议）· RAGAS + Qwen-plus（生成侧评测与异族裁判）·
SQLite（checkpoint 持久化）· Streamlit（演示前端）· FastAPI + uvicorn（薄服务层）**。
状态机有**手写与 LangGraph 两版并存**（环境变量切换，默认手写版）；检索器同样有
手写与 LangChain 两版，共用同一套融合逻辑，评测集证明两版逐项一致。
2026-09-18 起多了一个**只做薄壳的 HTTP 服务层**（`app/api/`，6 个端点），
让同一个 Agent 能被 curl / 任何程序调用——边界为什么不继续做厚，见 §13.1。

---

## 二、用户视角：走一轮对话是什么样

以浏览器演示（`streamlit run app/agent/webui.py`）为例：

```
你：SKU-10001 现在多少钱？
小搭：目前原价 799 元，活动价 599 元，还能叠加新人券再减 20 元。
      （左侧栏同时显示：意图=order_query，置信度 0.9x，工具调用 query_price → ok）

你：那它续航怎么样？
小搭：单次续航 8 小时，配合充电盒总续航 32 小时。
      （侧栏显示：意图=product_consult，RAG 检索 product_knowledge 库，
        召回 5 块，其中 spec 块被引用）

你：这个耳机口碑怎么样？
小搭：好评 82%（集中在降噪、续航），差评 12%（高频降噪一般、充电盒偏大）……
      最后给一句适用建议。   ← 第 8 个意图 review_consult，专门查"评价库"

你：ORD-20260908-002 的冲锋衣尺码偏大，我要退货
小搭：即将为您提交退货申请：订单 … 商品 …，原因：尺码偏大。确认提交吗？
      （侧栏出现"待确认的写操作"）          ← 模型自己不许说"请确认"，系统统一管

你：确认
小搭：已受理！工单号 AS-20260901……       ← 确认后才真正执行
```

每一轮对话，**内部发生了什么（走了哪个分支、检索到什么、调了什么工具、有没有触发
兜底）都摊在侧栏上**，这是本项目区别于"套壳 ChatGPT"的核心展示点。

---

## 三、整体架构：一张图看懂

```
用户输入（CLI 或 Streamlit 前端）
   │
   ▼
┌─────────────────────────────────────────────────────┐
│  Agent 状态机（app/agent/graph.py，自研，无框架）      │
│                                                     │
│  ① 确认门：上一轮有待确认的写操作？                    │
│     └─ 是 → 关键词判 同意/拒绝/听不懂，本轮结束        │
│  ② 意图识别（LLM，json 输出）                         │
│     ├─ 8 类意图 + 置信度 + 槽位 + 是否不满            │
│     ├─ 置信度 < 0.6 → 追问澄清（兜底1）               │
│     └─ 连续 2 次不满 → 主动转人工（兜底4）             │
│  ③ 按意图路由：                                      │
│     ├─ 商品咨询/口碑/对比/推荐 ──→ RAG 检索 ──┐        │
│     ├─ 订单查询/售后 ──→ 工具编排(LLM循环) ──┤        │
│     ├─ 点名转人工 ──→ transfer_to_human      │        │
│     └─ 闲聊 ──→ 直接生成回复                 │        │
│  ④ 回复生成（LLM）←─────────────────────────┘        │
│     兜底：检索空 →"暂无相关信息"；LLM挂 → 降级话术；    │
│           工具连败2次 → 转人工；编排超3轮 → 收尾       │
└─────────────────────────────────────────────────────┘
   │                    │                      │
   ▼                    ▼                      ▼
RAG 检索层            工具层                 LLM 层
app/retrieval        app/tools             app/llm
向量+BM25 混合        6 个工具定义(JSON)      DeepSeek 封装
RRF 融合             mock 实现              (OpenAI 兼容)
ChromaDB 三库        执行器：确认门/超时/    重试策略收口在
BGE-M3 向量化        缓存/日志/身份注入      safe_call
```

四层职责一句话概括（**服务层**是 2026-09-18 加上去的一层薄壳，附在最后）：

| 层 | 目录 | 职责 |
|---|---|---|
| 检索层 | `app/retrieval/` | 把 32 份知识文档建库（分块→向量化→入库），提供"给一句话、还回最相关的 5 个知识块"的混合检索，以及评测和埋点 |
| 工具层 | `app/tools/` | 定义 6 个业务工具（查订单/库存/价格/物流、售后申请、转人工），统一执行入口管住超时、缓存、日志和"写操作必须确认" |
| Agent 层 | `app/agent/` | 状态机：意图识别 → 路由 → 检索或调工具 → 生成回复，带四条兜底、会话记忆和本轮轨迹 |
| LLM 层 | `app/llm/` | 全项目**唯一**的大模型出口：DeepSeek 客户端、连接类故障重试一次、失败返回 None 让上层各自降级 |
| 服务层 | `app/api/` | **极薄**的一层：把 `agent.handle(sid, text)` 包成 6 个 HTTP 端点、把轨迹变成可 curl 的 JSON。**不含任何业务逻辑**，也不 import 任何业务模块 |

数据都在 `data/`（知识文档、工具定义、向量库、运行日志），设计文档在 `docs/`。

---

## 四、技术栈与名词小课堂（零基础看这节）

> 这节有两个作用：**4.1 回答"这项目用了什么、每样为什么用"**，4.2 把名字翻译成人话。
> 每样技术尽量写清三件事——**它是什么、在本项目里干哪件活、为什么是它而不是别的**。
> 读源码卡住时回这节查名字。

### 4.1 技术栈总表

| 层 | 技术（版本） | 在本项目里干的活 | 为什么是它 |
|---|---|---|---|
| 语言 | **Python 3.10+** | 全项目 | 路线要求；且 Agent 生态（LangChain / LangGraph / RAGAS）都是 Python 优先 |
| Agent 编排 · **框架版** | **LangGraph 1.2.11** | `lg_graph.py`：把状态机画成图（节点 / 条件边 / 循环 / 中断） | 手写版踩过坑之后再换框架对照；`checkpoint` 与 `interrupt` 是手写版给不了的能力 |
| Agent 编排 · **手写版** | **无框架**（自研状态机） | `graph.py`：if/else 分支 + 显式 Session 状态，**默认实现** | "先裸写再上框架"——不自己写一遍，理解不了框架替你做了什么 |
| 检索集成 | **LangChain**（`langchain-core` 1.6.3、`langchain-chroma` 1.1.0） | `lc_retriever.py`：只把**向量路**换成 LangChain 的 Embeddings + Chroma 封装 | 对照实验要控制变量：BM25 路 / RRF / 词法门 / 加权 / 阈值**全部复用**手写版，只变一处才能说清差异从哪来 |
| 向量库 | **ChromaDB 1.5.9** | 存 124 个知识块的向量，按 collection 分成三库 | 本地持久化、零运维；8 SKU 规模上 Milvus 的分布式能力一点用不上 |
| 向量模型 | **BGE-M3**（`FlagEmbedding` 1.4.2） | 文档与问题都变成 1024 维向量，语义相似度靠算距离 | 中文效果好、开源免费、能离线；约 2.3GB，进程内只加载一次 |
| 关键词检索 | **rank-bm25 0.2.2 + jieba 0.42.1** | 词法路打分；jieba 负责中文分词（中文不能按空格切） | 与向量互补：向量管"意思像"，它管"字面准"——用户报型号 `SKU-10001` 这种精确串，向量反而排不上 |
| 精排（rerank） | **bge-reranker-v2-m3**（随 FlagEmbedding 带入，本地推理） | 第五路检索：混合初检 top-20 → cross-encoder 精排 → top-5 | 两阶段检索是工业界标准做法；本地跑不花钱、不联网、不依赖第三方服务 |
| 大模型 | **DeepSeek**（`deepseek-chat`，走 `openai` 3.13.0 客户端） | 意图识别 / RAG 生成 / 工具编排 / 闲聊——**全项目唯一出网口** | 便宜、中文稳、且**OpenAI 兼容协议**：换供应商只改 `base_url` 和模型名 |
| 裁判模型 | **Qwen-plus**（阿里云百炼的 OpenAI 兼容端点） | 给 RAGAS 打分（评"这条答案好不好"） | **必须与生成侧不同族**——同族等于让模型评自己，有自评偏好、分数虚高 |
| 评测框架 | **RAGAS 0.4.3** | 生成侧三指标：faithfulness / answer_relevancy / context_precision | RAG 生成质量的事实标准；检索指标（hit@5）证明得了"找得准"，证不了"答得没编" |
| 持久化 | **SQLite**（`langgraph-checkpoint-sqlite` 3.1.1） | `data/graph_checkpoints.sqlite3`：存状态机每一步的快照 | 单文件、零运维，换来"进程重启后确认门仍可续跑"——手写版做不到 |
| 演示前端 | **Streamlit 1.63.0** | `webui.py`：聊天界面 + 侧栏摊开内部状态 | 纯 Python 写网页、不碰前端三件套；自带的 `AppTest` 还能跑无头端到端测试 |
| **服务层** | **FastAPI 0.141.1 + uvicorn 0.52.4** | `app/api/`：把 `agent.handle()` 包成 6 个 HTTP 端点；把每轮内部决策轨迹从侧栏里解放成**任何人可 curl 的 JSON** | 需要"能被别的程序调用"这个能力（curl / 前端 / 自动化评测都要它），而 CLI 和 Streamlit 都是给人用的。**刻意做得极薄**：3 个文件、6 个端点、不 import 任何业务模块——它一厚起来就不再是"服务化"而是"多一层要维护的代码" |
| 日志 | **JSONL 文件**（`data/logs/`） | 每次工具调用、每轮检索各追加一行 | 结构化、Python / `jq` 直接读、追加写不怕中途崩 |

**刻意不在技术栈里的**（是判断，不是欠债）：MySQL / Redis / Milvus / SSE / 鉴权 / CORS / 多副本。
理由写在 README「明确不做」——8 SKU 的演示量级下换上去，收益是零。
（**FastAPI 已从这一列移走**：2026-09-18 起它做了，但只做薄的那一层；做厚的那几项
——SSE / 鉴权 / CORS / 多副本——仍然是明确的"不做"，理由见 §13.1。）

> **两条实现的并存关系**（本项目最值得讲的一处工程选择）：
> 状态机有**手写版 `graph.py` 与 LangGraph 版 `lg_graph.py`**，检索器有**手写版 `retriever.py`
> 与 LangChain 版 `lc_retriever.py`**，都由环境变量切换（`SHOPMATE_AGENT=lg` /
> `SHOPMATE_RETRIEVER=lc`），手写版是默认。**换实现不换行为**——两版共用同一套融合逻辑
> 与提示词，评测集当裁判（LangChain 版与手写版同为 90/94/96/100）。

### 4.2 名词小课堂

**检索类**

- **RAG**（Retrieval-Augmented Generation，检索增强生成）：让大模型回答前先去知识库里
  查资料，把查到的资料和问题一起给模型，减少胡编（幻觉）。本项目里"商品咨询、口碑、
  对比、推荐"四类问题走这条路。
- **Embedding / 向量**：把一段文字变成一串数字（向量），语义相近的文字向量也相近，
  于是"意思层面的搜索"变成"算向量距离"。
- **BGE-M3**：开源中文向量模型（BAAI 出品），把文档和问题都变成 1024 维向量。约 2.3GB，
  首次加载要十几秒（所以前端有"预热模型"按钮）。
- **BM25**：经典关键词检索算法，按词频和稀有度打分。向量管"意思像"，BM25 管"字面准"
  （比如用户报型号"SKU-10001"这种精确字符串，向量反而排不上）。
- **混合检索（hybrid）**：向量路和 BM25 路各查一遍，再合并结果。两路错的地方不一样，
  合起来才 96%（单路只有 90% 和 94%）。
- **RRF**（Reciprocal Rank Fusion，倒数排名融合）：两路检索各给一份排名，
  每个候选按"排名的倒数"加分后求和。好处是不用把两路分数硬拉到同一量纲。
- **Chunk（分块）**：把长文档切成小段再入库，检索时只取相关的小段——省 token、更准。
  **切法比参数更重要**：实测把定长 512 字的大块拿去喂 BM25，命中率从 94% 崩到 68%
  （一个块混了多个主题，关键词被稀释）。
- **ChromaDB**：轻量向量数据库，本项目用它持久化向量并按元数据过滤。
- **hit@5 / hit@1 / MRR**：三个排序类检索指标。hit@5 = 正确答案进了前 5 名就算命中；
  hit@1 = 必须排第一；MRR = 第一个正确答案排名的倒数取平均（排第 1 得 1.0，排第 3 得 0.33）。
  **hit@5 顶到 100% 之后就问不出新问题了（"指标饱和"）**，此时要靠 hit@1 / MRR 才能看出
  rerank 带来的提升。
- **rerank / cross-encoder**：两阶段检索的第二阶段。第一阶段（向量+BM25）要快，所以
  问题和文档是**分开**算向量的，互相看不见；第二阶段用 cross-encoder 把"问题+段落"
  **拼在一起**过一遍模型，精度高但慢得多，所以只对初检回来的 20 条重排。

**生成与评测类**

- **幻觉（hallucination）**：模型一本正经地编造事实。RAG 和"只依据资料回答"的提示词都是为了压它。
- **grounding（接地）**：让模型的每句话都落在给它的资料上，不许自由发挥。本项目 RAG 支路
  的提示词明确要求"资料里没有就说没有"。
- **RAGAS**：RAG 生成侧的评测框架，本项目用了三个指标——
  **faithfulness**（答案是否忠于检索到的资料，防幻觉的量化证据）、
  **answer_relevancy**（有没有答到点子上）、
  **context_precision**（相关资料排得够不够靠前，与 MRR 互相印证）。
- **LLM-as-judge / 异族裁判**：用一个大模型当"阅卷老师"给另一个模型的答案打分。
  **裁判必须与考生不同族**——让 DeepSeek 评 DeepSeek 就是自己评自己，分数会虚高。
  所以生成用 DeepSeek、裁判用 Qwen（判分不对时脚本会打警告横幅，那模式下的分不能进文档）。
- **代理指标（proxy metric）**：真答案太贵或算不出来时，用一个便宜的相关信号顶替。
  本项目有两处，都如实标注了：`telemetry` 里的"引用率"是**字符 4-gram 重叠率**，
  度量的是"模型抄了多少字面"而**不是准确率**；RAGAS 的 `context_precision` 也是检索质量的
  代理，真值还得看 `eval.py` 的 MRR。**用代理指标必须写清它的盲区**——这是本项目的规矩。

**Agent 编排类**

- **Function Calling**：让大模型在对话中"决定调用某个函数并填参数"，函数执行结果再喂回
  给模型组织答案。本项目里"订单/库存/价格/物流"这类**实时数据**不走知识库，只走这条路。
- **意图识别（intent）**：每次对话开头先问模型"用户想干嘛"，8 类意图决定后面走哪条路。
  同时抽出**槽位**——从话里抽出的结构化信息（商品 ID、订单号），"它还有什么缺点"里的"它"
  就靠上一轮记住的商品 ID 补上。
- **确认门（confirmation gate）**：写操作（退货、转人工）执行前拦住，等用户点头才继续。
  用户回"确认"时用纯关键词判定、不调 LLM——零成本且不会猜错。**放在意图识别之前**，
  免得模型去猜"算了不换了"是什么意思。
- **工具编排**：一轮对话里连续调多个工具（先查订单 → 再查物流 → 再申请售后）。
  必须设轮数上限（本项目 3 轮），否则模型可能无限调下去。
- **兜底（fallback）**：主流程走不通时的备用出路（澄清追问、转人工、"暂无相关信息"话术）。
  本项目有四条，README 里标注了**哪两条演示现场不好触发、别赌**。
- **LangGraph / StateGraph**：把 Agent 流程画成图的框架。**节点**是"干一件事"，
  **边**是"下一步去哪"，**条件边**（conditional_edges）是按上一步结果走不同分支的箭头
  （手写时就是 `if/else`），**循环**是边指回前面的节点（手写时就是 `while`）。
  好处是流程一目了然、且框架帮你管状态。
- **checkpoint（检查点）**：把流程图每一步的状态快照存下来。换来两个能力：进程崩了/重启
  能从断点续跑；以及"时间旅行"——回滚到某个中间步骤调试。本项目用 SQLite 存。
- **interrupt() / Human-in-the-loop（HITL）**：让流程执行到某一步就**停下来**，等真人批准
  再往下走，而不是一股脑跑完。本项目的写操作确认门就是它实现的。
- **supervisor / 子代理**：一个"老板 Agent"把大任务拆给多个"员工 Agent"去干，收活后汇总。
  **本项目只到"知道是什么"，没有实现**——属于 W8 的内容，别在面试里说做过。
- **trace（执行轨迹）**：把 Agent 每一步"想了什么、做了什么、看到什么"记下来。本项目的
  轨迹摊在前端侧栏上，是区别于"套壳 ChatGPT"的核心展示点。

**工程类**

- **幂等（idempotent）**：同一个操作执行一次和执行三次，结果一样。本项目建库用同 ID 覆盖，
  所以可以反复重建不会翻倍。
- **懒加载（lazy loading）**：用到的时候才加载，不是一 import 就加载。BGE-M3 有 2.3GB，
  所以做成了懒加载单例——不检索就不加载。
- **JSONL**：一行一个 JSON 的日志格式。好处是追加写不怕崩、又可以直接按行读成结构化数据。
- **token**：模型计费和容量的单位，可以粗略理解成"字"。chunk 上限 750 字符（≈512 token）
  就是这么换算来的。

---

## 五、目录总览（每个文件是干什么的）

```
ShopMate/
├── introduce.md            ← 本文件
├── README.md               项目门面：架构图、快速开始、设计要点表、已知局限
├── requirements.txt        直接依赖 15 个，分层列清（检索 4 / LLM 1 / LangGraph·LangChain 7 / 前端 1 / 服务层 2，另有 2 个规划中注释掉）
├── .env.example            DEEPSEEK_API_KEY 和代理的填写模板（复制成 .env 用）
├── LICENSE                 MIT
│
├── app/                    全部源码，共 25 个模块（下面三、四、五节逐层讲）
│   ├── retrieval/          检索管线（11 个：schema/loader/chunker/indexer/retriever/eval
│   │                       /telemetry/reranker/chunk_compare/ragas_eval/lc_retriever）
│   ├── tools/              工具层（3 个：registry / mock / executor）
│   ├── agent/              Agent 状态机（8 个：session/trace_view/intent/graph
│   │                       /lg_graph/runtime/cli/webui）
│   ├── api/                薄服务层（2 个：schemas / server）—— 不含业务逻辑
│   └── llm/                LLM 封装（1 个：client）
│
├── data/
│   ├── rag_docs/           RAG 知识库源文档：32 份 / 3 品类 / 8 SKU
│   │   ├── products/       商品详情 md ×8
│   │   ├── specs/          规格参数 json ×8（含价格快照）
│   │   ├── reviews/        用户评价摘要 md ×8
│   │   ├── faq/            品类 FAQ ×3（耳机/服装/家电）
│   │   ├── policies/       平台政策 ×2（退换货/物流）
│   │   └── guides/         选购指南 ×3
│   ├── tools/              工具定义 JSON ×6（给 LLM 看的"工具说明书"）
│   ├── eval/               评测标注与产物（**整目录 gitignore**）
│   │   ├── ragas_ground_truth.json  50 条**人工**标注的标准答案 ← 唯一不可重建的产物
│   │   └── ragas_results.json       RAGAS 逐条明细（可重跑得到）
│   ├── chroma_compare/     chunk 策略对比实验的独立库（三种切法各一套，可重建）
│   ├── graph_checkpoints.sqlite3   LangGraph 状态快照（lg 版确认门断点续跑用）
│   ├── chroma/             ChromaDB 持久化目录（3 个 collection，可重建）
│   └── logs/               运行日志 JSONL ×2：tool_calls（工具）/ retrieval（检索埋点）

> **⚠️ 关于 `data/eval/` 被 gitignore**：`.gitignore` 里这条注释写的是
> "RAGAS 评测产物（**可重建**）"——这对 `ragas_results.json` 成立（重跑一次就有），
> 但**对 `ragas_ground_truth.json` 不成立**：那是 50 条人工撰写的标准答案，
> 是"必须人工、AI 不能替"的产物，**删掉/换机器就没了，且 clone 下来的人无法复现
> RAGAS 分数**。这一条按"运行产物"归类是错的，属于该修正的项。
│
└── docs/                   设计文档（决策依据全在这里）
    ├── 01_rag_knowledge_base.md   知识库：分域/分块/混合检索/评测结论（最厚的一篇）
    ├── 02_tool_definitions.md     工具总览与调用原则（只读/写操作分级）
    ├── 03_agent_workflow.md       状态机：节点/意图路由/置信度兜底
    ├── 04_data_schema.md          元数据/Collection/MySQL/Redis 规范
    └── 05_demo_runbook.md         演示手册：怎么跑、问什么、看哪里、别演哪两条
```

---

## 六、数据底座：data/ 里有什么

### 6.1 知识库 `data/rag_docs/`——32 份文档，6 种类型

模拟一个只卖 3 个品类的电商：**耳机 ×4、冲锋衣 ×2、洗衣机 ×2**（SKU-10001~40002）。
文档刻意造了"跨品类同义词冲突"的用例（比如"延迟"在耳机是参数、在洗衣机是噪音），
用来检验检索会不会串味。

6 种文档类型对应 6 种切法（检索层 chunker 的输入）：

| 目录 | doc_type | 内容举例 |
|---|---|---|
| products/ | product | 商品详情：卖点、适用人群、不适用场景、关联商品 |
| specs/ | spec | 结构化 JSON：参数、**价格快照**（original 799 / current 599） |
| reviews/ | review | 评价摘要：好评关键词 82%、差评 12%、典型摘录、导购建议 |
| faq/ | faq | 一问一答："苹果手机能用吗"、"保修多久" |
| policies/ | policy | 七天无理由、运费谁出、发货时效 |
| guides/ | guide | "预算三百以内怎么选耳机" |

**一个重要边界**：价格和库存是**实时数据，不进知识库**——specs 里的价格只是"发布时
快照"。用户问"现在多少钱"必须走工具层查（所以演示里答案永远是活的 599+券）。

### 6.2 工具定义 `data/tools/`——6 份 JSON

每份 JSON 声明工具名、用途描述、参数 schema、返回结构（OpenAI function-calling 格式，
"定义"与 Python "实现"分离，将来接真实业务系统只换实现）。四个只读
（query_order / query_stock / query_price / track_logistics），两个写操作
（apply_after_sale / transfer_to_human）。

### 6.3 向量库 `data/chroma/` 与日志 `data/logs/`

- `chroma/` 是建库产物（`python -m app.retrieval.indexer` 可随时重建），分 3 个
  collection（见下节）。
- `logs/` 每次运行追加两份 JSONL：`tool_calls.jsonl`（每次工具调用：用户、工具、参数、
  结果、耗时）和 `retrieval.jsonl`（每轮 RAG：query、召回块、引用判定、outcome、耗时）。
  都被 gitignore，是运行产物。

---

## 七、检索层 app/retrieval：从文档到"最相关的 5 块"

这是本项目做得最扎实的一层。管线五步 + 两个附加模块，**依赖顺序**：

```
loader → chunker → indexer → retriever → eval
（读文件） （分块）  （向量化+入库） （混合检索）  （50 条评测）
                         ↑
              telemetry（检索埋点，被 Agent 层调用）
```

### schema.py —— 常量与数据结构
所有"魔法数字"集中在这里：6 类文档 → 3 个 collection 的映射表、TOP_K=5、RRF_K=60、
兜底阈值 SCORE_THRESHOLD=0.012、向量距离硬门槛 SEMANTIC_MAX_DIST=0.45。
两个数据类：`RawDoc`（一个文件的原文+元数据）、`Chunk`（一块可向量化的文本）。

### loader.py —— 第 1 步：加载
遍历 rag_docs 六个子目录产出 RawDoc。细节：从文件名提取 product_id（`SKU-10001_reviews.md`
→ `SKU-10001`）；第二遍扫描**交叉回填** category/brand——类目只有 spec json 里有、品牌只有
商品 md 里有，所以要建查找表回填到所有文档。

### chunker.py —— 第 2 步：分块（124 块）
按文档类型分六种切法：md 按 `##` 切、FAQ 按问答对切、spec 整文件一块（JSON 展平成
"键: 值"文本行，bool 转是/否、null 跳过）、评价按好评/差评关键词分块。单块上限 750 字符
（≈512 token），超长在中点附近找换行下刀，不拦腰切句子。

两个关键设计：
- **商品锚点**：每块头部拼 `【SKU-10001 无线降噪蓝牙耳机 Pro | SoundCore】`。
  没有它，评价文档的标题（含 SKU 号）不入块，用户报型号提问时 BM25 无从匹配——
  这是评测发现的漏召回来源，锚点里**必须带商品 ID**。
- **父标题**：政策/FAQ 的块首拼上文档级标题（"特殊品类规则"脱离"退换货政策"会歧义）。

### indexer.py —— 第 3 步：向量化 + 入库
BGE-M3 懒加载单例（约 2.3GB，只加载一次；检测到本地缓存就离线加载，不再联网），
批量向量化后按 collection 分组 upsert 进 ChromaDB（同 ID 覆盖，幂等，可反复重建）。
cosine 相似度，1024 维。

**三个 collection 的划分是"按知识域"而不是"按文档类型"**：

| Collection | 装什么 | 为什么 |
|---|---|---|
| product_knowledge | product + spec + faq + guide | 商品咨询一次召回四类资料 |
| policy_knowledge | policy | 政策独立，误召回代价高 |
| review_knowledge | review | 评价是口语化文本，混进商品库会**污染相似度** |

### retriever.py —— 第 4 步：混合检索（本层核心）

`search(query, collection)` 的内部流程：

```
query
 ├─ 向量路：BGE-M3 向量化 → ChromaDB top-5 → cosine 距离 > 0.45 的丢掉
 ├─ BM25 路：jieba 分词（停用词过滤）→ rank-bm25 打分 top-5
 ▼
RRF 融合：score = Σ 1/(60 + rank)      ← 只看排名，两路都靠前的总分高
 ▼
词法置信门：只有 BM25 支持的候选，须与 query 相交 ≥2 个实词
            （拦"量子涨落对股市的影响"靠"对/的"攒分的假命中）
 ▼
doc_type 加权：只保留 spec ×1.3（价格只信结构化事实），其余一律 1.0
 ▼
最高分 < 0.012 → 返回空（上层走"暂无相关信息"兜底）
```

另一个入口 `search_with_product_focus(query, collection, product_id)` 是**话题商品锚点**：
用户明显在聊某件商品时（报了型号，或上一轮聊过），先无过滤检索（召回上界不变），
把该商品的块提到前排；该商品没占住一半席位时，再带 `where` 过滤补一次检索。
**刻意不做独占式过滤**——FAQ/政策/指南的 product_id 是空串，独占过滤会把它们整体排除，
而"耳机保修多久"的正解恰恰是 FAQ。

BM25 语料直接从 ChromaDB 里捞（不重读文件），保证两条路检索的永远是同一份数据；
索引在进程内存按 collection 缓存（重启重建，README 已声明这个局限）。

### telemetry.py —— 检索埋点
每轮 RAG 往 `data/logs/retrieval.jsonl` 落一行：query、5 个召回块（预览 120 字）、
RRF 分、耗时、outcome（answered / no_info / llm_fail）、引用判定。
埋点写在检索包、但**调用方在 Agent 层**——因为"回复"只有 Agent 层才有，
而且焦点检索内部会再调一次 search，记在检索器里会一个 query 落两行、命中率凭空翻倍。

**引用判定是一个诚实的"代理指标"**：用字符 4-gram 重叠率猜"这块资料用没用上"
（并扣掉本轮各块共有的样板文字，防止回复提一句商品名就把该商品所有块判成被引用）。
代码里明写它的盲区：同义改写会漏报；实测**同一条 query 跑两次，召回完全相同，
引用块数却是 1 和 3**——差异全在模型措辞。所以它度量"模型抄了多少字面"，
方差比绝对值更值得注意，**不能当准确率引用**。

### eval.py —— 50 条评测集，五路对照 + 三个指标
50 条真实客服话术改写的问题，人工标注"可接受的正确文档集合"（一条问法往往 2~3 个
文档都算对）和"发出时正在聊哪件商品"。跑五路对照：

| 检索方式 | hit@5 | hit@1 | MRR |
|---|---|---|---|
| 纯向量 | 90% | 70% | 0.776 |
| 纯 BM25 | 94% | 68% | 0.788 |
| RRF 混合 | 96% | 72% | 0.809 |
| **混合 + 话题商品锚点** | **100%** | **88%** | **0.926** |
| 混合 + bge-reranker 精排 | 96% | 74% | 0.826 |

**为什么从"四路"变成"五路 + 三个指标"**：hit@5 在第四路上已经顶到 100%，**指标饱和**
了——"进没进前五"问不出新问题。补上 hit@1 和 MRR 之后，排序质量的差异才重新可见
（锚点路 88% / 0.926，rerank 路 74% / 0.826，混合路 72% / 0.809）。

三点读表须知（比数字本身重要）：

- **不吹"+30%"**：小语料上 BM25 单路已 94%，混合的真实价值是补齐两条单路各自漏掉的查询；
  且每一路相对上一路**零倒退**（eval 每次都打这个差值，负了就是回归）。
- **锚点路的 100% 不等于"最强"**：focus 用了额外输入（知道用户在聊哪件商品），
  与其它路不同口径；rerank 真正的对照对象是**混合路**（+2pp hit@1、+0.017 MRR）。
- **rerank 收益温和是评测集的属性，不是 rerank 不行**：50 条里多数 query 正解本就排
  前一两名，精排的作为是"把第 3~5 名顶上去"，这种 case 在这份集子里只有零星几条；
  真实语料（初检更脏）上收益空间更大。

rerank 路要显式开启：`SHOPMATE_RERANK=1 python -m app.retrieval.eval`——不设这个变量，
表里根本不会有 rerank 那一行。**这里修过一个会印假数字的缺陷**：早先 rerank 未启用时，
表头除的是评测集总数而非该路样本数，于是**不报 ZeroDivisionError、安静印出一行
`rerank 0% 0% 0.000`**——照着 README 跑的人会以为 rerank 坏了、或把 0% 当结论抄走。
现在只列真正跑过的路。教训：**分母选错会让"没有数据"伪装成"零分数据"，这比抛异常危险。**

调参过程中实测推翻了三处设计假设（阈值 0.02→0.012、加权只留 spec、锚点要带商品 ID），
教训都回写进了 docs/01 和代码注释。

### reranker.py —— 精排（第五路，两阶段检索）
cross-encoder 版重排：混合初检 top-20 → bge-reranker-v2-m3 精排 → 取 top-5。
`SHOPMATE_RERANK=1` 开启。模型本地推理（快照在 `models/`，gitignore），不花钱不联网。

**它和 W5 的"话题商品锚点"重排是两回事，别混**：cross-encoder 是**模型精排**——把
"问题 + 段落"拼成一条送进模型打分，所以准；代价是慢，只能对少量候选做。话题锚点是
**业务规则**——"用户在聊某件商品，就把该商品的块提前"，不调模型。表格里两者不可互相
替代（focus 依赖"知道用户在聊哪件商品"这条外部信息）。

### chunk_compare.py —— chunk 策略对比（三种切法各建一套库）
同一评测集、判到文档级，比较"按类型切"（现状 typed）与两种定长切法：

| 策略 | 块数 | 平均块长 | hybrid hit@5 / hit@1 / MRR |
|---|---|---|---|
| **typed（现状）** | 124 | 149 字 | 96% / 72% / 0.809 |
| fixed512 | 42 | 393 字 | 98% / 72% / 0.830 |
| fixed256 | 86 | 206 字 | 98% / 76% / **0.857** |

三条结论（**比数字本身重要**）：

- **定长大块会弄残 BM25**：fixed512 的词法路从 94% 崩到 68%——一个块塞进多个主题后，
  词被稀释、IDF 区分度下降。这是"分块粒度是词法检索生命线"的量化证据。
- **fixed256 在检索指标上反超 typed，但要打折看**：判到文档级时，同一文档块越多、
  进 top5 的机会越多，这里的优势**部分来自口径而非纯质量**；而 typed 块的标题归属和
  节边界是为**生成侧**服务的（引用能指到节、答案自含上下文），检索指标量不到这部分。
- **选型维持 typed**：检索侧差距在噪声量级（MRR 0.809 vs 0.857），而结构化块在生成侧
  的收益是定长策略结构上给不了的。若将来只做检索不做生成（纯搜索接口），fixed256
  是值得考虑的便宜方案。

### ragas_eval.py —— 生成侧评测（RAGAS 三指标）
检索指标只能证明"找得准"，证不了"答得没编"。这个脚本对评测集前 N 条跑**真实检索 +
真实 RAG 生成**（prompt、检索方式、资料拼接格式全部从 `app.agent.graph` 原样复用，
评的就是线上那套路径），再交给 RAGAS 的三个指标打分。

实测（2026-09-17，前 15 条，裁判 qwen-plus）：

| 指标 | 分数 | 可信度 |
|---|---|---|
| faithfulness | **0.871** | 可用 |
| context_precision | **0.812** | 可用 |
| answer_relevancy | 0.542 | **不可信，别当结论引用** |

`answer_relevancy` 为什么不可信（这条值得单独学）：它要**先用答案反解出问题**，
再与原问题比向量余弦——而生成侧 `temperature=0.3`，每轮措辞都不同，反解出的问题
也就不同。同一条 query 复跑一次 0.000、一次 0.813。**faithfulness 和 context_precision
没有"反解"这一步，不受影响。** 所以这项要报就报多轮区间，不能报单轮均值。

用法：`RAGAS_N=15 python -m app.retrieval.ragas_eval`（`RAGAS_N` 控制条数，裁判调用按条计费）；
`--dump-template` 生成人工标注模板。**全量 50 条尚未跑**，正式分数以那一轮为准，
详见 `docs/01` §七。

---

## 八、工具层 app/tools：让模型"能动手"但动不了不该动的

### registry.py —— 定义加载
读 6 份 JSON 校验（名字冲突、缺参数、required 不在 properties 里都当场抛错），
产出给 LLM 的 tools 列表。**写操作清单硬编码在代码里**（`WRITE_OPS`），不放在 JSON 里——
安全级别是工程约束，必须由代码唯一裁决。`is_write_op` 对**未知工具名返回 True**：
宁可多问一次确认，不可漏一次确认。

### mock.py —— 模拟业务实现
6 个工具的假实现：价格从 specs 快照读，库存/订单/物流是内存表（u1001 有两笔订单：
一笔在途可查物流，一笔已完成可走售后；SKU-10003 刻意断货、有补货日期）。
售后申请会**真的**往内存售后表插记录、递增工单号——让"提案→确认→执行"有东西可测。
查不到就抛 `ToolInputError`，由执行器统一转结构化错误，mock 绝不返回半吊子的
`{"error":...}` 混进正常数据。

### executor.py —— 统一执行入口（安全设计都在这）
`execute(name, arguments, user_id, confirmed)` 是 Agent 调工具的**唯一**入口：

1. **身份注入（D6）**：无条件剥掉模型参数里自带的 user_id，以会话身份覆盖。
   不这么做，用户说"查一下 u2002 的订单"、模型照填，就能越权读到别人的数据。
   这也是将来接登录态的**唯一**改动点。
2. **确认门（D1）**：写操作且未 confirmed → 不执行，返回 `needs_confirmation` +
   给用户看的提案文案。状态机拿它去问用户，用户点头后带 `confirmed=True` 重调。
3. **3 秒超时（D2）**：线程池执行，超时返回降级话术。
4. **结果三态（D3）**：ok / error（查无此单等业务错误）/ timeout，结构永远完整，
   **LLM 永远拿不到裸异常**。
5. **5 分钟只读缓存（D5）**：key = 工具名 + hash(参数 + user_id)。user_id 必须进哈希，
   否则"我的订单"这类空参数工具会让两个用户命中同一条缓存、把别人的订单发出去。
   写操作永不缓存（副作用重放是事故）。
6. **日志（D4）**：每次调用追加一行到 `data/logs/tool_calls.jsonl`。

---

## 九、Agent 层 app/agent：状态机是全项目的"总调度"

### intent.py —— 意图识别（一次 LLM 调用全带走）

`classify(text, history)` 输出 `(意图, 置信度, 槽位, 是否不满)`。**8 类意图**：

| 意图 | 含义 | 路由去向 |
|---|---|---|
| product_consult | 客观信息（功能/参数/材质） | RAG · product_knowledge |
| review_consult | 主观体验（口碑/优缺点） | RAG · review_knowledge |
| param_compare | ≥2 个商品对比 | RAG · product_knowledge |
| recommendation | 有需求没点名商品 | RAG · product_knowledge |
| order_query | 订单/物流/实时价格/库存 | 工具编排 |
| after_sale | 退换货/维修/投诉 | 工具编排 |
| human_transfer | 点名转人工 | 直达转人工 |
| chitchat | 闲聊/其他 | 直接生成 |

边界拿不准时看问的是"它**是什么**"（咨询）还是"它**好不好用**"（评价）——
提示词里用一对例子固定："多久充一次电"是咨询，"续航够用吗"是评价。
评价单列成第 8 个意图，是因为它查的是**另一个库**：并进 product_consult 就得在节点内
塞关键词二次判定，规则难维护，还会让置信度语义分裂。

槽位抽取两步走：**字面型号走正则**（确定性的东西不让模型掷骰子），模糊指代
（"它续航多久"）才让 LLM 结合最近 4 条用户消息做实体链接；**LLM 给的商品 ID 必须落在
商品目录里**，查不到就丢弃——一个幻觉出来的 ID 会把检索结果筛成空集，比不过滤更糟。

任何环节失败（网络断、JSON 坏、意图名不认识）一律返回 confidence=0，交由状态机的
澄清兜底接管，绝不向上抛异常。

### graph.py —— 状态机主体（手写分支，没用 LangGraph）

每轮 `handle(session_id, text)` 的完整旅程：

```
1. 重置本轮轨迹（重新绑定新 dict，不是 clear——前端还抓着上一轮的引用）
2. 有待确认的写操作？──是──→ 确认门：关键词判 同意/拒绝/听不懂
   │                          （在意图识别之前，不让 LLM 猜"算了"啥意思）
   否
3. 意图识别（LLM）→ 记录意图/置信度/槽位
4. 表达不满？→ 计数，连续 2 次 → 主动转人工（附摘要）   兜底④
5. 置信度 < 0.6 → 追问澄清                            兜底①
6. 按意图路由：
   ├─ RAG 四意图 → _answer_with_rag
   │    · 拼检索词（对比/推荐的槽位进查询，比裸话术准）
   │    · 话题锚点：本句抽到的 SKU，或会话记住的上一轮锚点
   │    · search_with_product_focus / search
   │    · 空结果 → "暂无相关信息"                     兜底②（这轮也落埋点）
   │    · 有结果 → grounding 提示词（只依据资料答，没有就说没有）
   │              + 差异化侧重（口碑要好评差评两边讲、对比要表格+倾向、
   │                推荐要给完反问一句缩小范围）
   │    · 命中里 ≥2 块同商品 → 更新会话锚点（下一句可省主语）
   │    · LLM 挂了 → "没能连上后台服务"（≠"资料里没有"，两回事） 兜底③
   │    · 引用判定 + 埋点落盘
   ├─ 订单/售后 → _tool_flow：LLM function calling 循环，最多 3 轮
   │    · 每次调用都过 executor（确认门/超时/缓存/日志）
   │    · 写操作返回 needs_confirmation → 挂起 pending_write，本轮结束等确认
   │    · 连败 2 次 → 转人工                            兜底③
   ├─ 点名转人工 → _transfer
   └─ 闲聊 → 直接生成（温度 0.7）
7. finally：记录本轮耗时
```

转人工 `_transfer` 是三种触发源的汇聚点：统一生成排队工单 + **对话摘要**
（"人机切换不转述等于让用户重问一遍"），且这一轮也写进会话历史。

### session.py —— 会话记忆（进程内存版，接口按 Redis 设计）

`Session` 持有：history（最近 **10 轮**，成对截断，不留孤儿消息）、dissatisfaction
计数、pending_write（跨轮的待确认写操作）、tool_fail_streak、**current_product_id**
（话题商品锚点，让"那它防水吗"能接上上文）、trace（本轮诊断轨迹）。
`SessionStore.get/drop` 的签名按 Redis `chat:ctx:{session_id}` 设计，将来换 Redis
只改内部实现，Agent 代码一行不动。

**trace 是"报告"不是"状态"**：三条纪律——不进 prompt（history 才是唯一喂给模型的）、
不进将来的 Redis 序列化、每轮**重新绑定**而非原地 clear（前端抓着上一轮的引用，
clear 会把屏幕上正在看的轨迹当场掏空）。

### trace_view.py —— 轨迹的契约与渲染

`TRACE_KEYS` 定义轨迹的合法键名（graph 也 import 它，写错键当场告警），
7 个分支和 6 条兜底都有中文名标签。渲染函数**容忍半成品轨迹**（确认门那轮没跑意图、
空召回那轮没有 hits），且刻意不 import streamlit——CLI 的 `/trace` 和前端侧栏看的是
同一份轨迹的两种画法。（文件名不叫 trace.py，因为 `streamlit run` 会把本目录塞进
sys.path 最前面，会遮蔽标准库的 trace 模块。）

### cli.py / webui.py —— 两个人口

- `cli.py`：终端 REPL，`/trace` 看本轮内部状态、`/history` 看记忆、`/new` 换会话。
  命令在 REPL 层拦截，不进状态机（混进对话历史会污染意图识别）。
- `webui.py`：Streamlit 前端。对话区渲染自己的 messages（不是被截断到 10 轮的
  Session.history）；**侧栏摊开本轮内部状态**：分支、意图+置信度、召回块表格
  （含"被引用"列）、工具调用表、命中的兜底、待确认写操作的常驻警告。
  附"预热模型"按钮（免得第一轮加载 BGE-M3 的十几秒像卡死）和"结束会话"按钮。
  `python -m app.agent.webui --e2e` 是**真正的**无头端到端：用 Streamlit 的 AppTest
  在同进程执行整个脚本（桩掉检索与 LLM，全程离线），断言一问一答进会话、侧栏画出
  轨迹、确认门三轮走通——而 `curl /_stcore/health` 返回 200 只说明服务器起来了，
  证明不了应用正确，两者不能互相替代。

---

## 十、LLM 层 app/llm：全项目唯一的出网出口

`client.py` 封装 DeepSeek（`deepseek-chat`，OpenAI 兼容协议）：

- **key 管理**：import 时读项目根 `.env`（`os.environ.setdefault`，系统环境变量优先），
  缺 key 的报错直接告诉你往哪写。
- **两个接口**：`chat()` 返回消息对象（工具编排要拿 tool_calls）、`chat_text()` 便捷版。
  `json_mode=True` 走结构化输出（DeepSeek 要求 prompt 里含 "json" 字样，约定写在两边）。
- **safe_call**：连接类抖动（超时/断连/限流，按异常类名判断）重试一次，仍失败**返回
  None** 并打印真实异常。返回 None 而不是替调用方编一句话，因为"意图识别当作没认出来"
  和 "RAG 资料里没有" 是两种不同的降级，得由调用方各自决定；而打印那一行是必要的——
  上层普遍吞异常，这里是唯一还看得见原因的地方。

---

## 十一、项目的前后关系：它是怎么一步步长成这样的

从 git 历史（25 个 commit，2026-09）可以清楚看到演进顺序，这也是理解"为什么代码长这样"的钥匙：

```
第 1 阶段  设计先行：docs/01~04 四篇设计文档 + 知识库数据 + 工具定义
          （先想清楚分域/分块/检索/状态机/数据规范，再动手）
第 2 阶段  实现检索层五步 + 工具层 + Agent 层，逐模块带自测
第 3 阶段  评测驱动的修正：50 条评测集跑出 96%，分析 2 条 miss
          → 加"话题商品锚点"到 100% → 实测推翻三处设计假设 → 回写 docs/01
第 4 阶段  缺陷修复轮：转人工漏写历史、LLM 无兜底、工具身份可被模型伪造
          （越权）、缓存不按用户隔离（串户）等 P2 级缺陷集中修掉
第 5 阶段  评价域闭环：review_knowledge 建了库却没有任何意图路由过去
          → 新增第 8 个意图 review_consult
第 6 阶段  可观测性：检索埋点 + 本轮轨迹（trace_view 契约）+ Streamlit 侧栏
第 7 阶段  工程收尾：README 与代码同步、requirements 重写为真实直接依赖
          （原 freeze 转储漏了 jieba，照装必崩）、.env.example、演示手册
第 8 阶段  "手写 vs 框架"对照实验：新增 LangGraph 版状态机 + LangChain 版检索器
          （不覆盖手写版，环境变量切换，评测集当裁判）
第 9 阶段  评测收尾：补 hit@1/MRR 解指标饱和 + 第五路 rerank 精排 + chunk 策略对比
          + RAGAS 生成侧评测与 50 条人工标注；期间修掉一批"看起来在跑、其实没跑"
          的脚手架缺陷（详见踩坑记录）
第 10 阶段 服务化（2026-09-18）：把同一个 Agent 包成 6 个端点的 HTTP 服务 + 14 组
          离线断言（**只做薄壳**：不加 SSE / 鉴权 / CORS / 多副本，理由与边界见 §13.1）
```

**第 8、9 阶段的顺序本身就是一条经验**：先把东西做对（第 1~7 阶段，手写版跑通并评测到
100%），**再**引入框架做对照（第 8 阶段），最后才补评测的完整性（第 9 阶段）。
反过来做的话，"换框架后分数变了"永远说不清是框架的锅还是自己代码的锅。

几个贯穿始终的工作习惯（外人读代码时会到处看到）：

1. **每个模块都有 `__main__` 自测**，带断言、可独立运行（README 的快速开始里每条命令
   都标注了期望输出）。没有引入 pytest，自测覆盖"机制是否符合预期"，不是回归网。
2. **设计文档是活文档**：实测推翻假设后就回写（docs/01 §七记了三处修正和理由），
   而不是让文档停留在"美好设想"。
3. **诚实的数字**：hit@5=100% 特意注明"50 条小集上机制有效的证据，不是没有提升空间"；
   引用率明写是代理指标、同一条 query 两次跑引用块数 1 和 3；两条兜底（低置信澄清、
   检索为空）明说"演示现场不好触发，别赌"。
4. **中文注释密度很高**，且注释讲"为什么"不讲"是什么"——大量 D1/D2 设计决策编号，
   供面试讲述和日后维护。

---

## 十二、怎么跑起来（5 分钟版）

```bash
# 1. 环境（Windows 下注意：系统 Python 缺 jieba，一律用 .venv 里的解释器）
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. 配 key：复制 .env.example 为 .env，填入 DeepSeek 的 DEEPSEEK_API_KEY
#    （连不上 api.deepseek.com 时，.env.example 里写了代理怎么配）

# 3. 离线自测（不联网、不需 key，验证装对了）
python -m app.retrieval.loader        # 期望：共加载 32 个文档
python -m app.retrieval.chunker       # 期望：124 个 chunk + 断言通过
python -m app.tools.executor          # 期望：9 项断言通过
python -m app.agent.lg_graph          # 期望：6 组离线断言通过（有 key 时多跑一遍联网冒烟）
python -m app.retrieval.lc_retriever  # 期望：双版对照五用例逐项一致
python -m app.agent.webui --e2e       # 期望：e2e 通过
python -m app.api.server              # 期望：14 组断言通过（服务层，不联网）

# 4. 建库（首次会从 HuggingFace 拉 BGE-M3 约 2.3GB，之后离线可用）
python -m app.retrieval.indexer

# 5. 跑 Agent（需 key）
python -m app.agent.cli            # 终端对话
streamlit run app/agent/webui.py   # 浏览器演示（推荐，侧栏能看内部状态）

# 6. 可选：换实现 / 加精排 / 生成侧评测
SHOPMATE_AGENT=lg python -m app.agent.cli         # 换成 LangGraph 版状态机
SHOPMATE_RERANK=1 python -m app.retrieval.eval    # 评测加第五路 rerank
RAGAS_N=15 python -m app.retrieval.ragas_eval     # 生成侧三指标（要裁判 key，按条计费）

# 7. 可选：HTTP 服务（需 key）—— 把同一个 Agent 暴露成 JSON 接口
python -m app.api.server --serve                  # 绑 127.0.0.1:8000，单 worker
curl 127.0.0.1:8000/health                        # 存活 + 单 worker 的证据
curl 127.0.0.1:8000/tools                         # 6 个工具 + 读写分级
# 交互式文档：http://127.0.0.1:8000/docs（可直接在页面上发请求）
```

> **连不上 API 时先看这条**：本项目**默认直连**，刻意不认系统代理（Windows 上那个由
> Clash 开关写进注册表的代理会让 TLS 握手被掐断，报一句和网络无关的错）。
> 真要经代理，在 `.env` 里**显式**写 `HTTPS_PROXY=http://127.0.0.1:7890`。详见 README 环境坑。

演示问什么、看哪里、哪两条兜底别在现场赌，见 `docs/05_demo_runbook.md`。

---

## 十三、当前状态、已知局限与"明确不做"

**已完成**（截至 2026-09-18）：五层全部打通，离线自测全绿；检索五路对照
（hit@5 最高 100%，hit@1/MRR 补齐排序质量）；生成侧 RAGAS 三指标跑通（15 条试跑）；
"手写 vs 框架"两版并存且行为逐项一致；CLI + 浏览器双入口，检索埋点与轨迹可视化落地；
**新增薄服务层**（`app/api/`：6 个端点 / 14 组离线断言，把 Agent 与轨迹暴露成可 curl 的
JSON——**只做这一层，不加 SSE / 鉴权 / CORS / 多副本**，边界见 §13.1）。

**如实声明的局限**（README"已知局限"一节，防止被问穿）：

- 知识库只有 32 文档 / 8 SKU，未验证 10 万+ 商品量级；BM25 索引在进程内存，重启重建。
- 会话记忆是进程内存 dict，多实例部署会串——接口已按 Redis 设计，替换即可。
- 引用率是代理指标不是准确率（见第七节 telemetry）；前端 SessionStore 无锁、
  刷新页面会遗弃会话（演示规模靠"结束会话"按钮手动回收）。
- `user_id` 硬编码 `"u1001"`，但身份注入点已收口在 `executor.execute` 一处，
  接登录态只改那里。
- 只有模块自测、没有 pytest/CI。
- **RAGAS 的 `answer_relevancy` 不可信**（0.542）：生成侧温度导致同一 query 复跑分数
  从 0.000 到 0.813 都出现过，要引用就引用多轮区间（见第七节 ragas_eval）。
- **RAGAS 全量 50 条尚未跑**：现有分数是评测集前 15 条的试跑（裁判调用按条计费），
  正式分数以全量轮为准。
- **两个 W6 项没做完，别在简历上写"精通"**：checkpoint 的"时间旅行"回滚只读过文档、
  没实操；supervisor / 子代理只到"知道是什么"、没写过。
- **`data/eval/` 整目录被 gitignore**，其中的 50 条人工标注不可重建（见第五节 ⚠️）。
- **服务层的三条硬局限（2026-09-18 新增，别在演示里被问穿）**：
  ① **只能单 worker** —— 会话在进程内存，`--workers 2` 会跑但同一会话被分到不同进程、
  各拿一个空会话（表现是"聊着聊着记忆没了"）。`server.py` 把 `workers=1` 写死，
  `/health` 用 `pid` + `uptime_s` + `sessions` 三个字段把违约做成**可检验**的。
  ② **同一个 sid 不能并发发两条** —— 线程池默认 40 并发 + `SessionStore` 无锁，
  两条同名请求会互踩（`history.append` / `dissatisfaction += 1` 是读-改-写）。
  这是 CLI / Streamlit 下几乎踩不到、**加了 HTTP 才真踩得到**的竞态。
  ③ **无鉴权、无 CORS、无 SSE** —— `user_id` 收口在 `executor.execute` 但仍是硬编码；
  首轮 RAG 实测 12.9 秒（事件循环没被占住，但仍要干等），这正是"加 SSE"的正当理由，
  也是刻意停在边界外的第一件事。

**明确不做（是判断，不是欠债）**：MySQL / Redis 接入、多用户支持、自动化测试/CI、
以及**把服务层做厚**（SSE / 鉴权 / CORS / 多副本）——8 SKU 的演示量级下这几项换上去
收益是零，规范已写在 docs/04，接了真实业务量再按图替换。
（"做了但有保留"和"主动不做"是两回事，别混着讲。）

> **⚠️ 上面这句在 2026-09-18 之前有一处不准确**：那四项里 FastAPI 的理由一个字都没写，
> 全部痕迹只有 `requirements.txt` 里的两行注释，README 与 docs/ 都查不到——
> 而"你这项目怎么不做成服务"是面试高频问题。**这条欠账已经补上了**：FastAPI 做了，
> 但只做薄的那一层，理由见下面 §13.1（那一节也从"为什么不做"改成了"边界停在哪"）。

### 13.1 为什么只做一层"薄"的 FastAPI，以及边界停在哪

**这一节在 2026-09-18 被改写过一次。** 原文标题是「为什么不做成 FastAPI 服务」，
结论反了，但**没有整节删掉重写**——因为那份论证大部分仍然成立，丢掉它会丢掉一份
"有论证的诚实"。所以逐条交代哪些还成立、哪些被推翻：

1. **"FastAPI 不是没用过，是排期在后面"——仍然成立。** `05-projects/chat-api` 就是
   W1-2 用它做的：`FastAPI()` + Pydantic 请求模型（带 `Field` 约束）+ `/chat` 的
   `StreamingResponse` 流式（SSE）+ `/chat-with-tools` + `/sessions` + `HTTPException`，
   还配了 Dockerfile。所以这次做 ShopMate 的服务层，**不是补能力缺口，是补一个场景**
   ——把它从"一个能聊天的程序"变成"一个能被 curl / 任何程序调用的服务"。
2. **"ShopMate 的复杂度不在 HTTP 层"——仍然成立，而且这正是"薄"的定义。**
   它要讲的状态机、混合检索、工具编排、确认门，一个字都不该搬到 HTTP 层来。
   所以这一层**刻意不加** SSE、不加鉴权、不加持久会话、不加 CORS：
   3 个文件（`__init__` / `schemas` / `server`）、6 个端点、
   **不 import 任何业务模块**（`app/api/` 里出现业务 if/else 就说明逻辑漏到了错误的层）。
   只做两件事：把请求翻译成 `agent.handle(sid, text)` 调用，把结果翻译成 JSON。
   而**新拿到的东西是实的**：`trace_view` 那四个纯函数原来只喂 Streamlit 侧栏，
   现在暴露成 `GET /sessions/{sid}/trace`，谁都能读到——这才是"服务化"换来的，
   转发本身不值钱。
3. **"CLI + Streamlit 已经够用"——部分推翻。** 给人用确实够用，但
   "**能不能被别的程序调用**"是另一回事：curl、前端、自动化评测都要它。
   Streamlit 不走 HTTP 而是在同进程 `import` Agent，所以它替不了这一层。
4. **"拆服务会和没接 Redis 直接冲突"——这条必须保留，但从"否决理由"改成了"硬约束"。**
   `SessionStore` 是进程内存 dict，所以：**只有单 worker**、`/health` 必须暴露 pid 当证据、
   `--workers 2` 是明令禁止的（它会跑，只是同一会话被分到不同进程、各拿一个空会话）。
   所以 `workers=1` 写死在 `server.py` 的入口里，让"默认正确"至少有一个落点。

**原来那句"只做一半比都不做更糟"现在自相矛盾，已改写**（原文把"只拆服务不接 Redis"
当否决理由，而现在我们就是只拆了服务）：

> 只做一半的代价被**明确标价**了：单 worker 是运行约束，服务在代码里强制不了，
> 所以我们把违约做成**可检验的**（见 `/health` 的 pid + uptime_s + sessions）。
> **停在边界上是判断，越过边界才是欠债。**

**"薄"没有护栏就不会薄。** HTTP 一旦存在，SSE / 鉴权 / CORS / 多用户 / Redis 会一个
接一个来，而且**每一个都有正当理由**（第一个尤其，因为首轮实测 13 秒确实难等）。
所以别只把"薄"写在文档里，给它一个**可检查**的形式——`server.py` 自测里那条
「openapi 的操作数 == 6」就是：让"变厚"变成一次要改断言的有意识决定，而不是某次顺手。

**简历措辞**：写"实现了 FastAPI 服务化"一定会被追问 SSE / 鉴权 / 多实例，
而你刻意都没做，答起来像在解释为什么没做。更稳的写法是把它当
「**我知道边界在哪、并且把边界做成了可检验的证据**」来写——
`/health` 那条 pid + uptime_s 的证据链就是证据本身。

---

## 十四、文档地图：想深入了解，去读哪篇

| 想了解 | 去读 |
|---|---|
| 知识库怎么分域、分块、混合检索为什么这么设计、评测怎么做的 | `docs/01_rag_knowledge_base.md`（最厚最核心） |
| 6 个工具的定义、读写分级与调用原则 | `docs/02_tool_definitions.md` |
| 状态机节点、意图路由表、置信度与四条兜底 | `docs/03_agent_workflow.md` |
| 元数据规范、Collection 设计、MySQL 表与 Redis key 规划 | `docs/04_data_schema.md` |
| 演示操作手册（问什么、看哪里、故障对照表） | `docs/05_demo_runbook.md` |
| 快速开始、设计要点表（面试向）、已知局限 | `README.md` |
| 各模块的设计决策（D1/D2/…） | 对应源码文件顶部的 docstring |
| 代码阅读顺序、每个文件的职责与调用关系 | 本文下一节（§十五） |

---

## 十五、代码阅读指南：顺序、职责与调用关系

> 25 个源码模块（2026-09 起含 LangGraph / LangChain 三个框架版文件，见 15.6；
> 评测收尾新增的 reranker / chunk_compare / ragas_eval 三个，见 15.7；
> 2026-09-18 新增 `app/api/` 薄服务层两个文件，见 15.4 阶段 8），
> 按依赖方向自底向上分九个阶段读。
> 每个文件先读它顶部的 docstring——那里列的 D1/D2/… 设计决策是作者留的"阅读地图"；
> 再读"模块速览"列出的公共函数；最后跑它的 `__main__` 自测——**自测代码本身就是
> 用法示例和验收标准**。

### 15.1 先建立三张地图：三条运行主线

同一批文件在这几条主线上出现，分清它们，"谁调用谁"就清楚了一半：

```
【服务流】每一轮对话都走这条线（值得精读）

 cli.py（终端）／ webui.py（浏览器）／ api/server.py（HTTP，经 Depends 拿 Agent）
      ← 三个入口，互不依赖；前两个直接持有 Agent，第三个靠注入
      │  agent.handle(session_id, text)
      ▼
 graph.py · Agent（状态机，全项目总调度）
      ├─① 确认门（有待确认写操作时）──────► executor.execute(confirmed=True)
      ├─② 意图识别 ──► intent.classify ──► llm.client（json_mode 出网）
      ├─③ RAG 支路 ──► retriever.search* ─► indexer（embed_texts + ChromaDB）
      │                    └──────────► telemetry（引用判定 + 埋点落盘）
      ├─④ 工具支路 ──► registry.get_schemas → llm.chat(tools=…) 循环
      │                    └──────────► executor.execute ──► mock.HANDLERS
      ├─⑤ 转人工 ────► executor.execute("transfer_to_human", confirmed=True)
      └─每一轮 ──► session.Session（读 history/锚点，写回复/计数/trace）
               ──► trace_view（graph 只用 TRACE_KEYS 校验键名；渲染归 cli/webui/api）

【入口流】三条路都汇到同一个 Agent 上，`runtime` 选择器决定背后是手写版还是 LG 版

 cli.py · REPL ─────────┐
 webui.py · Streamlit ──┼──► runtime.get_agent() ──► graph.Agent / LgAgent
 api/server.py · HTTP ──┘         （api 走 Depends 注入而不是直接调它，
                                    这样自测能用 dependency_overrides 换成假 Agent，
                                    真 Agent 从头到尾不被构造 → 不联网、不花钱）
 ※ 三条入口**各在自己的进程里跑**，会话存进程内存 → 浏览器里聊过的 sid，
    在 HTTP 进程查是 404（不是 bug）。这也是 HTTP 服务必须单 worker 的原因。

【建库流】上线前一次性执行，服务期不经过（读一遍即可）

 loader.load_all ──► chunker.chunk_all ──► indexer.build_index ──► data/chroma/
 （32 份 md/json）   （切 124 块拼锚点）     （BGE-M3 向量化入库）

 注意：服务流里 loader / chunker / build_index 都不参与——数据已在 ChromaDB 里，
 retriever 直接查库。第一次见到这三个文件被"谁都不调"别疑惑，它们是建库工具。
```

### 15.2 依赖分层（谁在谁的下面）

| 层 | 文件 | 项目内依赖 |
|---|---|---|
| 0 · 叶子（无内部依赖） | schema / telemetry / trace_view / session / registry / mock / llm·client | 只依赖数据文件或第三方库 |
| 1 | loader / chunker / indexer | schema |
| 1 | intent | llm·client |
| 2 | retriever | schema + indexer |
| 2 | reranker | indexer（复用 BGE 的模型加载设施）+ 本地 bge-reranker 快照 |
| 2 | executor | registry + mock |
| 3 · 质检（都不参与服务） | eval / chunk_compare | retriever（含内部函数）；chunk_compare 另用 indexer 建对照库 |
| 3 · 质检 | ragas_eval | retriever + llm·client + ragas / langchain |
| 4 · 总调度 | graph | intent / session / trace_view / retriever / telemetry / executor / registry / llm |
| 4 · 总调度（框架版） | lg_graph | graph 的常量与提示词 + 与手写版同源的各层 + langgraph |
| 3（框架版） | lc_retriever | indexer / retriever（融合段） / schema + langchain_chroma |
| 5 · 入口 | cli / webui | runtime →（graph 或 lg_graph）/ session / trace_view |
| 5 · 入口（框架层） | api·schemas / api·server | **不依赖任何业务模块的具体实现**，只在运行时经 `Depends` 拿 `runtime.get_agent()`；渲染复用 trace_view 的纯函数 |
| 5 · 选择器 | runtime | 按环境变量在 graph 与 lg_graph 之间二选一（懒加载） |

读法：从第 0 层往上，每个文件被引用的名字在上一层的 import 里都能对上。
（小知识：`app/` 目录本身**没有** `__init__.py`，四个子包有——所以 webui 要手动把
仓库根塞进 `sys.path` 才能以 `app.agent.…` 绝对导入。）

### 15.3 阅读顺序：九个阶段

| 阶段 | 文件 | 为什么这时候读 |
|---|---|---|
| 1 词汇表 | schema → session → trace_view | 三个"契约"文件几乎无逻辑，先掌握全项目的名词和数据形状 |
| 2 建库流 | loader → chunker → indexer | 数据怎么变成 124 个带锚点的向量块 |
| 3 查询流 | retriever → telemetry → reranker → eval → chunk_compare → ragas_eval | 检索怎么工作、怎么记录它跑过、怎么证明它好（一路证到生成侧） |
| 4 LLM 出口 | llm/client.py | 一个文件；之后读 intent/graph 时出网逻辑不再陌生 |
| 5 工具层 | registry → mock → executor | 定义 → 实现 → 安全编排，三层各一件事 |
| 6 状态机（山顶） | intent → graph → cli | 所有前面读过的模块在这里汇合；graph 是全项目最值得精读的文件 |
| 7 演示前端 | webui.py | 消费方视角回看 trace 契约；e2e 教你"无头测试 Streamlit"的正确姿势 |
| 8 薄服务层 | api/schemas → api/server | 见 15.4 阶段 8。**放在前端之后读**：看清"同一套 Agent 换个入口"的最小代价是多少 |
| 9 框架版对照 | lg_graph → runtime → lc_retriever | 见 15.6。**读完手写版再读这几个**，对照感受最深 |

### 15.4 逐文件讲解（按阅读顺序）

#### 阶段 1 · 词汇表

**① `app/retrieval/schema.py` —— 常量与数据结构**（建库+服务共用）
- **干什么**：全项目魔法数字集中营：6 类文档→3 个 collection 的映射、TOP_K=5、RRF_K=60、
  兜底阈值 0.012、向量距离门槛 0.45、向量维度 1024；外加 `RawDoc`（一个文件的原文+元数据）
  和 `Chunk`（一块可向量化文本）两个数据类。
- **为谁提供服务**：整个检索包；retriever 的阈值、indexer 的维度都从这里取。
- **被谁调用**：loader / chunker / indexer / retriever（各 import 不同的常量与数据类）。
- **关键看点**：`DOC_TYPE_MAP` 一张表就是"按知识域分三库"的全部落地；`SCORE_THRESHOLD`
  的注释记着"0.02 为什么被评测杀掉"。
- **自测**：无 `__main__`，靠下游模块的自测间接覆盖。

**② `app/agent/session.py` —— 会话记忆**（服务）
- **干什么**：`Session` 装一个会话的全部可变状态（最近 10 轮 history、不满计数、
  待确认写操作、工具连败计数、话题商品锚点、本轮 trace）；`SessionStore` 是 dict 版容器，
  `get/drop` 签名按 Redis `chat:ctx:{session_id}` 设计。
- **为谁提供服务**：graph（状态机的一切跨轮状态都寄存在这里）。
- **被谁调用**：graph（`Session, store`）、cli 与 webui（拿 `store` 查会话）。
- **关键看点**：D2"按轮成对截断"（不留孤儿 user 消息）、D4 话题锚点属于会话不属于全局、
  D5"trace 是报告不是状态"的三条纪律（不进 prompt、不进 Redis、重绑定不 clear）。
- **自测**：`python -m app.agent.session`

**③ `app/agent/trace_view.py` —— 轨迹契约与渲染**（服务·观测）
- **干什么**：定义本轮轨迹的合法键名 `TRACE_KEYS`（graph 写轨迹时逐键校验）+ 7 个分支
  和 6 条兜底的中文名标签；提供 `summarize / hit_rows / tool_rows / render_text` 四个
  容忍缺键的格式化函数。
- **为谁提供服务**：cli 的 `/trace` 命令、webui 的侧栏——**同一份轨迹的两种画法**；
  graph 只引用它的键名契约，不引用渲染。
- **被谁调用**：graph（`TRACE_KEYS`）、cli（`render_text`）、webui（`summarize` 等）。
- **关键看点**：D1"契约一处定义"（两边各写一遍键名，拼错的表现是前端一栏永远空白）；
  D3 为什么文件不叫 `trace.py`（会遮蔽标准库）；D4 半成品轨迹（确认门那轮没有 intent）
  也必须能渲染。
- **自测**：`python -m app.agent.trace_view`（不联网、秒级）

#### 阶段 2 · 建库流

**④ `app/retrieval/loader.py` —— 读文件**（建库）
- **干什么**：遍历 `data/rag_docs/` 六个子目录产出 `list[RawDoc]`；从文件名提取
  product_id（`SKU-10001_reviews.md` → `SKU-10001`）；第二遍扫描交叉回填
  category（只在 spec json 里有）和 brand（只在商品 md 里有）。
- **为谁提供服务**：建库链的头一站；`load_one()` 留给将来的增量更新。
- **被谁调用**：chunker 与 indexer 的自测（生产服务流**不经过**它）。
- **关键看点**：读文件失败带文件名报错、绝不静默跳过。
- **自测**：`python -m app.retrieval.loader`

**⑤ `app/retrieval/chunker.py` —— 分块**（建库）
- **干什么**：按 doc_type 六种切法（md 按 `##`、FAQ 按问答对、spec 整块展平、评价按
  好差评分块）+ 每块拼商品锚点 + 超长块在换行处对半，产出 124 个 Chunk。
- **为谁提供服务**：indexer 的输入。
- **被谁调用**：indexer 自测（`build_index` 的上游）。
- **关键看点**：D2 锚点为什么必须含商品 ID（评价文档标题不入块，BM25 靠锚点里的
  SKU 定位）；D4 `_split_long` 的终止性证明写在注释里。
- **自测**：`python -m app.retrieval.chunker`

**⑥ `app/retrieval/indexer.py` —— 向量化与入库**（建库 + 服务）
- **干什么**：双身份。建库时 `build_index` 把 chunk 批量向量化 upsert 进 ChromaDB
  （同 ID 覆盖、幂等）；服务时它的 `get_model / embed_texts / get_collection` 被
  retriever 每一轮复用——所以它同时属于两条主线。
- **为谁提供服务**：建库期的自己；服务期的 retriever 与 webui。
- **被谁调用**：retriever（`embed_texts, get_collection`）、webui 的"预热模型"按钮
  （`get_model`）。
- **关键看点**：D1 懒加载单例（2.3GB 模型只加载一次）；本地缓存判定发生在
  `import FlagEmbedding` **之前**（HF_HUB_OFFLINE 是 import 时读的常量）。
- **自测**：`python -m app.retrieval.indexer`（首次会下载模型）

#### 阶段 3 · 查询流

**⑦ `app/retrieval/retriever.py` —— 混合检索（检索层核心）**（服务）
- **干什么**：`search()`：向量路（BGE-M3+cosine 门槛）与 BM25 路（jieba+停用词）各出
  排名 → RRF 融合 → 词法置信门 → doc_type 加权 → 阈值兜底；`search_with_product_focus()`：
  话题商品"重排+补充"，永不丢无过滤结果。
- **为谁提供服务**：graph 的 RAG 支路（生产入口）；eval 的五路对照（复用它的内部函数，
  保证对照的就是线上同一套代码）。
- **被谁调用**：graph `_answer_with_rag`、eval。
- **关键看点**：D1 语料从 ChromaDB 捞（两路永远同一份数据）；D3 为什么选 RRF 不选加权
  求和；D6 带 where 时 BM25 必须在子集内重排号；D8 为什么不独占过滤。自测里 6 个
  样例 query（含一条必须返回空的"量子涨落"）值得逐个跑。
- **自测**：`python -m app.retrieval.retriever "通勤降噪耳机推荐"`

**⑧ `app/retrieval/eval.py` —— 50 条评测**（质检，不参与服务）
- **干什么**：内嵌 50 条人工标注用例（query + 期望文档集合 + 话题商品标注），跑
  纯向量/纯 BM25/RRF/锚点/rerank 五路 × hit@5·hit@1·MRR 三个指标，打印各路未命中
  清单与"零倒退"差值。
- **为谁提供服务**：你——调任何检索参数前后各跑一遍，它就是回归网。
- **被谁调用**：无人（纯 `__main__`）。
- **关键看点**：D1 标"可接受文档集合"而非单一答案；D2 判到文档级而非块级。
- **自测**：`python -m app.retrieval.eval`（约 1 分钟）

**⑨ `app/retrieval/telemetry.py` —— 检索埋点**（服务·观测）
- **干什么**：把每轮 RAG 记成一行 JSONL：query、5 个召回块预览、RRF 分、outcome
  （answered/no_info/llm_fail）、引用判定（字符 4-gram 重叠率，扣掉本轮样板文字）。
- **为谁提供服务**：graph（在 `_answer_with_rag` 的**唯一**记账点调用）；以及日后读
  日志做分析的人——每块原始重叠率都留着，改阈值可离线重算。
- **被谁调用**：graph（`cite_flags / hit_rows / build_record / log_retrieval`）。
- **关键看点**：D1"写在检索包、调用方在 Agent 层"的两个理由（回复只有 Agent 层有；
  焦点检索内部会再调一次 search，记错地方命中率凭空翻倍）；D2 代理指标的诚实声明。
  本阶段读它只需记住结论，读完 graph 再回头看 D1 会更透。
- **自测**：`python -m app.retrieval.telemetry`

#### 阶段 4 · LLM 出口

**⑩ `app/llm/client.py` —— DeepSeek 封装（全项目唯一 LLM 出口）**（服务）
- **干什么**：import 时读 `.env`；`chat / chat_text` 两个调用接口 + `json_mode` 开关；
  `safe_call` 对连接类异常重试一次、仍失败返回 None 并打印真实异常。
- **为谁提供服务**：intent（意图识别的 json_mode 调用）、graph（RAG 生成、闲聊、
  工具编排三处）——全项目出网只此一家，重试策略收口在这层不重复造。
- **被谁调用**：`intent.classify`、`graph` 的 `_answer_with_rag / _tool_flow / _free_chat`。
- **关键看点**：D4 为什么返回 None 而不是替调用方编话术（"资料里没有"和"没连上"是
  两种降级）；为什么失败必须 print（上层普遍吞异常，这里是唯一看得见原因的地方）。
- **自测**：`python -m app.llm.client`（缺 key 路径不联网可验）

#### 阶段 5 · 工具层

**⑪ `app/tools/registry.py` —— 工具定义加载**（服务）
- **干什么**：读 `data/tools/*.json` 六份定义并校验（重名/缺 parameters/required 越界
  当场抛错）；产出给 LLM 的 tools schema；`is_write_op` 读写分级判定。
- **为谁提供服务**：executor 的确认门（`is_write_op`）；graph 组装 function-calling
  请求（`get_schemas`）。
- **被谁调用**：executor、graph、mock 自测（校验实现与定义对齐）。
- **关键看点**：D2 写操作清单硬编码在代码不在 JSON（安全约束由代码裁决）；
  `is_write_op(未知名) 返回 True` 的保守方向。
- **自测**：`python -m app.tools.registry`

**⑫ `app/tools/mock.py` —— 六个工具的模拟实现**（服务）
- **干什么**：`HANDLERS` 字典按名分发：价格从 specs 快照读（与 RAG 同源，示范
  "实时数据不进知识库"的边界）、库存/订单/物流是内存表、售后申请真插内存表递增
  工单号；查不到抛 `ToolInputError`。
- **为谁提供服务**：executor（它只认 `HANDLERS`，别人不许绕过它直接调工具）。
- **被谁调用**：executor.execute。
- **关键看点**：u1001 的两笔订单（一笔在途可查物流、一笔已完成可走售后）是演示剧本
  的数据根据；断货 SKU 是刻意造的兜底用例。
- **自测**：`python -m app.tools.mock`

**⑬ `app/tools/executor.py` —— 统一执行入口（工具层安全核心）**（服务）
- **干什么**：`execute()` 唯一入口：剥掉模型自填的 user_id 注入会话身份 → 写操作确认门
  （未确认返回 `needs_confirmation`+提案文案）→ 只读 5min 缓存（key 含 user_id）→
  线程池 3s 超时 → 结果收敛为 `ToolResult` 三态（ok/error/timeout，永不裸抛）→
  落一行工具日志。
- **为谁提供服务**：graph 的三个调用点——工具支路、确认后的执行、转人工。
- **被谁调用**：graph `_tool_flow / _resolve_confirmation / _transfer`。
- **关键看点**：D5 缓存 key 为什么必须含 user_id（"我的订单"空参数工具会串户）；
  D6 身份以会话为准（也是将来接登录态的唯一改动点）。
- **自测**：`python -m app.tools.executor`（9 项断言，含越权防护）

#### 阶段 6 · 状态机（山顶）

**⑭ `app/agent/intent.py` —— 意图识别**（服务）
- **干什么**：`classify()` 一次 LLM 调用带回 8 类意图、置信度、槽位、是否不满；
  product_id 两步抽取（字面型号正则优先 → 模糊指代 LLM 实体链接，结果必须落在商品
  目录里否则丢弃）；解析/出网失败一律降级为 confidence=0 不上抛。
- **为谁提供服务**：graph 的路由决策全靠它。
- **被谁调用**：graph `_handle_turn`。
- **关键看点**：D5"确定性的东西不让模型掷骰子"；D6 为什么评价单列第 8 意图（换库
  不是换话术）；D7"调模型"与"解析"分开留痕（网络断≠JSON 坏）。
- **自测**：`python -m app.agent.intent`（前两组断言不需 key）

**⑮ `app/agent/graph.py` —— 状态机主体（全项目最值得精读）**（服务）
- **干什么**：`Agent.handle()` 串起一切：确认门 → 意图识别 → 不满计数/低置信澄清
  两条兜底 → 按 8 意图路由（RAG 支路 / 工具编排 / 转人工 / 闲聊）→ 生成回复；
  外层 try/finally 管轨迹生命周期（重绑定+耗时），`_transfer` 收敛三种转人工触发源。
- **为谁提供服务**：cli 和 webui（也就是最终用户）；它是唯一同时 import 全部其他
  模块的文件。
- **被谁调用**：cli、webui（模块级单例 `agent`）。
- **关键看点**：D1 确认门在意图识别之前（"算了不换了"不让 LLM 猜）；D5 RAG 生成
  严格 grounding；D6 工具循环上限 3 轮；D8 锚点只对 `FOCUSABLE` 两意图生效；
  D10 轨迹为什么用"重新绑定"而不是 clear。自测用桩函数离线跑遍每条支路，是理解
  分支行为的最好材料。
- **自测**：`python -m app.agent.graph`（有 key 才跑联网冒烟）

**⑯ `app/agent/cli.py` —— 终端入口**（入口）
- **干什么**：REPL 循环；`/trace /history /new /exit` 四个命令在 REPL 层拦截，
  不进状态机（命令混进对话历史会污染意图识别）。
- **为谁提供服务**：终端用户；也是调试 Agent 最快的通道。
- **被谁调用**：无人调用（入口）；它调 graph（`agent.handle`）、session（`store`）、
  trace_view（`render_text`）。
- **关键看点**：会话 id 跟进程走——将来换 API 服务时这里改成从请求头取。
- **自测**：无（交互式；跑 `python -m app.agent.cli` 直接聊）。

#### 阶段 7 · 演示前端

**⑰ `app/agent/webui.py` —— Streamlit 前端**（入口）
- **干什么**：主区聊天；侧栏摊开本轮内部状态（分支/意图/召回块表格/工具表/兜底/
  待确认写操作警告）+ "预热模型"与"结束会话"按钮；`--e2e` 用 AppTest 无头端到端。
- **为谁提供服务**：浏览器用户（演示主战场）；e2e 供演示前 30 秒自检。
- **被谁调用**：无人调用（入口）；它调 graph/session/trace_view/indexer（预热）。
- **关键看点**：D1 `streamlit run` 下相对导入会 ImportError，所以手动引导 sys.path；
  D3 不套 `@st.cache_resource`（会拿到全新 Agent+空会话且不报错）；D4 渲染自己的
  messages 而不是被截断到 10 轮的 history；e2e 一节示范了"curl 200 证明不了应用
  正确"时该怎么测。
- **自测**：`python -m app.agent.webui --e2e`（离线）

#### 阶段 8 · 薄服务层（2026-09-18 新增）

> 这一层只有 2 个文件（另加一个空的 `__init__.py`），是刻意压到这个体量的。
> 读它的重点**不是学会写 FastAPI**，而是看"同一套 Agent 换一种调用方"能有多便宜，
> 以及**边界是怎么被写死的**。
>
> （小提醒：这里的编号是 ㉔㉕，因为全文的圈码**按"加入项目的时间"排，不是按阅读顺序**
> ——它比 15.6 / 15.7 里那批 `⑱`-`㉓` 都晚。阅读顺序一律以 15.3 的表为准。）

**㉔ `app/api/schemas.py` —— 请求/响应模型**（入口）
- **干什么**：9 个 Pydantic 模型：ChatRequest/ChatResponse（一轮对话）、TracePayload
  （轨迹）、HealthResponse（存活 + 单 worker 证据）、SessionView/TraceView（读会话与
  轨迹）、ToolInfo/ToolsResponse、DeleteResponse。
- **为谁提供服务**：`server.py` 的出参/入参形状；以及 `/docs` 那份**自动生成的接口说明书**。
- **被谁调用**：server.py（每个端点的 `response_model`）。
- **关键看点**：D1 显式字段不用裸 dict（否则 `/docs` 是空的——而那份文档就是"这服务
  能干什么"的全部说明）；D2 description 一律中文且**写明没做到的部分**（`user_id`
  是预留字段这件事就写在字段说明里，藏在文档角落等于没写）；D3 轨迹字段用宽松
  `dict / list[dict]`，**不逐字段建模**——轨迹是"走到哪记到哪"，确认门那轮根本没有
  intent、空召回那轮 hits 是空表，用严格模型会在半成品轨迹上直接 500，把"这轮没有
  这个数据"误报成"服务坏了"。
- **自测**：无 `__main__`，由 `server.py` 的自测间接覆盖。

**㉕ `app/api/server.py` —— 6 个端点 + 离线自测**（入口）
- **干什么**：FastAPI 实例 + `Depends` 注入的 Agent；6 个端点（`GET /health`、
  `POST /chat`、`GET /sessions/{sid}`、`GET /sessions/{sid}/trace`、
  `DELETE /sessions/{sid}`、`GET /tools`）；`--serve` 起 uvicorn（**单 worker**），
  不带参数跑 14 组离线断言。
- **为谁提供服务**：curl、任何程序、将来的前端。**只做两件事**：把 HTTP 请求翻译成
  `agent.handle(sid, text)`，把结果翻译成 JSON。
- **被谁调用**：uvicorn（`app.api.server:app`）；它自己只经 `runtime.get_agent()`
  拿 Agent，**不直接 import graph / retriever / executor**。
- **关键看点**（五个决策，每一个都是"看起来更现代、实际更糟"的对立面）：
  D1 **端点一律 `def`、不许 `async def`**——`handle` 首轮 RAG 实测 12.9 秒同步阻塞，
  写成 async 会占住事件循环，那 12.9 秒里连 `/health` 都不响应，**看起来像挂了其实
  只是正常工作**；写成 `def` 由 FastAPI 丢进线程池，事件循环空着（代价也一起说：
  线程池 40 并发 + SessionStore 无锁 = 同 sid 并发会互踩，所以配了"单 worker"
  和"同 sid 不并发"两条约束）。
  D2 **查询端点只用 `store.peek()`，不许用 `store.get()`**——`get()` 走 setdefault，
  是**读即写**，在 GET 上返回 200 + 空会话，既撒谎又往无 TTL 的内存里漏一个空对象。
  D3 **Agent 用 `Depends` 注入**，不在端点里直接 `get_agent()`、更不在模块级构造——
  自测才能用 `dependency_overrides` 换成假 Agent（真 Agent 从头到尾不被构造 →
  不联网、不加载 BGE-M3、不花钱）。
  D4 轨迹字段用宽松 dict，不在 API 侧给 `TRACE_KEYS` 开第二个副本。
  D5 **"薄"要有可检查的护栏**：路径数 == 6 写进自测——让"变厚"变成一次要改断言的
  有意识决定，而不是某次顺手。
- **自测**：`python -m app.api.server`（14 组断言，**不联网、零成本**；
  第 13 组就是那条"路径数 == 6"的护栏）

### 15.5 读完后的一条验收线

全部读完后，这条命令链应当每一环你都**知道它在验证什么**——说不出来说明那一层
该回去重读：

```bash
python -m app.agent.session        # ① 词汇：会话状态与截断
python -m app.retrieval.chunker    # ② 建库：32 文档 → 124 块
python -m app.retrieval.eval       # ③ 查询：五路 × hit@5/hit@1/MRR 对照
python -m app.tools.executor       # ④ 工具：确认门/缓存隔离/越权防护
python -m app.agent.graph          # ⑤ 状态机：每条支路桩测
python -m app.agent.webui --e2e    # ⑥ 前端：无头端到端
python -m app.api.server           # ⑦ 服务层：6 端点 × 14 组断言（读即写陷阱/错误码/薄层护栏）
```

另外三条**不在这条链上**（要模型、要联网或要花钱，按需跑）：
`python -m app.retrieval.reranker`（精排，需本地模型）、
`python -m app.retrieval.chunk_compare`（三套库重建，慢）、
`RAGAS_N=15 python -m app.retrieval.ragas_eval`（生成侧评测，**要裁判 API key**）。

配套设计文档对照着读：阶段 2/3 配 `docs/01`，阶段 5 配 `docs/02`，阶段 6 配
`docs/03`，数据规范配 `docs/04`。

### 15.6 框架版文件（2026-09 新增：LangGraph / LangChain 两版并存）

三个新文件把"手写 vs 框架"变成可运行的对照实验——**不覆盖手写版，环境变量切换，
评测集当裁判**。建议在读完阶段 1-8 之后当"第九阶段"读，对照感受最深。

**⑱ `app/agent/lg_graph.py` —— LangGraph 编排版状态机**（服务，可选实现）
- **干什么**：用 StateGraph 重写 graph.py：`interrupt()` 承载写操作确认门
  （取代手写 pending_write + 关键词前置门），`conditional_edges` 承载 8 意图
  路由与四条兜底，tool_llm ⇄ tool_exec 环取代 for 循环，**SqliteSaver checkpoint
  持久化暂停点**——进程重启后确认门仍可续，这是手写版给不了的能力。
- **为谁提供服务**：与手写版 `Agent` 同签名的 `LgAgent.handle()`，经 runtime
  选择器供 cli/webui 使用（`SHOPMATE_AGENT=lg` 开启）。
- **被谁调用**：cli / webui（切换开关打开时）；提示词、轨迹契约、executor、
  retriever、client 全部与手写版共用，只有编排层换框架。
- **关键看点**：D2 interrupt 必须放在节点任何副作用之前（resume 会重执行该节点）；
  自测第 4 节的断点续跑演示（落盘→换图实例→resume 成功）值得逐行读。
- **自测**：`python -m app.agent.lg_graph`（6 组断言全程离线；只有第 7 节联网冒烟需要 key，
  没 key 会跳过并打印提示）

**⑲ `app/agent/runtime.py` —— 实现选择器**（入口的装配点）
- **干什么**：按环境变量 `SHOPMATE_AGENT` 在手写版与 LangGraph 版之间二选一，
  默认手写版；懒加载——不开开关就不 import lg_graph（不背 SqliteSaver）。
- **为谁提供服务**：cli 与 webui（两者只改了这一处 import，前端零改动）。
- **被谁调用**：cli.py、webui.py 的 `get_agent()`。
- **关键看点**：为什么两条实现路线能并存的答案就在这——`handle(sid, text) → str`
  签名一致是契约，Session 镜像（lg_graph D4）是前端无感的代价。
- **自测**：无独立自测（两个分支分别由 graph / lg_graph 自测覆盖）。

**⑳ `app/retrieval/lc_retriever.py` —— LangChain 版检索器**（服务，可选实现）
- **干什么**：只换向量路的实现——`BGEEmbeddings`（langchain Embeddings 接口的
  十几行包装，内部仍委托 indexer.embed_texts）+ `langchain_chroma.Chroma` 向量库
  对象；BM25 路、RRF、词法置信门、doc_type 加权、阈值兜底**全部复用手写版抽出的
  `retriever._fuse_and_format`**（两版共用的唯一融合事实源）。
- **为谁提供服务**：eval 的 `SHOPMATE_RETRIEVER=lc` 开关；将来 graph 若要切
  LangChain 版检索，`search()` 签名与手写版逐字段一致。
- **被谁调用**：eval（切换开关）；当前 graph 仍默认手写版检索器。
- **关键看点**：D2 用"同 query 双版距离逐项对齐"的断言把 langchain_chroma 的
  score 语义（=Chroma 原始 cosine 距离）钉死；自测第 1 节双版五用例逐项一致
  （含空召回兜底）就是"换实现不换行为"的证明。
- **自测**：`python -m app.retrieval.lc_retriever`（双版对照，不联网）；回归看
  `SHOPMATE_RETRIEVER=lc python -m app.retrieval.eval`（应与手写版同为
  90/94/96/100）。

### 15.7 评测收尾新增的三个文件（2026-09-17）

阶段 3 的清单在 W7 又长了三个——**检索指标之外，怎么证明"没编"，怎么选切法**。
前两个不参与服务（纯质检），第三个要花钱。

**㉑ `app/retrieval/reranker.py` —— cross-encoder 精排**（服务，可选路）

- **干什么**：`rerank(query, hits)` 把"问题 + 候选段落"拼成一条送进 bge-reranker-v2-m3，
  直接输出相关性分并重排。**只对初检回来的候选跑**——它准得多但慢得多，不能对全库跑。
- **为谁提供服务**：`retriever.search_reranked`（`SHOPMATE_RERANK=1` 开启）；`eval.py` 的第五路。
- **被谁调用**：retriever、eval。
- **关键看点**：D1 **cross-encoder（模型精排）≠ 话题锚点（业务规则）**——前者把问题和段落
  拼一起过模型，后者只是"把话题商品的块提前"，表格里两者不可互相替代；D2 懒加载单例
  （模型约 1.1GB，开关没开就不该背上这份内存）；D3 批量打分一次吃全部候选对。
- **自测**：`python -m app.retrieval.reranker`（需要本地模型快照，不联网）

**㉒ `app/retrieval/chunk_compare.py` —— chunk 策略对比**（质检）

- **干什么**：typed(750) / fixed512 / fixed256 三套库 × 三路检索 × 三指标，产出对比表。
- **为谁提供服务**：切法的**选型决策**——结论是维持 typed，理由见第七节。
- **被谁调用**：无人（纯 `__main__`）。
- **关键看点**：D1 **每种策略用独立持久化目录**（块边界一变 chunk_id 语义就变，混在一个库里
  upsert 会得到两种策略的杂交结果）；D2 **换库必须同时清 BM25 缓存**——缓存键是 collection
  名、不按库区分，不清的话会"向量路查新库、词法路还在评旧库"，两条路在评两份不同数据。
- **自测**：`python -m app.retrieval.chunk_compare`（要重建三套库，比较慢）

**㉓ `app/retrieval/ragas_eval.py` —— 生成侧评测**（质检，**要联网 + 花钱**）

- **干什么**：`build_samples()` 跑真实检索 + 真实 RAG 生成攒样本 → `load_ground_truth()`
  读人工标注 → `main()` 组装 RAGAS 三指标打分打表；`--dump-template` 生成标注模板。
- **为谁提供服务**：生成侧质量（faithfulness 等）的**唯一**来源。
- **被谁调用**：无人（纯 `__main__`）；但它复用的 `RAG_PROMPT` 来自 `graph.py`——
  评的就是线上那条生成路径，不是另写一个评不对位的简化版。
- **关键看点**：D1 **异族裁判是分数可用的前提**（qwen-plus，`DASHSCOPE_API_KEY`；退回
  DeepSeek 会打警告横幅，那模式下的分不能进文档）；D2 embedding 复用本地 BGE-M3
  （不走 OpenAI，省钱且与线上同口径）；D3 标准答案外置 JSON 且**必须人工写**；
  D4 `RAGAS_N` 控制条数。**外加一个坑**：`answer_relevancy` 因为"反解问题 + 生成侧温度"
  而不可信，别当结论引用（详见第七节）。
- **自测**：没有离线自测（它天生要联网）。跑法：`RAGAS_N=15 python -m app.retrieval.ragas_eval`

---

*本文由通读全部源码与文档后整理，**2026-09-17 更新**（补第四节技术栈总表、第五/七/十一/
十三节的实况，以及 15.7 三个新文件）。若代码与本文冲突，以代码为准；
若发现冲突，说明 README/文档需要同步——按本项目惯例，那是该修的 bug。*
