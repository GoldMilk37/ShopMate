# ShopMate · AI 电商智能客服与导购系统

> 面向电商平台的智能客服 Agent：RAG 商品知识库检索 + Function Calling 业务系统调用，
> 覆盖商品咨询、参数对比、个性化推荐、订单查询、售后处理全场景。

**当前状态**（2026-09-13）：RAG 层、工具层、Agent 层三层已打通，离线自测全绿，
检索评测 50 条集 hit@5 达到 **100%**。可直接 `python -m app.agent.cli` 跑人机对话。

| 层 | 状态 | 自测入口 |
|---|---|---|
| RAG 检索（混合 + 话题商品锚点） | 已完成 | `python -m app.retrieval.eval` |
| 工具层（6 工具，读写分级 + 确认门） | 已完成 | `python -m app.tools.executor` |
| Agent 层（8 类意图 + 状态机 + 会话记忆） | 已完成 | `python -m app.agent.intent` / `.graph` |
| MySQL / Redis 接入 | 未开始（现为 SQLite + 进程内存） | — |

## 架构

```
用户输入
   │
   ▼
┌──────────────────┐   ① 确认门（纯关键词判定，不进 LLM）
│  待确认写操作？    │──是─▶ 是/否 → 执行或取消，本轮结束
└────────┬─────────┘
         │ 否
         ▼
┌──────────────────┐   ② 意图识别（LLM，7 类 + 置信度 + 槽位）
│  intent.classify │   置信度 < 0.6 → 追问澄清
└────────┬─────────┘   点名转人工 / 连败×2 / 不满×2 → 转人工（附摘要）
         │
   ┌─────┼──────────┬─────────────┐
   ▼     ▼          ▼             ▼
 商品咨询  参数对比    个性化推荐      订单/售后
   │     │          │             │
   ▼     ▼          ▼             ▼
 RAG 检索                      Function Calling
 向量 + BM25 → RRF 融合         （6 工具，写操作需二次确认）
 + 话题商品锚点重排              3s 超时降级 / 5min 只读缓存
 （ChromaDB · BGE-M3）
   └─────┴──────────┴─────────────┘
         │
         ▼
   回复生成 ── 检索为空 / LLM 异常 ──▶ 兜底话术或转人工
```

三个知识域各自独立建库（`product_knowledge` / `policy_knowledge` / `review_knowledge`），
按**知识域**而非文档类型切分——评价是口语化文本，与商品详情混在一库会污染商品咨询的相似度。

## 目录结构

```
ShopMate/
├── app/
│   ├── retrieval/          # 检索管线：加载 → 分块 → 入库 → 混合召回 → 评测
│   │   ├── schema.py       #   常量与数据结构（6 类文档 → 3 collection 映射）
│   │   ├── loader.py       #   加载 rag_docs 产出 RawDoc（结构化 + 非结构化）
│   │   ├── chunker.py      #   按文档类型分块（六种切法）→ 124 chunk
│   │   ├── indexer.py      #   BGE-M3 向量化 + ChromaDB 持久化入库
│   │   ├── retriever.py    #   向量+BM25 混合检索、RRF 融合、元数据过滤
│   │   └── eval.py         #   50 条评测集，四路 hit@5 对照
│   ├── tools/              # Function Calling 工具层
│   │   ├── registry.py     #   加载 JSON 工具定义，读写分级（WRITE_OPS）
│   │   ├── mock.py         #   模拟业务实现（SQLite + JSON 数据）
│   │   └── executor.py     #   执行编排：确认门 / 超时 / 缓存 / 异常收敛
│   ├── agent/              # Agent 状态机
│   │   ├── intent.py       #   意图识别：7 类 + 置信度 + product_id 槽位
│   │   ├── graph.py        #   状态机主体：确认门 / 四条兜底 / 工具编排
│   │   ├── session.py      #   会话记忆（进程内存版，接口按 Redis 设计）
│   │   └── cli.py          #   命令行交互入口
│   └── llm/
│       └── client.py       #   DeepSeek 封装（OpenAI 兼容）+ json_mode
├── data/
│   ├── rag_docs/           # RAG 知识库：32 文档 / 3 品类 / 8 SKU
│   │   ├── products/       #   商品详情 ×8
│   │   ├── specs/          #   规格参数 JSON ×8
│   │   ├── reviews/        #   用户评价摘要 ×8
│   │   ├── faq/            #   品类 FAQ ×3
│   │   ├── policies/       #   平台政策 ×2
│   │   └── guides/         #   选购指南 ×3
│   ├── tools/              # 工具定义 JSON ×6
│   ├── chroma/             # ChromaDB 持久化目录（3 collection）
│   └── logs/               # 工具调用日志（jsonl）
├── docs/                   # 设计文档（决策依据都在这里）
│   ├── 01_rag_knowledge_base.md   # 知识库：分域/分块/混合检索/评测结论
│   ├── 02_tool_definitions.md     # 工具总览与调用原则（只读/写操作分级）
│   ├── 03_agent_workflow.md       # 状态机：意图路由/置信度兜底/上下文管理
│   └── 04_data_schema.md          # 元数据/Collection/MySQL/Redis 规范
├── requirements.txt
└── README.md
```

## 快速开始

```bash
# 1. 环境（Python 3.10+）
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

# 2. 填 LLM key（DeepSeek，OpenAI 兼容协议）
echo DEEPSEEK_API_KEY=sk-xxxxx > .env

# 3. 离线自测（不联网、不依赖 key）
python -m app.retrieval.loader    # 应打印：共加载 32 个文档 + SKU-10001 当前价 599
python -m app.retrieval.chunker   # 应打印：124 个 chunk + 三项断言全通过
python -m app.tools.registry      # 应打印：6 个工具 + 读写分级
python -m app.tools.executor      # 应打印：7 项断言全通过
python -m app.agent.session       # 应打印：10 轮成对截断

# 4. 建库（首次会从 HuggingFace 拉 BGE-M3，之后离线可用）
python -m app.retrieval.indexer
python -m app.retrieval.retriever "通勤降噪耳机推荐"

# 5. 检索评测（四路对照，约 1 分钟）
python -m app.retrieval.eval

# 6. 跑 Agent（需 key）
python -m app.agent.intent        # 意图识别 7 条用例
python -m app.agent.graph         # 状态机冒烟（含确认门、工具编排、转人工）
python -m app.agent.cli           # 人机对话；/new 换会话 /history 看记忆 /exit 退出
```

> **环境坑**：系统 Python 缺 `jieba` 等依赖，一律用 `.venv/Scripts/python.exe` 跑。
> BGE-M3 已缓存在本地，indexer 有缓存时不再联网。

## 设计要点（面试可讲）

| 决策 | 理由 |
|---|---|
| 6 类文档 → 3 个 Collection | 按知识域而非文档类型分库：评价单独分库，避免口语化文本污染商品咨询的相似度 |
| 混合检索（向量 + BM25）+ RRF 融合 | 向量管"意思像"（"防水靠谱吗"），BM25 管"字面准"（"SKU-10001"）；RRF 只看排名不看分数，免去两路分数的归一化标定 |
| **话题商品锚点（Agent 层槽位过滤）** | 用户报型号或用"它"指代时，锚定该商品：检索结果里属于它的块提到前排；没占住前排时补一次带过滤的检索。补回了纯相似度排不上但实际在问的问题 |
| **锚点不做独占式过滤** | faq/guide/policy 的 `product_id` 是空串，独占 `where` 会把它们整体排除——而"耳机保修多久"的正解恰恰是 faq。所以只能重排 + 补充，不能筛除 |
| LLM 给的 SKU **必须落在商品目录内** | 幻觉出的 ID 会把正确答案筛成空集。抽槽位时字面型号走正则、模糊指代走 LLM + 注入目录，校验不过就丢弃 |
| 确认门放在意图识别之前 | 用户回"算了不换了"时不再让 LLM 猜意图，纯关键词零成本判定。顺序反了会多花一次调用还可能误判 |
| 写操作走「提案 → 用户确认 → 执行」 | `apply_after_sale` 由用户二次确认，防止 LLM 幻觉参数造成真实业务损失 |
| 工具异常一律收敛成 ToolResult | LLM 幻觉参数导致的 `TypeError`、慢查询超时、查无此单，三种情况都变结构化结果，LLM 侧永远拿不到裸异常 |
| `is_write_op(未知名) 返回 True` | 宁可多问一次确认，不可漏一次确认——保守方向要选对 |
| 实时数据不走 RAG | 价格/库存是实时状态，只经 Function Calling；RAG 里的 spec 只是带时间戳的快照 |
| 转人工必附对话摘要 | 人机切换不转述等于让用户重问一遍；摘要让坐席直接接手 |

## 检索评测（50 条集，docs/01 §七）

评测集是 50 条真实客服话术改写，人工标注应命中的 chunk，`hit@5` 四路对照：

| 检索方式 | hit@5 | 未命中 |
|---|---|---|
| 纯向量 | 90% | 5 |
| 纯 BM25 | 94% | 3 |
| RRF 混合 | 96% | 2 |
| **混合 + 话题商品锚点** | **100%** | 0 |

关于提升幅度：**不吹"+30%"**。小语料（50 条级、32 文档）上 BM25 单路已经有 94%，
天花板就在那儿。混合的真实价值是**补齐两条单路各自漏掉的查询**——向量漏 5 条、
BM25 漏 3 条，融合后只剩 2 条；再叠应用层锚点补到最后 2 条，且全程零倒退
（上一路命中的查询在下一路必须还命中，`eval.py` 每次都打这个差值，负了就是回归）。

调参过程中有三个反直觉的结论，都写进了 docs/01 §七：`SCORE_THRESHOLD` 从 0.02 降到
0.012（原值会把 BM25 单路 rank1 误杀）、`doc_type` 加权只留 spec 1.3（faq/guide 不该衰减）、
**带过滤的检索要把词法置信门放宽**（"属于这件商品"本身就是第二重独立证据，
不必再苛求两个实词相交——但无过滤路径的门槛一动没动）。

## 路线图

- [x] 知识库数据（32 文档 / 3 品类 / 8 SKU，含跨品类同义词冲突用例）
- [x] 工具定义与执行层（6 工具，含 `transfer_to_human` 的转接原因枚举）
- [x] 设计文档（4 篇：知识库 / 工具 / 状态机 / 数据规范）
- [x] 检索管线五步（loader → chunker → indexer → retriever → eval）
- [x] 评测：50 条集四路对照（hit@5 = 100%，达标线 85%）
- [x] Agent 层：意图识别 + 状态机 + 会话记忆 + CLI 入口
- [x] Agent 层话题商品锚点（product_id 槽位过滤）
- [x] 评价域路由：新增第 8 个意图 `review_consult`，把"口碑/评价/优缺点"
      路由到 `review_knowledge`（此前该库建好但无人查，会拿商品详情作答）
- [ ] MySQL / Redis 接入（现为 SQLite + 进程内存会话）
- [ ] 多用户支持（工具层当前硬编码 `user_id="u1001"`）
- [ ] FastAPI 服务 + Streamlit 演示前端

## 技术栈

Python 3.10+ · ChromaDB · BGE-M3(FlagEmbedding) · rank_bm25 · jieba · DeepSeek API
(OpenAI 兼容) · python-dotenv · SQLite（演示数据）
规划接入：MySQL · Redis

> 状态机是**自研**的（`app/agent/graph.py` 手写分支 + 显式 Session 状态），没用 LangGraph。
> `requirements.txt` 里 langchain / langgraph 等是早期探索时留下的，当前代码并未引用，待裁剪。

## 已知局限（如实写，防止面试官问穿）

**演示规模**

- 知识库 32 文档 / 8 SKU，未验证 10 万+ 商品量级下的索引与检索性能
- BM25 索引建在进程内存，重启需重建（生产应外置，见 docs/04 Redis 设计）
- 会话记忆是进程内存 dict，多实例部署会串——接口已按 Redis `chat:ctx` 设计，替换即可

**尚未完成**

- 三个 RAG 类意图都检索 `product_knowledge`，评价域（`review_knowledge`）未接进路由
- 工具层硬编码 `user_id="u1001"`，多用户隔离未做
- 业务数据是 SQLite + JSON 模拟，未接真实 MySQL

**已知待修缺陷**

- `graph._transfer()` 未写会话历史：转人工这一轮在 `/history` 里看不到，会话摘要也会缺用户输入
- `graph.py` 里若干处 LLM 调用没有 try/except，网络抖动会中断会话（`intent.classify` 有兜底，其它没有）
- `executor` 只读缓存的 key 只哈希了 `arguments`，未含 `user_id`；将来加"我的订单列表"这类按用户维度的只读工具会串户（也是多用户支持的前置）
- `requirements.txt` 是 UTF-16 编码且含未使用的依赖，`pip install -r` 前需留意
