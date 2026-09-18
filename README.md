# ShopMate · AI 电商智能客服与导购系统

> 面向电商平台的智能客服 Agent：RAG 商品知识库检索 + Function Calling 业务系统调用，
> 覆盖商品咨询、参数对比、个性化推荐、订单查询、售后处理全场景。

**当前状态**（2026-09-18）：四层全部打通，离线自测全绿，检索评测 50 条集 hit@5 达到 **100%**。
三个入口：终端对话 `python -m app.agent.cli`，浏览器演示 `streamlit run app/agent/webui.py`
——后者把每轮**内部状态**（意图/置信度、检索召回与引用判定、工具调用、命中的兜底）摊在侧栏上，
是"点得开、看得见"的那个版本；HTTP 服务 `python -m app.api.server --serve`
——把同一套 Agent 包成**薄服务层**，让每轮内部决策轨迹变成任何人可 curl 的 JSON。

| 层 | 状态 | 自测入口 |
|---|---|---|
| RAG 检索（混合 + 话题商品锚点） | 已完成 | `python -m app.retrieval.eval` |
| 检索埋点（每轮检索落 JSONL） | 已完成 | `python -m app.retrieval.telemetry` |
| 工具层（6 工具，读写分级 + 确认门） | 已完成 | `python -m app.tools.executor` |
| Agent 层（8 类意图 + 状态机 + 会话记忆） | 已完成 | `python -m app.agent.intent` / `.graph` |
| LangGraph 编排版状态机（interrupt + checkpoint） | 已完成 | `python -m app.agent.lg_graph` |
| LangChain 版检索器（与手写版逐项对齐） | 已完成 | `python -m app.retrieval.lc_retriever` |
| 本轮轨迹契约与渲染 | 已完成 | `python -m app.agent.trace_view` |
| 演示前端（Streamlit） | 已完成 | `python -m app.agent.webui --e2e` |
| 服务层（FastAPI 薄封装，6 个端点） | 已完成 | `python -m app.api.server` |
| MySQL / Redis 接入 | 主动不做（见文末「明确不做」） | — |

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
┌──────────────────┐   ② 意图识别（LLM，8 类 + 置信度 + 槽位）
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

每一轮走完，走过的分支、意图与置信度、召回块与引用判定、工具调用、命中的兜底，
都被记进 `Session.trace`：前端侧栏照着它画，CLI 用 `/trace` 打印，RAG 轮另落一行
`data/logs/retrieval.jsonl`。轨迹是**报告**不是状态——不进 prompt、不进将来的 Redis
会话序列化，每轮重新绑定（不是清空），所以前端上一轮抓着的引用仍然完整。

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
│   │   ├── retriever.py    #   向量+BM25 混合检索、RRF 融合、元数据过滤（手写版）
│   │   ├── lc_retriever.py #   LangChain 版检索器：只换向量路，融合段与手写版共用
│   │   ├── telemetry.py    #   检索埋点：每轮落一行 JSONL + 引用代理指标
│   │   └── eval.py         #   50 条评测集，四路 hit@5 对照（支持双实现切换）
│   ├── tools/              # Function Calling 工具层
│   │   ├── registry.py     #   加载 JSON 工具定义，读写分级（WRITE_OPS）
│   │   ├── mock.py         #   模拟业务实现（SQLite + JSON 数据）
│   │   └── executor.py     #   执行编排：确认门 / 超时 / 缓存 / 异常收敛
│   ├── agent/              # Agent 状态机（手写与 LangGraph 两版并存）
│   │   ├── intent.py       #   意图识别：8 类 + 置信度 + product_id 槽位
│   │   ├── graph.py        #   状态机主体（手写版）：确认门 / 四条兜底 / 工具编排
│   │   ├── lg_graph.py     #   状态机（LangGraph 版）：interrupt 确认门 + checkpoint
│   │   ├── runtime.py      #   实现选择器：SHOPMATE_AGENT=lg 切 LangGraph 版
│   │   ├── session.py      #   会话记忆（进程内存版，接口按 Redis 设计）
│   │   ├── trace_view.py   #   本轮轨迹的键名契约 + 渲染（CLI 与前端共用）
│   │   ├── webui.py        #   Streamlit 演示前端：对话 + 侧栏摊开内部状态
│   │   └── cli.py          #   命令行交互入口
│   ├── api/                # 薄服务层：把上面这些包成 HTTP（**不含业务逻辑**）
│   │   ├── schemas.py      #   Pydantic 请求/响应模型（/docs 就是照它生成的）
│   │   └── server.py       #   FastAPI app + 6 个端点 + 离线自测
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
│   ├── graph_checkpoints.sqlite3  # LangGraph checkpoint（lg 版确认门断点续跑）
│   └── logs/               # 运行日志（jsonl）：工具调用 + 每轮检索埋点
├── docs/                   # 设计文档（决策依据都在这里）
│   ├── 01_rag_knowledge_base.md   # 知识库：分域/分块/混合检索/评测结论
│   ├── 02_tool_definitions.md     # 工具总览与调用原则（只读/写操作分级）
│   ├── 03_agent_workflow.md       # 状态机：意图路由/置信度兜底/上下文管理
│   ├── 04_data_schema.md          # 元数据/Collection/MySQL/Redis 规范
│   └── 05_demo_runbook.md         # 演示手册：怎么跑、问什么、看哪里
├── requirements.txt        # 只列直接依赖（6 + 4 个），不是 pip freeze 转储
├── .env.example            # key / 代理的填写模板，复制成 .env 用
├── LICENSE
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
cp .env.example .env             # Windows cmd 用 copy .env.example .env
# 然后编辑 .env，把 DEEPSEEK_API_KEY 换成你申请到的真 key
# 连不上 api.deepseek.com 的话，.env.example 里也写了代理怎么配

# 3. 离线自测（不联网、不依赖 key）
python -m app.retrieval.loader    # 应打印：共加载 32 个文档 + SKU-10001 当前价 599
python -m app.retrieval.chunker   # 应打印：124 个 chunk + 三项断言全通过
python -m app.retrieval.telemetry # 引用判定代理指标：锚点误报用例 + 落盘/开关
python -m app.tools.registry      # 应打印：6 个工具 + 读写分级
python -m app.tools.executor      # 应打印：9 项断言全通过（含缓存按用户隔离、越权防护）
python -m app.agent.session       # 应打印：10 轮成对截断
python -m app.agent.trace_view    # 轨迹渲染：标签覆盖 + 半成品轨迹不抛
python -m app.agent.lg_graph      # LangGraph 版状态机：确认门三路 + 断点续跑（桩测，不联网）
python -m app.retrieval.lc_retriever  # LangChain 版检索器：双版对照逐项一致（需已建库，不联网）
python -m app.agent.webui --e2e   # 前端无头端到端（AppTest，同样不联网）
python -m app.api.server          # 服务层 14 组断言（TestClient，同样不联网；lg 版再跑一遍）

# 4. 建库（首次会从 HuggingFace 拉 BGE-M3，之后离线可用）
python -m app.retrieval.indexer
python -m app.retrieval.retriever "通勤降噪耳机推荐"

# 5. 检索评测（四路对照，约 1 分钟）
python -m app.retrieval.eval

# 6. 跑 Agent（需 key）
python -m app.agent.intent        # 意图分类 9 条用例 + 解析降级 5 条断言（后者不需 key）
python -m app.agent.graph         # 状态机冒烟（含确认门、工具编排、转人工）
python -m app.agent.cli           # 人机对话；/new 换会话 /history 看记忆 /trace 看本轮状态
SHOPMATE_AGENT=lg python -m app.agent.cli   # 同样的事，换 LangGraph 编排版跑（interrupt 确认门 + 断点续跑）

# 7. 浏览器演示（需 key；首次检索要加载 BGE-M3，十几秒）
streamlit run app/agent/webui.py  # 侧栏能点开看本轮内部状态；有「预热模型」按钮
                                  # 同样支持 SHOPMATE_AGENT=lg 切 LangGraph 版

# 8. HTTP 服务（需 key）：把同一个 Agent 暴露成 JSON 接口
python -m app.api.server --serve            # 绑 127.0.0.1:8000，单 worker（代码里写死）
# python -m app.api.server --serve --port 8001   # 8000 被占时换端口
curl 127.0.0.1:8000/health                  # 存活 + 单 worker 证据（pid / uptime_s / sessions）
curl 127.0.0.1:8000/tools                   # 6 个工具 + 读写分级
curl -X POST 127.0.0.1:8000/chat -H "Content-Type: application/json" \
     -d '{"message":"SKU-10001 续航多久"}'   # 响应里带本轮完整轨迹（trace.text 与 CLI /trace 同源）
# 交互式文档：浏览器打开 http://127.0.0.1:8000/docs
```

> ⚠️ **调用 `/chat` 的客户端必须自己设超时**。首个 RAG 轮实测约 13 秒（加载 BGE-M3），
> 而 `httpx` 默认超时是 5 秒——用默认值会"看起来服务挂了，其实它在正常工作"。
> 本项目自测用的是 `fastapi.testclient.TestClient`，**实测它不施加超时**
> （`def` 端点里 `sleep(7)` 也正常返回 200），所以自测不会替你把这个问题暴露出来。

> 前端怎么验：`python -m app.agent.webui --e2e` 是**真正的**端到端——它用
> `streamlit.testing.v1.AppTest` 在同进程里执行整个脚本（桩掉检索与 LLM，全程离线），
> 断言脚本真的跑起来了、一问一答进了会话、侧栏把轨迹画出来了。
> 而 `curl /_stcore/health` 返回 200 **只说明服务器启动成功**：脚本要等 websocket
> 连上才执行，所以健康探测证明不了应用正确。两者别互相替代。

> **要演给别人看？** 见 [docs/05_demo_runbook.md](docs/05_demo_runbook.md)：
> 问什么、看哪里、哪个问题对应哪条支路、哪两条兜底别在现场赌。

> **环境坑**：系统 Python 缺 `jieba` 等依赖，一律用 `.venv/Scripts/python.exe` 跑。
> BGE-M3 已缓存在本地，indexer 有缓存时不再联网。

> **环境坑（Windows 系统代理）**：Python 的 HTTP 客户端会经由
> `urllib.request.getproxies()` 读到**注册表**里 Clash 的"系统代理"设置，而
> **`curl` 不读注册表**——于是会出现"`curl` 探着通、脚本却连不上"，报错还是
> 一句与网络无关的 `EOF occurred in violation of protocol`。本项目所有出网
> 客户端统一走 `app/llm/client.py` 的 D5：**只有显式设了 `HTTPS_PROXY` 才走
> 代理，否则一律直连**。所以要带代理时别去动系统设置，直接在命令行写
> `HTTPS_PROXY=http://127.0.0.1:7890` 即可。

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
| **轨迹重置用"重新绑定"而非 `.clear()`** | 前端上一轮抓着的那份 trace 引用要还能指向完整的旧轨迹；`clear()` 会把它当场掏空，表现是"侧栏刚才还有，一刷新没了" |
| **轨迹重置包在 `handle` 外层 `try/finally`** | 主干有 4 个返回点、工具支路内部还会再转人工，逐点写重置必然漏一个；耗时也只在 `finally` 里记才不漏异常路径 |
| **埋点只记在 Agent 层，不在检索器里** | `search_with_product_focus` 内部会再调一次 `search`，记在检索器里同一个 query 会落两行、命中率凭空翻倍；而且"回复"只有 Agent 层才有 |
| **"检索为空"那轮也要记** | 原先这条路径是提前 return 的，那轮不落日志——而**兜底触发率**恰恰是埋点里最有意思的数，漏掉它等于白埋 |

## 检索评测（50 条集，docs/01 §七）

换检索实现时评测就是回归网：`SHOPMATE_RETRIEVER=lc python -m app.retrieval.eval` 跑
LangChain 版（langchain_chroma 向量路 + 共用融合段），与手写版结果应逐项一致
（当前两版均为 90 / 94 / 96 / 100，未命中清单相同）。

评测集是 50 条真实客服话术改写，人工标注应命中的 chunk，`hit@5` 五路对照：

| 检索方式 | hit@5 | hit@1 | MRR | 未命中 |
|---|---|---|---|---|
| 纯向量 | 90% | 70% | 0.776 | 5 |
| 纯 BM25 | 94% | 68% | 0.788 | 3 |
| RRF 混合 | 96% | 72% | 0.809 | 2 |
| **混合 + 话题商品锚点** | **100%** | 88% | 0.926 | 0 |
| 混合 + rerank 精排 | 96% | 74% | 0.826 | 2 |

最后一路要 `SHOPMATE_RERANK=1` 才跑（cross-encoder 本地推理，全量约 16 分钟）。
它的 hit@5 低于"话题商品锚点"那路**不是回归**：锚点那路额外吃了 `product_id` 槽位
这个输入（Agent 层已经知道用户在聊哪件商品），rerank 那路只用 query 本身，两者
不是同口径；rerank 真正的对照组是它上面那行"RRF 混合"，+2% hit@1、+0.017 MRR。

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
- [x] 工程收尾：README 与代码同步、`requirements.txt` 重写（补回漏掉的 `jieba`）、
      `.env.example` / `LICENSE`
- [x] 检索埋点（docs/01 §五 的设计落地）：每轮 RAG 落一行 `data/logs/retrieval.jsonl`，
      含 query / 召回块 / 引用判定 / outcome；引用率按**代理指标**标注（见「已知局限」）
- [x] Streamlit 演示前端：对话 + 侧栏把本轮内部状态摊开（意图、召回、工具、兜底），
      并附无头端到端自测 `python -m app.agent.webui --e2e`
- [x] LangGraph 编排版状态机（`app/agent/lg_graph.py`）：interrupt() 确认门 +
      SqliteSaver checkpoint（断点续跑），`SHOPMATE_AGENT=lg` 切换，手写版保留
- [x] LangChain 版检索器（`app/retrieval/lc_retriever.py`）：langchain_chroma
      向量路 + 共用融合段，`SHOPMATE_RETRIEVER=lc` 切换；双版评测逐项一致
      （均为 90 / 94 / 96 / 100）
- [x] 服务层（`app/api/`，FastAPI 薄封装）：把 `agent.handle()` 包成 HTTP，把每轮内部
      决策轨迹从 Streamlit 侧栏里解放成**任何人可 curl 的 JSON**。6 个端点 / 3 个文件 /
      零新增业务逻辑（"薄"的护栏：`openapi` 操作数 == 6，加端点必须改断言）。
      单 worker 是运行约束、代码里强制不了，于是把违约做成**可检验**的：
      `/health` 暴露 pid + uptime_s + sessions，配 `--workers 2` 能当场演示会话分裂
- [x] ~~MySQL / Redis 接入~~ → 主动不做，理由见文末「明确不做」
- [x] ~~多用户支持~~ → 注入点已收口（D6），接登录态时改一处即可

## 技术栈

Python 3.10+ · LangGraph（Agent 编排版）· LangChain（Embeddings/Chroma 集成）·
ChromaDB · BGE-M3(FlagEmbedding) · rank-bm25 · jieba · DeepSeek API
(OpenAI 兼容) · FastAPI + uvicorn（薄服务层 `app/api/`）· Streamlit（演示前端）·
SQLite（演示数据 + checkpoint 落盘）

> **手写与框架并存，两种实现可切换**（2026-09 起）：
> - 状态机：手写版 `app/agent/graph.py`（分支 + 显式 Session）与 LangGraph 版
>   `app/agent/lg_graph.py`（StateGraph + `interrupt()` 确认门 + SqliteSaver
>   checkpoint，断点续跑）同任务并存；`SHOPMATE_AGENT=lg` 切换，默认手写版。
> - 检索器：手写版 `app/retrieval/retriever.py`（chromadb 直连）与 LangChain 版
>   `app/retrieval/lc_retriever.py`（langchain_chroma + Embeddings 接口）共用同一
>   融合段（RRF/词法门/加权/阈值在 `retriever._fuse_and_format`），评测集证明两版
>   逐项一致；`SHOPMATE_RETRIEVER=lc` 切换，默认手写版。
> - 早期探索遗留的 langchain 曾因"从未被引用"移除，本次是带着手写理解后的换回，
>   `requirements.txt` 只列代码真正 import 的包。
> MySQL / Redis 的方案已设计但**演示规模不接**，理由见文末「明确不做」。

## 已知局限（如实写，防止面试官问穿）

**演示规模**

- 知识库 32 文档 / 8 SKU，未验证 10 万+ 商品量级下的索引与检索性能
- BM25 索引建在进程内存，重启需重建（生产应外置，见 docs/04 Redis 设计）
- 会话记忆是进程内存 dict，多实例部署会串——接口已按 Redis `chat:ctx` 设计，替换即可

**已实现，但有保留（别当成准确率或生产级）**

- **引用率是代理指标，不是准确率**：`app/retrieval/telemetry.py` 每轮记
  query / 召回块 / outcome / 引用判定，但"这块资料用没用上"是用**字符 4-gram 重叠率**
  猜的（扣掉本轮各块共有的样板文字，否则回复提一句商品名就会把该商品所有块判成被引用）。
  它度量的是**模型抄了多少字面**，不是用了多少资料——**实测同一条 query 前后跑两次，
  召回完全相同（5 个 chunk_id 与 rrf_score 逐一对齐），引用块数却是 1 和 3**，差异全在
  措辞（一次把块里的"（312 次）"原样抄了，一次改写成自然语言）。所以它**会低报，且
  方差比绝对值更值得注意**，别把单轮的数当质量结论。另一个盲区是**看不见"该召回而没召回"**
  （那是检索侧的事，本埋点只看生成侧）。记录里带 `citation.method` 字段、每块原始重叠率
  也都留着，改阈值不必重跑 LLM。它 **≠** 上面那个 hit@5——hit@5 靠 50 条**人工标注集**、
  回答的是"该召回的召回了没有"，两者不许混着说
- **前端的两条进程内假设**：`SessionStore` 无锁（两个标签页并发会互踩）、浏览器刷新会
  遗弃一个 Session（无 TTL）。演示规模靠侧栏「结束会话」按钮手动回收，与
  "会话记忆是进程内存"那条同源
- **服务层的四条已知代价**（`app/api/`，都记录但不修，因为修完就不是"薄"了）：
  ① **单 worker 服务自己强制不了**——`--workers 2` 不报错，只会让会话随机丢失；
  ② **同 sid 并发会互踩**（threadpool 默认 40 + `SessionStore` 无锁，`dissatisfaction += 1`
  和 `history.append` 是读-改-写）。**这条的性质变了**：CLI 单线程、Streamlit 一个连接一条
  脚本线程，几乎踩不到；加了 HTTP 之后它变成**能通过网络真踩到的竞态**，所以配两个约束——
  部署上单 worker、使用上同一个 sid 不要并发发两条；
  ③ **无 TTL / 无上限**：HTTP 让 sid **由客户端给**（CLI 时代是进程自己生成的），这是新的
  外部可控的内存增长入口。只挡了长度（≤64 字符）没挡数量，`/health` 的 `sessions` 是唯一
  观测口；④ **`/sessions/{sid}/trace` 只有最近一轮**，不是审计日志（`session.py` D5 的既定
  取舍：轨迹是报告不是状态）。跨轮历史在 `data/logs/retrieval.jsonl`
- **lg 版下 `dissatisfaction` / `tool_fail_streak` 恒为 0**：Session 在 lg 下只是视图模型，
  只镜像 `trace` / `history` / `current_product_id` / `pending_write` 四个字段。
  `GET /sessions/{sid}` 的响应里用 `notes` 自动声明，**不许让调用方读成"用户从没不满过"**
- **RAGAS 的 answer_relevancy 这一项不可信，别当结论引用**：15 条试跑里出现 4 个
  **精确的 0.000**，查下去不空回复、也不是抽样被剥（对 Qwen 放行 n 无效，langchain
  那条路径压根没把 n 传进请求体）。真正的成因是**生成侧的不确定性传导**：RAG 用
  `temperature=0.3`，同一 query 每轮回复都不同，而该指标要先用答案**反解出问题**
  再比向量余弦，对措辞极敏感——同一条 query 复跑一次 0.000、一次 0.8127。
  所以它的分数由"这一轮生成恰好长什么样"主导，**要报就得报多轮区间，不能报单轮均值**。
  faithfulness 与 context_precision 没有反解这一步，不受影响。详见 docs/01 §七
**明确不做（不是欠债，是判断）**

- **MySQL / Redis 接入**：8 SKU 的演示量级下 SQLite + 进程内存没有可观察的差别，
  换上去收益是零。规范已写在 [docs/04_data_schema.md](docs/04_data_schema.md)，
  接了真实业务量再按图替换。**注意它和下面那条是同一件事的两面**：正因为会话在进程内存
  （没接 Redis），HTTP 服务才只能单 worker
- **把服务层做厚（SSE / 鉴权 / CORS / 多副本）**：`app/api/` 有一个 FastAPI 薄封装，
  但它**刻意停在边界上**——3 个文件、6 个端点、不 import 任何业务模块。SSE（首轮 13 秒
  确实难等）、鉴权（每个 `/chat` 是 1~4 次 DeepSeek 调用）、CORS、多副本，**每一条都有
  正当理由，每一条也都会让"薄"不再是薄**；加鉴权的那一刻它就不再是薄层，所以那是这个
  定位的**天然期限**，不是欠账。单 worker 是运行约束、服务在代码里强制不了
  （`--workers 2` 照样跑，只是同一会话被分到不同进程、各拿一个空会话），
  所以把它做成了**可检验的**：`/health` 的 `pid` + `uptime_s` + `sessions` 就是证据。
  **停在边界上是判断，越过边界才是欠债。**
- **多用户会话**：`user_id` 目前硬编码 `"u1001"`，但**身份注入点已经收口**在
  `executor.execute`（决策 D6：丢弃模型自填的 user_id，一律以会话身份覆盖）。
  接登录态时只改这一处来源。当前单用户演示形态下不会串户
- **自动化测试**：只有各模块的 `__main__` 自测（有断言、可跑），未引入 pytest，
  因而没有 CI。自测覆盖的是"机制是否符合预期"，不是回归网

**上一轮已知缺陷（已全部修复，留档备查）**

| 缺陷 | 修法 | commit |
|---|---|---|
| `graph._transfer()` 未写会话历史 | 转人工这一轮补进 sessions，摘要带上系统发起的 reason | `61cfec0` |
| 若干处 LLM 调用没兜底，网络抖动中断会话 | D9 降级 + 重试判据收口到 `client.safe_call` | `61cfec0` / `7dce383` |
| 模型自填的 `user_id` 即权威身份（可越权查他人订单） | D6：丢弃模型填的，以会话身份覆盖 | `61cfec0` |
| 只读缓存 key 未含 `user_id`，按用户维度的工具会串户 | D5：key 哈希 `{user_id, arguments}` | `61cfec0` |
| 评价域建了库但没意图会路由过去 | 新增第 8 个意图 `review_consult` → `review_knowledge` | `876089c` |
| `intent.classify` 无重试，抖一下每句话都变"请澄清" | 复用 `client.safe_call`，并把解析失败与出网失败分开报 | `7dce383` |
| `requirements.txt` 是 UTF-16、漏 `jieba`、混入未引用依赖 | 转 UTF-8；重写为直接依赖清单并补回 `jieba` | `05230df` / `e005907` |
| 每轮内部决策无留痕，前端无从展示，docs/01 §五 的埋点承诺空着 | 本轮轨迹（`trace_view` 定契约）+ 检索埋点（`telemetry`）+ 侧栏可视化 | `31a6330` |
