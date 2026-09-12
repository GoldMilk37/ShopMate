# ShopMate · AI 电商智能客服与导购系统

> 面向电商平台的智能客服 Agent：RAG 商品知识库检索 + Function Calling 业务系统调用，
> 覆盖商品咨询、参数对比、个性化推荐、订单查询、售后处理全场景。

## 架构

```
用户输入
   │
   ▼
┌──────────────────┐    意图路由（置信度 < 0.6 → 追问澄清）
│  意图识别          │
└────────┬─────────┘
         │ 按意图分发（03_agent_workflow.md）
   ┌─────┼──────────┬─────────────┐
   ▼     ▼          ▼             ▼
 商品咨询  参数对比    个性化推荐      订单/售后
   │     │          │             │
   ▼     ▼          ▼             ▼
 RAG 检索（混合：向量+BM25）      Function Calling
   │     （ChromaDB · BGE-M3）    （6 个工具，写操作需确认）
   └─────┴──────────┴─────────────┘
         │
         ▼
   回复生成 ── 低置信度/检索失败/用户不满 ×2 ──▶ 转人工（附对话摘要）
```

## 目录结构

```
ShopMate/
├── app/
│   └── retrieval/          # 检索管线：加载 → 分块 → 入库 → 混合召回
│       ├── schema.py       #   常量与数据结构（6 目录→3 collection 映射）
│       ├── loader.py       #   加载 rag_docs 产出 RawDoc ✅
│       ├── chunker.py      #   按文档类型分块（六种切法）🚧
│       ├── indexer.py      #   BGE-M3 向量化 + ChromaDB 入库 🚧
│       └── retriever.py    #   向量+BM25 混合检索，RRF 融合 🚧
├── data/
│   ├── rag_docs/           # RAG 知识库（32 文档 / 3 品类 / 10 商品）
│   │   ├── products/       #   商品详情 ×10
│   │   ├── specs/          #   规格参数 JSON ×10
│   │   ├── reviews/        #   用户评价摘要 ×10
│   │   ├── faq/            #   品类 FAQ ×3
│   │   ├── policies/       #   平台政策 ×2
│   │   └── guides/         #   选购指南 ×3
│   └── tools/              # 工具定义（6 个，data/tools/*.json）
├── docs/                   # 设计文档
│   ├── 01_rag_knowledge_base.md   # 知识库：分域/分块/混合检索/评测方案
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

# 2. 检索管线自测（当前可用部分）
python -m app.retrieval.loader  # 应打印：共加载 32 个文档 + SKU-10001 当前价 599

# 3. 建库 + 检索（chunker/indexer/retriever 完成后开放）
# python -m app.retrieval.indexer   # 全量入库
# python -m app.retrieval.retriever "通勤降噪耳机推荐"   # 混合检索
```

## 设计要点（面试可讲）

| 决策 | 理由 |
|---|---|
| 6 类文档 → 3 个 Collection | 按知识域而非文档类型分库：评价单独分库，避免口语化文本污染商品咨询 |
| 混合检索（向量 + BM25）+ RRF 融合 | 向量管"意思像"（"防水靠谱吗"），BM25 管"字面准"（"SKU-10001"）；RRF 免调分数阈值 |
| 实时数据不走 RAG | 价格/库存是实时状态，只经 Function Calling；RAG 里的 spec 只是带时间戳的快照 |
| 写操作走「提案 → 用户确认 → 执行」 | apply_after_sale 类工具由用户二次确认，防止 LLM 幻觉参数造成真实业务损失 |
| 转人工必附对话摘要 | 人机切换不转述等于重问一遍；摘要让坐席直接接手，多轮上下文记忆落到这个细节 |

## 评测方案（docs/01 §7）

- 评测集：50 条真实客服话术改写，人工标注应命中 doc_id
- 指标：hit@5；三组对照 = 纯 BM25 / 纯向量 / RRF 混合
- 目标：混合 ≥ 85%，比纯关键词提升约 30%

## 路线图

- [x] 知识库数据（32 文档，3 品类梯度 + 跨品类同义词冲突用例）
- [x] 工具定义（6/6，含 transfer_to_human 的转接原因枚举）
- [x] 设计文档（4 篇：知识库 / 工具 / 状态机 / 数据规范）
- [ ] 检索管线四步（loader ✅ → chunker → indexer → retriever）
- [ ] 评测：RAGAS + hit@5 对照实验（这就是“85%+”的出处）
- [ ] FastAPI 服务 + Streamlit 演示前端

## 技术栈

Python · ChromaDB · BGE-M3(FlagEmbedding) · rank_bm25 · jieba · FastAPI · Streamlit

## 已知局限（如实写，防止面试官问穿）

- 知识库为演示规模（32 文档），未验证 10 万+ 商品量级下的索引性能
- 工具调用目前仅有 schema，模拟执行器在 Agent 循环阶段实现
- BM25 索引建在进程内存，重启需重建（生产应外置，见 docs/04 Redis 设计）
