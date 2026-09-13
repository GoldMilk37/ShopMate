# RAG 知识库设计说明

> 术语速览（面向零基础）：
> - **RAG**（Retrieval-Augmented Generation，检索增强生成）：让 LLM 回答前先去知识库里查资料，把查到的内容连同问题一起给模型，减少胡编（幻觉）。
> - **Embedding / 向量**：把一段文字变成一串数字，语义相近的文字数字也相近，用来做"意思层面"的搜索。
> - **BGE-M3**：开源中文向量模型（BAAI 出品），"M3"指多语言/多粒度/多功能。负责把文档和问题都变成向量。
> - **BM25**：经典关键词检索算法，按词出现频率和稀有程度打分。向量检索管"意思像"，它管"字面准"（如型号 SKU-10001、专有名词）。
> - **ChromaDB**：轻量向量数据库，存向量并支持"找出最相似的 k 条"。
> - **Chunk（分块）**：把长文档切成小段再入库，检索时只取相关的小段，省 token 且更准。

## 一、设计目标

| 指标 | 目标值 |
|---|---|
| 混合检索命中率（hit@5） | ≥ 85% |
| 检索召回方式 | 向量（BGE-M3）+ BM25 混合 |
| 知识库规模 | 目标 10 万+ 商品（当前示例数据 1 件） |
| 单次检索延迟 | < 500ms |
| 检索质量基线 | 纯关键词检索（用于对比实验，预期提升约 30%） |

## 二、知识域划分与 Collection 映射

检索粒度按**知识域**而非文档类型划分：同一域的文档入库到同一个 Collection，检索时只查相关知识域，减少跨域噪音。

源文档 6 类 → ChromaDB 3 个 Collection，映射规则：

| rag_docs 目录 | doc_type | 入库 Collection | 说明 |
|---|---|---|---|
| products/ | product | product_knowledge | 商品详情 |
| specs/ | spec | product_knowledge | 规格参数，与详情同域 |
| guides/ | guide | product_knowledge | 选购指南，导购语义同域 |
| faq/ | faq | product_knowledge | 常见问答，同域 |
| policies/ | policy | policy_knowledge | 平台政策，独立域（客服高频，误召回代价高） |
| reviews/ | review | review_knowledge | 用户评价，独立域（避免口语化评价污染商品咨询） |

依据：
1. 商品咨询 / 选购推荐意图 → 查 `product_knowledge`（详情+规格+FAQ+指南一次可召回）
2. 售后、物流、退换货意图 → 查 `policy_knowledge`
3. "口碑怎么样 / 有什么缺点"类查询 → 查 `review_knowledge`

> 路由到哪个 Collection 由 Agent 状态机的意图识别节点决定（见 [03_agent_workflow.md](03_agent_workflow.md)），本文件只定义知识库侧规则。

## 三、元数据规范

入库时每个 chunk 携带元数据（ChromaDB metadata 字段），用于过滤与过滤式检索：

| 字段 | 类型 | 说明 | 示例 |
|---|---|---|---|
| doc_id | string | 文档唯一ID | prod:SKU-10001 |
| doc_type | enum | product/spec/faq/policy/review/guide | spec |
| product_id | string | 关联商品ID，非商品类为空 | SKU-10001 |
| category | string | 商品类目（一级） | 数码/耳机 |
| brand | string | 品牌，用于按品牌过滤 | SoundCore |
| source | string | 数据来源文件 | specs/SKU-10001.json |
| updated_at | string | 更新时间 | 2026-09-12 |
| version | string | 版本号，增量更新用 | v3 |

用途示例：用户问"SoundCore 的耳机"→ 过滤 `brand=SoundCore` 后再向量检索；价格类问题只信 spec 文档，避免 review 里的旧价格干扰。

## 四、分块策略（Chunking）

| 文档类型 | 分块方式 | 理由 |
|---|---|---|
| products/*.md | 按 `##` 二级标题切 | 每个卖点/场景天然独立 |
| specs/*.json | 整文件一个 chunk（1 件商品 < 500 token） | 结构化数据不宜拆散 |
| faq/*.md | 一问一答一个 chunk | 问答对是最小检索单元 |
| policies/*.md | 按 `##` 切，父标题拼进正文 | "特殊品类规则"脱离"退换货政策"会歧义 |
| reviews/*.md | 好/差评关键词各自一块，摘录跟各自块 | 正负面分开，避免召回混淆 |
| guides/*.md | 按 `##` 切 | 每个"按需求选"小节独立成篇 |

通用规则：
- chunk 上限 512 token，超出再对半切
- 每个 chunk 头部拼接商品名 + 品牌：`【无线降噪蓝牙耳机 Pro | SoundCore】正文...`（提高向量区分度）
- spec 类 chunk 的 json 展平为 "键: 值" 文本行再入库

## 五、混合检索流程

```
query
  ├─→ 向量检索：BGE-M3(query) → ChromaDB top-k（k=5）
  ├─→ BM25 检索：同一 Collection 倒排索引 top-k
  ↓
RRF 融合（Reciprocal Rank Fusion，按两边排名取倒数求和）
  ↓
按 doc_type 加权（咨询意图：spec > product > faq > guide）
  ↓
top-5 进入 LLM 上下文
```

> RRF 通俗版：两个渠道各给候选排名，两条都排前面的最终排名高。不用调分数阈值，工程上省事且稳。

兜底规则（与状态机衔接）：
- RRF 融合后最高分 < 阈值 γ（初始 0.02，待调参）→ 回复"暂无相关信息" + 转人工入口
- 命中率统计埋点：每次检索记录 query、命中文档、最终是否被 LLM 引用（用于算 85%+ 指标）

## 六、更新与版本

- 增量入库：同一 `doc_id` 新 version 覆盖旧 chunk（按 metadata `doc_id` 先删后插）
- 商品价格/库存不进 RAG——属实时数据，只经 Function Calling 查询（见 [02_tool_definitions.md](02_tool_definitions.md)），RAG 里的 spec 只存"发布时快照"并标注 updated_at
- 全量重建：脚本一键重跑（每个 Collection 独立重建，互不影响）

## 七、评测方案（怎么证明 85%+）

1. 构造评测集：50 条真实客服话术改写的问题，人工标注每条应命中的 doc_id（按"可接受文档集合"标注，实现见 `app/retrieval/eval.py`）
2. 指标：hit@5（top-5 里包含正确文档即为命中，判到文档级）
3. 对照组：纯 BM25 / 纯向量 / RRF 混合，三者跑同一评测集
4. 预期：混合 ≥ 85%，比纯关键词提升约 30%（即纯 BM25 约 55-65%）

### 首轮实测结果（2026-09-13，50 条）

| 检索方式 | hit@5 |
|---|---|
| 纯向量 | 90% |
| 纯 BM25 | 94% |
| RRF 混合 | **96%** ✅（≥85% 达标） |

实测推翻/修正了三处设计假设（教训已回写进代码注释）：

1. **兜底阈值 0.02 过严**（§五）：会把"向量路超距、BM25 单路精确命中"的 query（如"压胶洗完会起泡吗"）整批误杀为"暂无信息"。已降为 0.012，并新增词法置信门——仅 BM25 支持的候选须与 query 相交 ≥2 个实词，专拦停用词级别的词法巧合（"量子涨落对股市的影响"实测靠"对/的"攒出 4.76 分，停用词过滤后归零）。BM25 两侧统一走停用词过滤的 `_tokenize`。
2. **doc_type 加权不宜惩罚 faq/guide**（§五）：guide 0.8 的衰减把"预算三百以内"的指南文档挤出 top5。加权只保留 spec 1.3（价格只信 spec 的原意），其余一律 1.0。
3. **chunk 锚点必须含商品ID**（§四）：评价文档的 H1（含 SKU 号）不入块，用户报型号提问时 BM25 无从匹配。锚点改为 `【SKU-10001 无线降噪蓝牙耳机 Pro | SoundCore】`。

剩余 2 条未命中均为"报了型号的评价类查询"，按 §三元数据规范的正解是在 Agent 层做 `product_id` 元数据过滤后再检索（槽位过滤），不在检索器内打补丁。另注：小语料（50 条级）下 RRF_K 20~60 对 hit@5 无影响，k 值暂不动。