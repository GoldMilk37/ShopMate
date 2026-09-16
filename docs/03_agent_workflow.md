# Agent 状态机设计

## 一、整体流程

用户输入 → 意图识别 → 路由分发 → 执行节点 → 回复生成 → 结束/追问

## 二、状态节点定义

| 节点 | 职责 | 下一步 |
|---|---|---|
| intent_recognition | 识别用户意图 | 按意图路由 |
| rag_retrieval | 商品咨询 / 口碑评价类检索 | generate_response |
| param_compare | 参数对比 | generate_response |
| recommendation | 个性化推荐 | generate_response |
| tool_calling | 业务查询/操作 | generate_response |
| generate_response | 生成最终回复 | END |
| transfer_human | 转人工 | END |

## 三、意图路由规则

| 意图 | 路由目标 | 触发条件 |
|---|---|---|
| 商品咨询 | rag_retrieval | 询问商品**客观信息**：功能、材质、参数、规格、适用场景 |
| 用户评价 | rag_retrieval | 询问**主观体验**：口碑、评价、优缺点、值不值得买、别人怎么说 |
| 参数对比 | param_compare | 涉及两个及以上商品对比 |
| 个性化推荐 | recommendation | 表达购买需求但无明确商品 |
| 订单查询 | tool_calling | 涉及订单号、物流、价格、库存 |
| 售后处理 | tool_calling | 退换货、维修、投诉 |
| 闲聊/其他 | generate_response | 无法归类 |

> 商品咨询与用户评价是**同一节点、不同库**：前者查 `product_knowledge`，
> 后者查 `review_knowledge`，生成端的侧重点也不同（咨询答事实，评价要好评
> 差评两边都讲）。分成两个意图而不是在节点内做关键词二次判定，是因为
> 规则难维护，且会让置信度语义分裂——一半来自模型、一半来自规则。
> 边界拿不准时看问的是"它是什么"（咨询）还是"它好不好用"（评价）。

## 四、置信度与兜底

- 意图识别置信度 < 0.6：追问澄清
- RAG 检索最高分 < 阈值：回复"暂无相关信息"+ 转人工入口
- 工具调用连续失败 2 次：降级转人工
- 用户连续 2 次表达不满：主动转人工

## 五、上下文管理

- 短期记忆：Redis 存储最近 10 轮对话
- 长期记忆：MySQL 存储用户画像、历史订单、偏好标签
- 跨轮意图追踪：记录当前对话主题，避免话题漂移

## 六、LangGraph 编排版（2026-09 新增，与手写版并存）

`app/agent/lg_graph.py` 用 LangGraph StateGraph 重写了本文档的状态机，
两种实现同任务并存，`SHOPMATE_AGENT=lg` 切换（默认手写版）。对照表：

| 本文概念 | 手写版（graph.py） | LangGraph 版（lg_graph.py） |
|---|---|---|
| 确认门（§二 tool_calling 的跨轮确认） | pending_write 存 Session + 意图识别前关键词判定 | `interrupt()` 挂起 + `Command(resume=)`，暂停态由 checkpointer 承载 |
| 置信度兜底 / 不满兜底（§四） | handle 内 if 分支 | classify 节点后的 conditional_edges |
| 工具编排循环 | for 循环 ≤3 轮 | tool_llm ⇄ tool_exec 环 + 计数器 |
| 会话记忆（§五，Redis chat:ctx） | 进程内存 SessionStore | SqliteSaver checkpoint 落盘（进程重启后确认门仍可续） |
| 本轮轨迹 | Session.trace + try/finally 重置 | state["trace"]，ingest 节点每轮重建 |

关键差异：手写版确认门挂在进程内存里，重启即丢；LangGraph 版的暂停点
持久化在 `data/graph_checkpoints.sqlite3`，换进程 resume 照样成立——
这是框架原语替代手写逻辑后**多出来的能力**。轨迹契约（TRACE_KEYS）、
提示词、工具执行（executor）、检索（retriever）、LLM 出口（client）两版
完全共用，行为除持久化外等价（各有离线自测钉住：`python -m app.agent.graph`
/ `python -m app.agent.lg_graph`）。
