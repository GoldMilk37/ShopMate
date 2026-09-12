# Agent 状态机设计

## 一、整体流程

用户输入 → 意图识别 → 路由分发 → 执行节点 → 回复生成 → 结束/追问

## 二、状态节点定义

| 节点 | 职责 | 下一步 |
|---|---|---|
| intent_recognition | 识别用户意图 | 按意图路由 |
| rag_retrieval | 商品咨询类检索 | generate_response |
| param_compare | 参数对比 | generate_response |
| recommendation | 个性化推荐 | generate_response |
| tool_calling | 业务查询/操作 | generate_response |
| generate_response | 生成最终回复 | END |
| transfer_human | 转人工 | END |

## 三、意图路由规则

| 意图 | 路由目标 | 触发条件 |
|---|---|---|
| 商品咨询 | rag_retrieval | 询问商品功能、材质、适用场景 |
| 参数对比 | param_compare | 涉及两个及以上商品对比 |
| 个性化推荐 | recommendation | 表达购买需求但无明确商品 |
| 订单查询 | tool_calling | 涉及订单号、物流、价格、库存 |
| 售后处理 | tool_calling | 退换货、维修、投诉 |
| 闲聊/其他 | generate_response | 无法归类 |

## 四、置信度与兜底

- 意图识别置信度 < 0.6：追问澄清
- RAG 检索最高分 < 阈值：回复"暂无相关信息"+ 转人工入口
- 工具调用连续失败 2 次：降级转人工
- 用户连续 2 次表达不满：主动转人工

## 五、上下文管理

- 短期记忆：Redis 存储最近 10 轮对话
- 长期记忆：MySQL 存储用户画像、历史订单、偏好标签
- 跨轮意图追踪：记录当前对话主题，避免话题漂移
