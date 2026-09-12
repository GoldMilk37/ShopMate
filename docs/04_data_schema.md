# 数据结构规范

## 一、RAG 文档元数据规范

每个 RAG 文档入库时需携带以下元数据：

| 字段 | 类型 | 说明 |
|---|---|---|
| doc_id | string | 文档唯一ID |
| doc_type | enum | product/spec/faq/policy/review/guide |
| product_id | string | 关联商品ID，非商品类为空 |
| category | string | 商品类目 |
| source | string | 数据来源 |
| updated_at | string | 更新时间 |
| version | string | 版本号 |

## 二、ChromaDB Collection 设计

| Collection | 用途 | 向量维度 |
|---|---|---|
| product_knowledge | 商品知识 | 1024 (BGE-M3) |
| policy_knowledge | 平台政策 | 1024 |
| review_knowledge | 用户评价 | 1024 |

## 三、MySQL 表设计（核心表）

- users：用户基础信息
- products：商品基础信息
- orders：订单表
- order_items：订单明细
- after_sales：售后记录
- chat_logs：对话日志
- tool_call_logs：工具调用日志

## 四、Redis Key 设计

| Key 模式 | 用途 | TTL |
|---|---|---|
| chat:ctx:{session_id} | 对话上下文 | 30min |
| user:profile:{user_id} | 用户画像缓存 | 1h |
| tool:cache:{tool}:{hash} | 工具结果缓存 | 5min |
