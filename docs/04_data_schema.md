# 数据结构规范

## 一、RAG 文档元数据规范

每个 RAG 文档入库时需携带以下元数据：

| 字段 | 类型 | 说明 |
|---|---|---|
| doc_id | string | 文档唯一ID |
| doc_type | enum | product/spec/faq/policy/review/guide |
| product_id | string | 关联商品ID，非商品类为空 |
| category | string | 商品类目 |
| brand | string | 品牌，用于按品牌过滤 |
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

"""
1. chat:ctx:{session_id} —— 对话上下文
按 session_id 存，因为对话是"一次会话"维度
TTL 30min：用户 30 分钟不说话，就认为会话结束，清掉省内存
存的内容：最近 10 轮对话（docs/03 说的）
2. user:profile:{user_id} —— 用户画像缓存
注意"缓存"两个字：画像主存在 MySQL，Redis 只是缓存加速
TTL 1h：比对话长，因为画像不常变
为什么用缓存？画像查询频繁（每次推荐都要用），但更新少，典型的缓存场景
3. tool:cache:{tool}:{hash} —— 工具结果缓存
这是新东西，前三篇没提
{tool} = 工具名，{hash} = 参数的哈希
TTL 5min：很短，因为工具查的是实时数据（库存、价格），缓存太久就失真了
用途：用户连问两次"这个还有货吗"，第二次直接读缓存，不重复调业务系统
"""