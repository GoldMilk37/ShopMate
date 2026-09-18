# ShopMate 服务层包（薄封装）
# 数据流：HTTP 请求 → server.py 端点 → runtime.get_agent().handle() → 已有的 Agent 状态机
#
# 这一层**不含业务逻辑**。所有分支判断仍在 app/agent/graph.py / lg_graph.py 里，
# 本包只做两件事：把请求翻译成 agent.handle(sid, text) 调用，把结果翻译成 JSON。
# 验收标准：本包里出现业务 if/else 就说明逻辑漏到了错误的层。
