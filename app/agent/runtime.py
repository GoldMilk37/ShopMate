"""Agent 实现选择器：手写版 graph.Agent 与 LangGraph 版 LgAgent 二选一。

用法（默认手写版，行为与改造前完全一致）：
    python -m app.agent.cli                    # 手写版
    SHOPMATE_AGENT=lg python -m app.agent.cli  # LangGraph 版

两个设计决策：
    D1 懒加载：默认路径不 import lg_graph——它连带 SqliteSaver 与 checkpoint
       落盘文件，不该让没切开关的用户无感背上。切换开关读环境变量
       SHOPMATE_AGENT（lg / langgraph → LangGraph 版，其余 → 手写版）。
    D2 两者 handle(session_id, text) -> str 签名一致，cli/webui 只换这一处
       import，前端（侧栏读 store 里的 Session 视图）零改动。
"""
import os


def get_agent():
    """按环境变量返回状态机实例（手写版 / LangGraph 版）。"""
    if os.environ.get("SHOPMATE_AGENT", "").lower() in ("lg", "langgraph"):
        from .lg_graph import lg_agent
        return lg_agent
    from .graph import agent
    return agent
