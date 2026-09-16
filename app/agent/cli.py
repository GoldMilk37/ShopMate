"""聊天入口：命令行 REPL。已实现。

模块速览：
    main()              整个 Agent 的最终用户入口

两个设计决策：
    D1 会话 id 跟进程走：每次启动生成新 session_id（时间戳+随机后缀），
       进程内 SessionStore 天然隔离，不会把上一次调试的上下文带进来。
       将来换 API 服务时，这里改成从请求头取真实 session/user id。
    D2 命令不进状态机：/exit 退出、/new 换会话、/history 查看记忆，
       在 REPL 层拦截。业务命令混进对话历史会污染意图识别的上下文。
    D3 /trace 直接复用 trace_view.render_text：终端和 Streamlit 侧栏看的是
       同一份轨迹的两种画法，格式化逻辑只有一处（trace_view），这里不重写。
"""
import random
import time

from .runtime import get_agent
from .session import store
from .trace_view import render_text


def main() -> None:
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    agent = get_agent()   # SHOPMATE_AGENT=lg 切 LangGraph 版（runtime.py D1）

    sid = f"s-{time.strftime('%H%M%S')}-{random.randint(100, 999)}"
    print("=" * 56)
    print("ShopMate 智能客服「小搭」  "
          "(exit 退出 / new 新会话 / history 看记忆 / trace 看本轮内部状态)")
    print("=" * 56)
    while True:
        try:
            text = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):            # Ctrl+C / Ctrl+Z 都优雅退出
            print("\n再见！")
            break
        if not text:
            continue
        if text in ("/exit", "exit", "退出"):
            print("再见！")
            break
        if text in ("/new", "new"):                      # 换会话 = 全新上下文
            sid = f"s-{time.strftime('%H%M%S')}-{random.randint(100, 999)}"
            print(f"(已开启新会话 {sid})")
            continue
        if text in ("/history", "history"):
            h = store.get(sid).history
            print(f"(共 {len(h) // 2} 轮)")
            for m in h:
                print(f"  [{m['role']}] {m['content'][:60]}")
            continue
        if text in ("/trace", "trace"):
            # D3：与 Streamlit 侧栏同源。换个会话后这里会是空的——那是实话
            print(render_text(store.get(sid).trace))
            continue
        print(f"\n小搭: {agent.handle(sid, text)}")


if __name__ == "__main__":
    main()
