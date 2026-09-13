"""LLM 调用层：DeepSeek 客户端封装（全项目唯一的出网 LLM 出口）。已实现。

模块速览：
    get_client()       进程级单例的 OpenAI 兼容客户端
    chat(...)          返回 assistant 消息对象（要 tool_calls / 结构化输出时用）
    chat_text(...)     便捷版，只要文本内容

三个设计决策：
    D1 .env 先于环境变量读取：key 不进代码、不进 git（.gitignore 已忽略），
       格式一行一个 KEY=VALUE。os.environ.setdefault——系统里已设置的
       环境变量优先，本地 .env 兜底，两种给 key 的方式都行。
    D2 单例 + 懒加载：和 indexer.get_model 同一个模式，首次调用才建
       客户端（也才校验 key），不联网。缺 key 的报错里直接告诉用户
       往哪写，而不是抛一个裸的 openai 认证错。
    D3 json_mode 的使用纪律：DeepSeek 的 response_format=json_object
       要求 prompt 里出现 "json" 字样才生效（官方约束），调用方写
       prompt 时必须遵守；这里只传开关不替调用方改 prompt——约定
       写在两边 docstring 里，出错时对人。

依赖链：intent.classify / graph 的生成与工具编排都只 import 这里。
"""
import os
import threading
from pathlib import Path

# 项目根的 .env（与 data/ 平级）
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

# DeepSeek 的 OpenAI 兼容端点与模型名
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"


def _load_dotenv() -> None:
    """D1：把 .env 里的 KEY=VALUE 塞进环境变量（已有的不覆盖）。"""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()          # import 即生效，get_client 拿到的环境一定已就绪


_client = None
_lock = threading.Lock()


def get_client():
    """D2：返回单例客户端。缺 key 抛 RuntimeError，报错里带补救指引。"""
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is None:
            key = os.environ.get("DEEPSEEK_API_KEY")
            if not key:
                raise RuntimeError(
                    "缺少 DEEPSEEK_API_KEY。请在项目根目录 .env 文件里加一行：\n"
                    "    DEEPSEEK_API_KEY=sk-你的key\n"
                    "（.env 已被 git 忽略，不会提交）")
            from openai import OpenAI
            _client = OpenAI(base_url=BASE_URL, api_key=key)
    return _client


def chat(messages: list[dict], *, tools: list | None = None,
         temperature: float = 0.3, json_mode: bool = False):
    """调一轮对话，返回 assistant 消息对象（含 .content / .tool_calls）。

    messages 用 OpenAI 格式 [{"role": "system/user/assistant/tool", ...}]。
    json_mode=True 时走 DeepSeek 结构化输出（prompt 里必须含 "json" 字样，D3）。
    """
    kwargs: dict = {"model": MODEL, "messages": messages, "temperature": temperature}
    if tools:
        kwargs["tools"] = tools
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return get_client().chat.completions.create(**kwargs).choices[0].message


def chat_text(messages: list[dict], *, temperature: float = 0.3) -> str:
    """便捷版：只要文本。工具编排别用这个（拿不到 tool_calls）。"""
    return chat(messages, temperature=temperature).content or ""


# ---------------- 自测：python -m app.llm.client ----------------
if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    # 1. 无 key 时的报错路径（不联网，任何人都能跑）
    # （本块在模块层运行，_client 本就是全局名，无需 global 声明）
    saved = os.environ.pop("DEEPSEEK_API_KEY", None)
    try:
        _client = None
        get_client()
        raise AssertionError("缺 key 应该报错")
    except RuntimeError as e:
        print("1 缺 key 报错 OK：", str(e).splitlines()[0])
    finally:
        if saved is not None:
            os.environ["DEEPSEEK_API_KEY"] = saved
        _client = None            # 让下次 get_client 重新走完整流程

    # 2. 有 key 才联网冒烟：一句话验证端点/模型名/key 三件套
    if os.environ.get("DEEPSEEK_API_KEY"):
        reply = chat_text([{"role": "user", "content": "只回复两个字：在的"}],
                          temperature=0.0)
        print("2 联网冒烟:", reply)
    else:
        print("2 联网冒烟: 跳过（.env 里还没有 DEEPSEEK_API_KEY）")
