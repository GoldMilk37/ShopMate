"""工具层第 1 步：加载并校验 data/tools/*.json 的工具定义。已实现。

模块速览：
    load_definitions()          全量加载，返回 {工具名: 定义dict}
    get_schemas()               给 LLM 用的 OpenAI function-calling 格式列表
    is_write_op(name)           是否写操作（executor 的确认门要用）

三个设计决策：
    D1 定义与实现分离：JSON 只声明"是什么"（参数/返回/用途），Python 只写
       "怎么做"。将来接真实业务系统时换实现不动定义，反之亦然。
    D2 写操作清单硬编码在代码里而不是 JSON 里：JSON 里只有 description
       里的自然语言提示（"此工具为写操作"），机器不可靠读；安全级别是
       工程约束，应当由代码唯一裁决（02 文档 §调读原则 2 的落地）。
    D3 定义加载即校验：name 冲突、缺 parameters、required 字段不在
       properties 里，加载时就抛错——坏定义进不了运行时。

对应设计文档 docs/02_tool_definitions.md。
"""
import json
from pathlib import Path

# 工具定义根目录：项目根/data/tools
TOOLS_ROOT = Path(__file__).resolve().parents[2] / "data" / "tools"

# D2：写操作清单（02 文档的"执行"类工具）。其余默认只读。
WRITE_OPS = frozenset({"apply_after_sale", "transfer_to_human"})


def load_definitions() -> dict[str, dict]:
    """读全部 *.json，返回 {工具名: 定义}。定义有硬伤直接抛错（D3）。"""
    defs: dict[str, dict] = {}
    for path in sorted(TOOLS_ROOT.glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        name = d.get("name") or path.stem          # 优先 JSON 里的 name
        if name in defs:
            raise ValueError(f"工具名重复: {name} ({path.name})")
        params = d.get("parameters")
        if not isinstance(params, dict) or params.get("type") != "object":
            raise ValueError(f"{name}: 缺 object 型 parameters")
        props = params.get("properties", {})
        for req in params.get("required", []):
            if req not in props:
                raise ValueError(f"{name}: required 字段 '{req}' 不在 properties 里")
        defs[name] = d
    return defs


def get_schemas() -> list[dict]:
    """LLM 侧的工具列表（OpenAI tools 参数格式）。

    [{"type": "function", "function": {name, description, parameters}}]
    returns 字段是我们自己看的文档，OpenAI 格式不认，剥掉。
    """
    out = []
    for d in load_definitions().values():
        out.append({"type": "function",
                    "function": {"name": d["name"],
                                 "description": d.get("description", ""),
                                 "parameters": d["parameters"]}})
    return out


def is_write_op(name: str) -> bool:
    """D2：写操作判定。未知工具名返回 True——按最坏情况处理，
    宁可多要一次确认，不可漏掉一次确认。"""
    return name not in load_definitions() or name in WRITE_OPS


# ---------------- 自测：python -m app.tools.registry ----------------
if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    defs = load_definitions()
    print(f"共加载 {len(defs)} 个工具:", sorted(defs))

    schemas = get_schemas()
    assert len(schemas) == 6, f"应 6 个工具，实际 {len(schemas)}"
    assert all("returns" not in s["function"] for s in schemas), "returns 字段应剥掉"

    for name in ("query_order", "query_stock", "query_price", "track_logistics"):
        assert not is_write_op(name), f"{name} 应为只读"
    for name in ("apply_after_sale", "transfer_to_human"):
        assert is_write_op(name), f"{name} 应为写操作"
    assert is_write_op("不存在的工具") is True, "未知工具按写操作处理"
    print("校验通过：6 工具 / returns 已剥 / 读写分级正确 / 未知工具偏保守")
