"""演示前端：Streamlit 人机对话 + **内部状态可视化**。已实现。

用法：
    streamlit run app/agent/webui.py        # 浏览器打开
    python -m app.agent.webui               # 普通解释器下会提示怎么启动
    python -m app.agent.webui --e2e         # 无头端到端自测（AppTest，离线）

模块速览：
    main()          全部渲染
    _e2e_check()    无头端到端：同进程跑脚本，桩掉检索与 LLM，全程离线
    _render_*       侧栏/轨迹面板的分块

七个设计决策：
    D1 引导 sys.path + 绝对导入：`streamlit run` 下脚本以伪造的 `__main__` 模块
       执行（script_runner.py:672 `types.ModuleType("__main__")`），**没有
       __package__，相对导入直接 ImportError**。所以先把仓库根塞进 sys.path，
       再走 `from app.agent...` 绝对导入。顺带一个副作用要知道：脚本所在目录
       （app/agent/）也会被塞进 sys.path 最前面，所以本目录下的模块名不能和
       标准库重名——`trace_view.py` 就是这么来的（叫 trace.py 会遮蔽标准库）。
    D2 `if __name__ == "__main__"` 在 `streamlit run` 下**成立**：同一个伪造
       模块的名字就叫 `__main__`（源码第 672/679 行已核实；那句"不能用
       __name__ == '__main__'"的注释说的是 Streamlit 自己的模块）。所以本文件
       可以安全 import——渲染全在 main() 里，import 不产生任何 st 调用。
       AppTest 也靠这一点才能驱动它。
    D3 不套 @st.cache_resource：agent 和 store 是模块级单例，import 一次就
       一直在（Streamlit 重跑的是**脚本**，不是已导入的模块）。套上缓存反而
       危险——缓存未命中会给你一个全新的 Agent + 空 SessionStore，表现是
       "刷新一下对话就没了"，而且不报错。
    D4 对话记录渲染 st.session_state["messages"]，**不是** Session.history：
       后者被 append_round 截断到 10 轮（那是喂给 LLM 的窗口，不是给用户看的
       记录）。长演示下照 history 渲染会把自己的滚动记录吃掉。history 的
       轮数单独显示，正好把"记忆窗口"这个设计讲清楚。
    D5 侧栏是重点：意图、置信度、槽位、召回块（含是否被引用）、工具调用、
       命中的兜底。这是本项目区别于"套壳 ChatGPT"的地方——面试官点得开、
       看得见写操作确认门和工具编排。
    D6 "结束会话"按钮：st.session_state 是每个浏览器连接一份，刷新页面会换新
       sid 并留下一个永不回收的旧 Session（进程内 dict，无 TTL）。演示规模不
       值得引入过期策略，给个手动回收的口子。
    D7 首次检索很慢：实测第一轮 12943ms（BGE-M3 加载），之后 527ms。所以
       加 spinner 文案 + "预热"按钮，别让面试官以为卡死了。
"""
import random
import sys
import time
from pathlib import Path

# D1：必须在 import app.* 之前
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st                                        # noqa: E402

from app.agent import trace_view                              # noqa: E402
from app.agent.graph import agent                             # noqa: E402
from app.agent.session import store                           # noqa: E402

SPINNER = "小搭正在思考……（首次检索要加载向量模型，约十几秒）"


def _in_streamlit() -> bool:
    """判断是否跑在 Streamlit 运行时里（普通解释器下是 False）。"""
    try:
        from streamlit.runtime import exists
    except ImportError:          # 没装 streamlit 也不该炸
        return False
    return exists()


def _get_sid() -> str:
    """D6：一个浏览器连接一个会话 id，进程内的 Session 靠它隔离。"""
    if "sid" not in st.session_state:
        st.session_state["sid"] = (f"s-{time.strftime('%H%M%S')}"
                                   f"-{random.randint(100, 999)}")
    return st.session_state["sid"]


def _project(rows: list[dict], columns: tuple) -> list[dict]:
    """按固定列序投影——顺便挡掉意外字段，表格不会因为多一个键就变形。"""
    return [{c: r.get(c) for c in columns} for r in rows]


def _render_trace(trace: dict) -> None:
    """D5：把本轮轨迹画出来。D4（trace_view）保证半成品轨迹也不会抛。"""
    st.subheader("本轮内部状态")
    if not trace:
        st.caption("还没有轨迹——先问一句试试。")
        return

    summ = trace_view.summarize(trace)
    # 先摆最能讲故事的四项，其余折叠
    st.markdown(f"**分支**：{summ['分支']}")
    st.markdown(f"**意图**：{summ['意图']}　**置信度**：{summ['置信度']}")

    fb = trace_view.fallback_label(trace)
    if fb:
        st.warning(f"触发兜底：{fb}")
    if trace.get("gate"):
        st.info(f"确认门判定：{trace['gate']}（没让 LLM 猜，零成本）")

    with st.expander("全部摘要", expanded=False):
        for k, v in summ.items():
            st.markdown(f"- **{k}**：{v}")

    if trace.get("slots"):
        with st.expander("槽位", expanded=False):
            st.json(trace["slots"])

    rows = trace_view.hit_rows(trace)
    with st.expander(f"召回块（{len(rows)}）", expanded=bool(rows)):
        if rows:
            st.dataframe(_project(rows, trace_view.HIT_COLUMNS),
                         hide_index=True, width="stretch")
            st.caption("**被引用**一列是**代理指标**（字符 4-gram 重叠率，已扣掉"
                       "本轮样板文字），会漏掉同义改写，不等于 hit@5。")
        else:
            st.caption("本轮没有召回任何块。")

    rows = trace_view.tool_rows(trace)
    with st.expander(f"工具调用（{len(rows)}）", expanded=bool(rows)):
        if rows:
            st.dataframe(_project(rows, trace_view.TOOL_COLUMNS),
                         hide_index=True, width="stretch")
        else:
            st.caption("本轮没调工具。")


def _render_sidebar() -> None:
    sid = _get_sid()
    s = store.get(sid)

    st.markdown("### 会话")
    st.caption(f"`{sid}`　记忆窗口 {len(s.history) // 2} 轮（上限 10）")
    if s.current_product_id:
        st.caption(f"话题商品锚点：`{s.current_product_id}`")

    if s.pending_write:                                      # D1 确认门可见
        st.warning(f"**有一笔待确认的写操作**\n\n`{s.pending_write['name']}`\n\n"
                   "请回复「确认」或「取消」。")

    st.divider()
    _render_trace(st.session_state.get("last_trace") or s.trace)

    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("结束会话", width="stretch"):
        store.drop(sid)
        st.session_state.pop("sid", None)
        st.session_state["messages"] = []
        st.session_state.pop("last_trace", None)
        st.rerun()
    if col2.button("预热模型", width="stretch"):
        from app.retrieval.indexer import get_model
        with st.spinner("加载 BGE-M3…"):
            get_model()
        st.success("向量模型就绪，可以开始问了。")


def main() -> None:
    st.set_page_config(page_title="ShopMate 智能客服", page_icon="🛍️",
                       layout="wide")
    st.session_state.setdefault("messages", [])
    sid = _get_sid()

    with st.sidebar:
        _render_sidebar()

    st.title("🛍️ ShopMate 智能客服「小搭」")
    st.caption("RAG 混合检索 + Function Calling + 自研状态机。左栏是本轮的内部"
               "状态：意图、检索、工具调用、兜底，全都看得见。")

    # D4：渲染自己的记录，不是 Session.history
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    text = st.chat_input("问点什么…（商品咨询 / 口碑 / 对比 / 推荐 / 订单 / 售后）")
    if not text:
        return

    st.session_state["messages"].append({"role": "user", "content": text})
    with st.chat_message("user"):
        st.markdown(text)

    with st.chat_message("assistant"):
        try:
            with st.spinner(SPINNER):                        # D7
                reply = agent.handle(sid, text)
        except Exception as e:
            # 走到这里说明状态机自己炸了（不是它内部兜住的那几种）。把本轮
            # 已经走到的轨迹一并显示出来——"走到哪一步炸的"比错误信息有用。
            st.error(f"内部错误：{type(e).__name__}: {e}")
            st.session_state["last_trace"] = store.get(sid).trace
            raise
        st.markdown(reply)

    st.session_state["messages"].append({"role": "assistant", "content": reply})
    st.session_state["last_trace"] = store.get(sid).trace
    st.rerun()          # 让左栏立刻反映本轮（否则它显示的还是上一轮）


# ---------------- 无头端到端：python -m app.agent.webui --e2e ----------------
def _e2e_check() -> None:
    """同进程跑整个脚本，桩掉检索与 LLM。

    为什么不用 curl 探测代替：`GET /` 返回 200 只说明**服务器起来了**，脚本要
    等 websocket 连上才执行——探测不到应用本身是否正常。AppTest 是真的执行
    脚本（__name__ 同样是 "__main__"，见 D2），且与应用同进程，所以这里打的桩
    能直接生效，全程离线、确定、秒级。
    """
    from streamlit.testing.v1 import AppTest

    import app.agent.graph as g
    import app.llm.client as llm

    anchor = "【SKU-10001 无线降噪蓝牙耳机 Pro | SoundCore】"
    hits = [
        {"chunk_id": "spec:SKU-10001#01", "text": anchor + "续航时间约 30 小时。",
         "score": 0.031, "meta": {"doc_type": "spec", "product_id": "SKU-10001",
                                  "collection": "product_knowledge"}},
    ]
    reply = "SKU-10001 的续航时间约 30 小时。"

    class _FakeResult:
        """工具执行的返回值替身——确认门那轮要是真调工具，会往 data/ 里写单子。"""
        status, ok, message = "ok", True, "已创建售后单"
        data = {"after_sale_id": "AS-0001", "next_step": "客服将在 24 小时内联系您"}

    saved = (g.classify, g.search_with_product_focus, g.rag_search, llm.chat_text,
             g.tool_execute)
    g.classify = lambda text, history=None: g.IntentResult(
        "product_consult", 0.95, False, {"product_id": "SKU-10001"})
    g.search_with_product_focus = lambda *a, **k: hits
    g.rag_search = lambda *a, **k: hits
    llm.chat_text = lambda *a, **k: reply
    g.tool_execute = lambda *a, **k: _FakeResult()
    try:
        at = AppTest.from_file(str(Path(__file__).resolve()), default_timeout=120)
        at.run()
        assert not at.exception, at.exception
        assert at.chat_input, "页面上应该有输入框"

        # ---- 第一轮：RAG 支路 ----
        at.chat_input[0].set_value("SKU-10001 续航多久").run()
        assert not at.exception, at.exception

        msgs = at.session_state["messages"]
        assert len(msgs) == 2, f"一问一答应有两条记录，实际 {len(msgs)}: {msgs}"
        assert msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant"
        assert reply in msgs[1]["content"], msgs[1]

        tr = at.session_state["last_trace"]
        assert tr["stage"] == "rag", tr
        assert tr["collection"] == "product_knowledge", tr
        assert tr["intent"] == "product_consult" and tr["confidence"] == 0.95, tr
        assert tr["cited"] == [True], f"回复逐字用了那块，应判为被引用: {tr['cited']}"
        assert "elapsed_ms" in tr

        assert len(at.chat_message) == 2, f"应有两个聊天气泡，实际 {len(at.chat_message)}"
        heads = " ".join(h.value for h in at.sidebar.subheader)
        body = " ".join(m.value for m in at.sidebar.markdown)
        assert "本轮内部状态" in heads, f"侧栏轨迹面板没渲染: {heads}"
        assert "**分支**：RAG 检索" in body, f"侧栏没显示分支名: {body[:200]}"
        assert "**判定被引用**：1" in body, f"摘要没反映引用判定: {body[:400]}"

        # ---- 第二轮：确认门走通 + 工具面板（"套壳 ChatGPT"做不出来的那部分）----
        # 直接把会话推进到"有待确认写操作"的状态，省掉伪造一次 tool_calls 协议。
        sid = at.session_state["sid"]
        store.get(sid).pending_write = {
            "name": "create_after_sale", "arguments": {"order_id": "SO-2024-0001"}}

        at.chat_input[0].set_value("确认").run()
        assert not at.exception, at.exception

        tr = at.session_state["last_trace"]
        assert tr["stage"] == "confirmation" and tr["gate"] == "accept", tr
        assert [r["name"] for r in tr["tool_calls"]] == ["create_after_sale"], tr
        assert store.get(sid).pending_write is None, "确认后应清空"
        body = " ".join(m.value for m in at.sidebar.markdown)
        assert "**工具调用**：1" in body, f"摘要没反映工具调用: {body[:400]}"
        assert len(at.sidebar.dataframe) == 1, \
            f"工具表应有 1 张（这轮没有召回块），实际 {len(at.sidebar.dataframe)}"

        # ---- 第三轮：听不懂 → 不兑现、也不丢弃，警告条常驻 ----
        # 这条只能放在**没被 rerun 冲掉**的那一轮上验：待确认提示是在脚本开头
        # 画的，而确认成功那轮结尾会 rerun，重画时 pending_write 已经清空了。
        store.get(sid).pending_write = {
            "name": "create_after_sale", "arguments": {"order_id": "SO-2024-0001"}}

        at.chat_input[0].set_value("这个…什么意思").run()
        assert not at.exception, at.exception

        tr = at.session_state["last_trace"]
        assert tr["stage"] == "confirmation" and tr["gate"] == "unclear", tr
        assert store.get(sid).pending_write is not None, "没听清就不该动这笔写操作"
        warns = " ".join(w.value for w in at.sidebar.warning)
        assert "待确认的写操作" in warns, f"侧栏没提示待确认写操作: {warns}"
    finally:
        (g.classify, g.search_with_product_focus, g.rag_search, llm.chat_text,
         g.tool_execute) = saved

    print("e2e 通过：脚本在 __main__ 下真的执行了；RAG 轮（轨迹/引用判定/召回表）"
          "与确认门三轮（accept 后清空、unclear 后保留并常驻提示）都在侧栏渲染出来")


# ---------------- 自测：python -m app.agent.webui（不联网、不需要 streamlit 运行时） ----
if __name__ == "__main__":
    if _in_streamlit():
        main()                       # D2：streamlit run 走这条
    elif "--e2e" in sys.argv:
        _e2e_check()
    else:
        print("这是 Streamlit 应用，不是命令行程序。启动方式：")
        print("    streamlit run app/agent/webui.py")
        print("离线自测：")
        print("    python -m app.agent.webui --e2e       # 无头端到端（不联网）")
        print("    python -m app.agent.trace_view        # 轨迹渲染的纯自测")
        print("想要终端对话用：python -m app.agent.cli")
