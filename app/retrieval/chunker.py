"""第 2 步：把 RawDoc 切成 Chunk 列表（01 文档 §4 分块策略）。

这块比 loader 难一点，核心是"按文档类型选切法"。

六种切法（01 文档 §4 表格的代码化）：
  product/review → 按 ## 标题切；父标题拼进块首
  spec(json)     → 展平成 "键: 值" 文本行；整文件一个 chunk（当前商品 <512 token）
  faq            → 一问一答一个 chunk（按 ## Q 标题切）
  policy         → 按 ## 切，且一级标题（# 开头那行）拼进每个块
  guide          → 按 ## 切

通用规则（所有类型都做）：
  a. 单块超 512 token → 对半切（token 数按 len(约)/1.5 估即可，不必装 tokenizer）
  b. 块首拼接商品锚点："【{商品名} | {品牌}】正文..."（01 文档 §4）
  c. 空块丢弃

你的任务：实现 chunk_document(doc: RawDoc) -> list[Chunk]，
并把 _chunk_by_h2、_flatten_spec、_split_long 补完。
"""
from .schema import Chunk, RawDoc


def chunk_all(docs: list[RawDoc]) -> list[Chunk]:
    """入口：对每个 RawDoc 调 chunk_document，汇总所有 Chunk。"""
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d))
    return out


def chunk_document(doc: RawDoc) -> list[Chunk]:
    """按 doc_type 分发到具体切法，再套通用规则 a/b/c。

    建议：先拿到初步块列表，再统一过 _attach_anchor 和 _split_long。
    """
    raise NotImplementedError


def _chunk_by_h2(text: str) -> list[str]:
    """按 ## 二级标题切（# 不算切点）。返回 [(标题, 正文)] 也行，返回纯正文也行——内部函数，自己定。"""
    raise NotImplementedError


def _flatten_spec(text: str) -> str:
    """spec 的 json 文本 → "键: 值" 每行一条的可读文本，便于向量化。"""
    raise NotImplementedError


def _split_long(text: str, max_len: int = 750) -> list[str]:
    """超长块对半切。max_len 用字符数近似 token（中文 1 字≈1.5 token 的宽松版）。"""
    raise NotImplementedError
