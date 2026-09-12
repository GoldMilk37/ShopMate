"""第 1 步：加载 data/rag_docs/ 下的全部文档，产出 RawDoc 列表。

你的任务：实现 load_all()。这是整条管线里最简单的一块，适合先写。

要求（对应 01 文档 §3 元数据表、§6 更新规则）：
1. 遍历 data/rag_docs/ 六个子目录，每个文件产出一个 RawDoc
2. doc_id 规则："{doc_type}:{商品ID或文件名}"，如 "product:SKU-10001"、"policy:return_policy"
3. product_id 提取：文件名以 "SKU-" 开头 → 取文件名去扩展名；否则为空字符串
   （faq/guide/policy 类整个文件不属于单件商品）
4. meta 至少包含：category（从 specs/*.json 的 category 字段或文档正文推断，拿不到就空串）、
   brand、updated_at（可先用文件修改时间）、source
5. .json 读成 dict 后展开成可读文本交给 text？不——这里只负责"装进袋子"：
   spec 的 json 原样读入文本（json.dumps 保证中文不转义），chunker 再决定怎么切块
6. 容错：文件读不出来要抛出带文件名的异常，不要静默跳过

写完自测：python -m app.retrieval.loader 应打印 10 件商品的 spec 里
SKU-10001 的 price.current（用来验证你确实读到了内容）。
"""
from pathlib import Path

from .schema import RawDoc, DOC_TYPE_MAP

# 知识库根目录：从本项目根找 data/rag_docs
KB_ROOT = Path(__file__).resolve().parents[2] / "data" / "rag_docs"


def load_all() -> list[RawDoc]:
    """遍历 rag_docs，返回全部文档的 RawDoc 列表。

    TODO: 按上面 docstring 的 1-6 条实现。
    提示：遍历用 KB_ROOT.iterdir() 拿到子目录，目录名查 DOC_TYPE_MAP。
    """
    raise NotImplementedError


def load_one(source: str) -> RawDoc:
    """（进阶，可不做）只加载一个文件，供增量更新用（01 文档 §6）。"""
    raise NotImplementedError
