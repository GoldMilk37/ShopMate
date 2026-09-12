"""第 1 步：加载 data/rag_docs/ 下的全部文档，产出 RawDoc 列表。（已实现）

模块速览：
    load_all()                     全库加载，日常用这个
    load_one("specs/SKU-10001.json") 单文件加载，留给以后的增量更新

内部辅助（带下划线的函数都是"只给本文件用"的）：
    _make_raw_doc()    把一个文件变成一个 RawDoc
    _fill_category_brand()  第二遍扫描，补齐每件商品的 category/brand

对应设计文档 docs/01_rag_knowledge_base.md §3 元数据、§6 更新规则。
"""
import json
import time
from pathlib import Path

from .schema import RawDoc, DOC_TYPE_MAP

# 知识库根目录：从本文件往上退两级（app/retrieval/ → ShopMate/），再拼 data/rag_docs
KB_ROOT = Path(__file__).resolve().parents[2] / "data" / "rag_docs"


def load_all() -> list[RawDoc]:
    """遍历 rag_docs 六个子目录，返回全部文档的 RawDoc 列表。"""
    docs: list[RawDoc] = []
    for folder in sorted(KB_ROOT.iterdir()):          # sorted 是为了让每次加载顺序稳定
        if not folder.is_dir() or folder.name not in DOC_TYPE_MAP:
            continue                                   # 跳过临时目录 / .DS_Store 之类
        doc_type, collection = DOC_TYPE_MAP[folder.name]
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            docs.append(_make_raw_doc(path, doc_type, collection))
    _fill_category_brand(docs)
    return docs


def load_one(source: str) -> RawDoc:
    """只加载一个文件（给增量更新用）。source 是相对 rag_docs 的路径。

    注意：category/brand 需要交叉查同商品的 spec/product 文档才能补齐，
    单文件加载时这两个字段留空——增量更新场景够用了。
    """
    path = KB_ROOT / source
    if not path.is_file():
        raise FileNotFoundError(f"知识库中没有这个文件: {source}")
    folder = path.parent.name
    if folder not in DOC_TYPE_MAP:
        raise ValueError(f"不认识的目录（不在 DOC_TYPE_MAP 里）: {folder}")
    doc_type, collection = DOC_TYPE_MAP[folder]
    return _make_raw_doc(path, doc_type, collection)


# ---------------- 内部辅助 ----------------

def _make_raw_doc(path: Path, doc_type: str, collection: str) -> RawDoc:
    """一个文件 → 一个 RawDoc。读文件失败时带着文件名报错，绝不静默跳过。"""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise RuntimeError(f"读取失败: {path} ({e})") from e

    stem = path.stem          # Stem: "SKU-10001" / "SKU-10001_reviews" / "return_policy"
    product_id = stem.removesuffix("_reviews") if stem.startswith("SKU-") else ""
    doc_id = f"{doc_type}:{product_id or stem}"   # e.g. "review:SKU-10001" / "policy:return_policy"
    updated_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))

    return RawDoc(
        doc_id=doc_id,
        doc_type=doc_type,
        collection=collection,
        source=path.relative_to(KB_ROOT).as_posix(),   # "specs/SKU-10001.json"，统一正斜杠
        product_id=product_id,
        text=text,          # spec 的 json 原文照传——展平是 chunker 的事
        meta={
            "source": path.relative_to(KB_ROOT).as_posix(),
            "updated_at": updated_at,
        },
    )


def _fill_category_brand(docs: list[RawDoc]) -> None:
    """第二遍扫描，就地补齐 category / brand（docstring 要求第 4 条）。

    数据在哪就查哪：category 只有 spec 的 json 里有；品牌只有 products 的
    Markdown 里有。所以先建两张查找表，再回填到所有文档。
    """
    spec_by_pid: dict[str, dict] = {}
    brand_by_pid: dict[str, str] = {}
    for d in docs:
        if d.doc_type == "spec":
            spec_by_pid[d.product_id] = json.loads(d.text)
        elif d.doc_type == "product":
            for line in d.text.splitlines():
                if line.startswith("- 品牌："):
                    brand_by_pid[d.product_id] = line.split("：", 1)[1].strip()
                    break
    for d in docs:
        spec = spec_by_pid.get(d.product_id)
        d.meta["category"] = spec["category"] if spec else ""
        d.meta["brand"] = brand_by_pid.get(d.product_id, "")


# ---------------- 自测：python -m app.retrieval.loader ----------------
if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，中文会乱码
    docs = load_all()

    count_by_type: dict[str, int] = {}
    for d in docs:
        count_by_type[d.doc_type] = count_by_type.get(d.doc_type, 0) + 1
    print(f"共加载 {len(docs)} 个文档:", count_by_type)

    spec = next(d for d in docs if d.doc_id == "spec:SKU-10001")
    data = json.loads(spec.text)
    print("SKU-10001 当前价:", data["price"]["current"], "元  (应得 599)")
    print("SKU-10001 meta:", spec.meta, " (category=数码/耳机/蓝牙耳机, brand=SoundCore)")

    # 验证 product_id 从带 _reviews 后缀的文件名里也提取对了
    rev = next(d for d in docs if d.doc_id == "review:SKU-10001")
    print("评价文档的 product_id:", rev.product_id, " (应得 SKU-10001，不带 _reviews)")
