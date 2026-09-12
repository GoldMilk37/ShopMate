"""检索管线的公共约定（常量与数据结构）。

对应设计文档 docs/01_rag_knowledge_base.md 第 2、3 节。
改这里不用动其他模块——所有"魔法数字/字符串"集中在这。
"""
from dataclasses import dataclass, field


# ---- 知识域映射：源目录 → (doc_type, 入库 Collection) —— 01 文档 §2 ----
# doc_type 用于元数据过滤；collection 决定文档进哪个 ChromaDB 库
DOC_TYPE_MAP = {
    "products": ("product", "product_knowledge"),
    "specs":    ("spec",    "product_knowledge"),
    "guides":   ("guide",   "product_knowledge"),
    "faq":      ("faq",     "product_knowledge"),
    "policies": ("policy",  "policy_knowledge"),
    "reviews":  ("review",  "review_knowledge"),
}

# ---- 检索参数 —— 01 文档 §5 ----
TOP_K = 5                # 每路召回条数
RRF_K = 60               # RRF 融合常数：rank 越靠后贡献越小，60 是业内常用值
SCORE_THRESHOLD = 0.02   # 融合分兜底阈值：低于它回复"暂无相关信息"

# BGE-M3 输出维度（ChromaDB 建 collection 时用）
EMBED_DIM = 1024


@dataclass
class RawDoc:
    """loader 的输出：一个文件的原始内容 + 元数据。

    field 的意思是：dataclass 里每个属性叫一个"字段"。
    """
    doc_id: str          # 文档唯一ID，如 "spec:SKU-10001"
    doc_type: str        # product/spec/faq/policy/review/guide
    collection: str      # 要入库到哪个 collection
    source: str          # 相对路径，如 "specs/SKU-10001.json"
    product_id: str      # 关联商品ID；policy 等非商品类留空字符串
    text: str            # 文件全文（chunker 的输入）
    meta: dict = field(default_factory=dict)  # 其余元数据（brand/category/updated_at...）


@dataclass
class Chunk:
    """chunker 的输出：一块可被向量化的文本。

    text 头部应已拼好商品名锚点（01 文档 §4 通用规则）。
    """
    doc_id: str          # 所属文档，回溯用
    chunk_id: str        # 全局唯一，建议 f"{doc_id}#{序号}"
    text: str            # 块正文
    meta: dict = field(default_factory=dict)   # 继承 RawDoc.meta，可加块级字段

    def to_embedding_text(self) -> str:
        """返回送去向量化的文本。默认就是 text；将来想加前缀重写这里。"""
        return self.text
