"""第 3 步：把 Chunk 向量化并写入 ChromaDB。

这块我来写（涉及模型加载和 ChromaDB API，第一次接触面太大）——
你写完 loader/chunker 后，用本文件跑一次入库，再进第 4 步。
目前先放着，TODO 只为了占位。
"""
from .schema import Chunk, EMBED_DIM


def embed_texts(texts: list[str]) -> list[list[float]]:
    """BGE-M3 批量向量化。"""
    raise NotImplementedError


def build_index(chunks: list[Chunk], persist_dir: str = "data/chroma") -> None:
    """全量入库/覆盖式重建（01 文档 §6：按 doc_id 先删后插）。"""
    raise NotImplementedError
