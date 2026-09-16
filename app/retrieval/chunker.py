"""第 2 步：把 RawDoc 切成 Chunk 列表（01 文档 §4 分块策略）。已实现。

模块速览：
    chunk_all(docs)                 入口：批量切块 + 拼锚点
    chunk_document(doc)              单文档切块（按 doc_type 分发）
    _split_by_h2 / _flatten_spec / _split_long / _h1_title / _anchor   内部辅助

四个设计决策（读代码时对着看）：
    D1 父标题只给 policy/faq：政策条款和问答答案单独看会歧义（"特殊品类规则"
       脱离"退换货政策"不知道是谁的规则），块首要拼上文档级标题；
       product/guide/review 的各节天然自含，不拼，靠 D2 的锚点补身份。
    D2 锚点跨文档查询：商品名/品牌不在本文件里（spec 的 json 有 name、
       产品的 Markdown 有品牌），所以 chunk_all 建了两张查找表再统一拼。
    D3 spec 展平：嵌套 dict 递归走，键名带层级前缀（"specs·尺码表·胸围"），
       层级信息对向量检索有用；null 字段跳过（"降噪深度: null" 是误导），
       bool 转成 是/否（"无线充电: True" 不像人话）。
    D4 切长块找换行：在中点 ±150 字符窗口里找最近的换行处下刀，
       避免把句子/表格拦腰切断；窗口里没有换行才硬切。每次递归长度严格
       减半（见 _split_long 注释），保证终止。

管线约定：brand 依赖 loader 填好的 meta（load_all 做了交叉回填）；
单文件喂进来没有 meta.brand 时锚点自动退化为只用商品名。
"""
import json

from .schema import Chunk, RawDoc

# 单块上限（字符）：中文 1 字 ≈ 1.5 token，750 字符 ≈ 512 token（01 文档 §4 通用规则 a）
MAX_CHUNK_CHARS = 750

# 需要拼父标题的文档类型（设计决策 D1）
PARENT_TYPES = {"policy", "faq"}


def chunk_all(docs: list[RawDoc]) -> list[Chunk]:
    """批量入口：对每个文档切块，按商品拼锚点，产出可入库的 Chunk 列表。"""
    # D2：先扫一遍建"商品ID → 商品名/品牌"查找表（name 在 spec json 里）
    name_by_pid: dict[str, str] = {}
    brand_by_pid: dict[str, str] = {}
    for d in docs:
        if d.doc_type == "spec":
            data = json.loads(d.text)
            name_by_pid[d.product_id] = str(data.get("name", ""))
        if d.product_id and d.meta.get("brand"):
            brand_by_pid.setdefault(d.product_id, d.meta["brand"])

    chunks: list[Chunk] = []
    for d in docs:
        pieces = chunk_document(d)          # 切块（含超长对半，通用规则 a/c 在里面）
        for i, text in enumerate(pieces, start=1):
            anchor = _anchor(d.product_id, name_by_pid, brand_by_pid)  # 通用规则 b
            cid = f"{d.doc_id}#{i:02d}"
            chunks.append(Chunk(
                doc_id=d.doc_id,
                chunk_id=cid,
                text=f"{anchor}{text}" if anchor else text,
                # chunk_id 也进 meta：LangChain 版检索器（lc_retriever）从
                # Document.metadata 里取 ID，Chroma 的 ids 字段它拿不到
                meta={**d.meta, "doc_type": d.doc_type, "product_id": d.product_id,
                      "collection": d.collection, "chunk_id": cid},   # indexer 按 meta["collection"] 分组入库
            ))
    return chunks


def chunk_document(doc: RawDoc) -> list[str]:
    """单文档切块，返回块正文列表（还没拼锚点，锚点是全库层面的事）。"""
    if doc.doc_type == "spec":
        return _split_long(_flatten_spec(doc.text))

    parent = _h1_title(doc.text) if doc.doc_type in PARENT_TYPES else ""
    out: list[str] = []
    for title, body in _split_by_h2(doc.text):
        block = "\n".join(part for part in (parent, title, body) if part)
        out.extend(_split_long(block))
    return out


# ---------------- 内部辅助 ----------------

def _split_by_h2(text: str) -> list[tuple[str, str]]:
    """按 ## 切。第一个 ## 之前的行（H1 标题 + 空行）不算块，H1 的处理见 D1。

    返回 [(标题行, 正文)]，空正文的节直接丢弃（通用规则 c 的一半）。
    """
    sections: list[tuple[str, str]] = []
    title: str | None = None
    buf: list[str] = []

    for line in text.splitlines():
        if line.startswith("## "):
            if title is not None:
                sections.append((title, "\n".join(buf).strip()))
            title, buf = line.strip(), []
        elif title is not None:
            buf.append(line)
    if title is not None:
        sections.append((title, "\n".join(buf).strip()))

    return [(t, b) for t, b in sections if b]


def _h1_title(text: str) -> str:
    """取第一个 # 开头行的内容（不带 #），找不到返回空串。"""
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _flatten_spec(text: str) -> str:
    """spec 的 JSON → "键: 值" 文本行（D3）。

    嵌套 dict 键名带层级前缀："specs·尺码表·胸围(cm): {...}"；
    list 用「、」连接；bool 转是/否；null 跳过。
    """
    data = json.loads(text)
    lines: list[str] = []

    def walk(obj: dict, prefix: str) -> None:
        for k, v in obj.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                walk(v, f"{key}·")
            elif isinstance(v, bool):               # 必须在 int 前判：bool 是 int 的子类
                lines.append(f"{key}: {'是' if v else '否'}")
            elif v is None:                          # "降噪深度: null" 是误导，跳过
                continue
            elif isinstance(v, list):
                lines.append(f"{key}: {'、'.join(str(x) for x in v)}")
            else:
                lines.append(f"{key}: {v}")

    walk(data, "")
    return "\n".join(lines)


def _split_long(text: str, max_len: int = MAX_CHUNK_CHARS) -> list[str]:
    """超长块递归对半（D4）。返回的每块都 ≤ max_len 且非空。

    终止性：len > max_len 时，左半 ≤ mid+150、右半 ≤ len-mid+150，两者都
    严格小于 len（max_len ≥ 300 即可保证），所以每层长度严格递减。
    """
    text = text.strip()
    if not text:                                    # 通用规则 c：空块丢弃
        return []
    if len(text) <= max_len:
        return [text]

    mid = len(text) // 2
    lo, hi = max(0, mid - 150), min(len(text), mid + 150)
    window, target = text[lo:hi], mid - lo          # target = 窗口里对应原文中点的下标
    best, best_dist = -1, 1 << 30
    idx = window.find("\n")
    while idx != -1:                                # 找最接近中点的换行（D4）
        if abs(idx - target) < best_dist:
            best, best_dist = idx, abs(idx - target)
        idx = window.find("\n", idx + 1)
    cut = (lo + best) if best != -1 else mid        # 窗口内没有换行才硬切

    return _split_long(text[:cut], max_len) + _split_long(text[cut:], max_len)


def _anchor(pid: str, names: dict[str, str], brands: dict[str, str]) -> str:
    """商品锚点："【SKU-10001 无线降噪蓝牙耳机 Pro | SoundCore】\\n"（D2，通用规则 b）。

    商品ID也拼进去：评价/评价类文档的 H1（含 SKU 号）不入块，用户
    报型号提问时（"SKU-10001 口碑怎么样"）BM25 要靠锚点里的 ID 才
    能定位到那件商品的文档——评测发现的漏召回就是这么来的。
    非商品文档（policy/guide/faq）product_id 为空，锚点为空串，
    它们靠 D1 的父标题保上下文。
    """
    if not pid:
        return ""
    name = names.get(pid, "")
    if not name:
        return ""
    brand = brands.get(pid, "")
    head = f"{pid} {name}" + (f" | {brand}" if brand else "")
    return f"【{head}】\n"


# ---------------- 自测：python -m app.retrieval.chunker ----------------
if __name__ == "__main__":
    import sys

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    from .loader import load_all

    docs = load_all()
    chunks = chunk_all(docs)

    # 1. 规模与分布
    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c.meta["doc_type"]] = by_type.get(c.meta["doc_type"], 0) + 1
    print(f"共 {len(chunks)} 个 chunk:", by_type)

    # 2. 硬校验（不过就抛错，别静默）
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_id 有重复"
    assert all(c.text.strip() for c in chunks), "有空块"
    assert all(len(c.text) <= MAX_CHUNK_CHARS + 40 for c in chunks), \
        f"有块超过上限：{max(len(c.text) for c in chunks)} 字符"
    print("校验通过：chunk_id 唯一 / 无空块 / 无超长块（上限+锚点宽限 40）")

    # 3. 抽样肉眼看三种形态（D1/D2/D3 各验一个）
    print("\n--- 样本1：商品块（应带 D2 锚点）---")
    s = next(c for c in chunks if c.doc_id == "product:SKU-10001" and "核心卖点" in c.text)
    print(s.chunk_id, "|", "\n".join(s.text.splitlines()[:3]))

    print("\n--- 样本2：spec 展平块（应见 D3 层级前缀与 是/否）---")
    s = next(c for c in chunks if c.doc_id == "spec:SKU-10001")
    print(s.chunk_id, "|", "\n".join(s.text.splitlines()[:4]))

    print("\n--- 样本3：policy 块（应带 D1 父标题）---")
    s = next(c for c in chunks if c.doc_id == "policy:return_policy" and "特殊品类" in c.text)
    print(s.chunk_id, "|", "\n".join(s.text.splitlines()[:2]))

    print("\n--- 样本4：review 块（_reviews 文档的商品身份靠锚点找回）---")
    s = next(c for c in chunks if c.doc_id == "review:SKU-40002" and "回南天" in c.text)
    print(s.chunk_id, "|", "\n".join(s.text.splitlines()[:2]))
