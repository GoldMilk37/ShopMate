# ShopMate 交接文档

> 写给接手的人（或未来的自己）。目标是不用翻聊天记录就能把项目跑起来、知道坑在哪、
> 知道 Git 现在停在什么状态。文档时间：2026-09-13。
>
> 配套阅读：`README.md`（项目概览与设计要点）、`docs/01~04`（四篇设计文档，
> **所有设计决策的「为什么」都写在那里**，代码注释里也有 D1/D2/D3 形式的决策编号）。

---

## 0. 一句话状态

三层（RAG 检索 / 工具层 / Agent 状态机）**都已跑通**，50 条评测集 hit@5 达到 100%，
`python -m app.agent.cli` 可以直接人机对话。

**但 Git 历史整理做了一半就停了**：新改动的 4 个 commit 已按要求命名，**旧的 4 个
`goo` 消息还没改**，且本地比远端领先 4 个 commit **尚未推送**。详见 §5。

---

## 1. 环境怎么起

```bash
cd E:/Learning-Agent/05-projects/ShopMate
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt        # 已转 UTF-8，之前是 UTF-16 导致 pip 读不了
echo DEEPSEEK_API_KEY=sk-xxxxx > .env  # .env 不入库
```

**必须用 `.venv/Scripts/python.exe` 跑，不要用系统 Python**（系统 Python 缺 jieba 等依赖）。
本机 venv 已装好，直接用即可。

### 日常命令

| 目的 | 命令 |
|---|---|
| 离线自测（不联网，5 组） | `python -m app.retrieval.loader` / `.chunker` / `app.tools.registry` / `app.tools.executor` / `app.agent.session` |
| 建库 | `python -m app.retrieval.indexer`（首次拉 BGE-M3，之后离线） |
| 单条检索 | `python -m app.retrieval.retriever "通勤降噪耳机推荐"` |
| 检索评测 | `python -m app.retrieval.eval`（四路对照，约 1 分钟） |
| 意图识别用例 | `python -m app.agent.intent`（**需 key**） |
| 状态机冒烟 | `python -m app.agent.graph`（**需 key**） |
| 人机对话 | `python -m app.agent.cli`（`/new` 换会话 `/history` 看记忆 `/exit` 退出） |

**clone 之后务必先跑 indexer**——`data/chroma/` 已加入 `.gitignore`，索引不再入库。

---

## 2. 代码地图

```
app/
├── retrieval/   加载 → 分块 → 入库 → 混合召回 → 评测
│   ├── schema.py    常量与数据结构（6 类文档 → 3 collection）
│   ├── loader.py    读 rag_docs 产出 RawDoc（商品名→SKU 目录也在这读）
│   ├── chunker.py   按文档类型六种切法 → 124 chunk
│   ├── indexer.py   BGE-M3 向量化 + ChromaDB 持久化
│   ├── retriever.py 向量+BM25 → RRF 融合 → 阈值 → 元数据过滤
│   └── eval.py      50 条评测集，四路 hit@5 对照
├── tools/       Function Calling
│   ├── registry.py  加载 JSON 工具定义，读写分级（WRITE_OPS）
│   ├── mock.py      模拟业务实现（订单/物流/价格/库存/售后）
│   └── executor.py  执行编排：确认门 / 超时 / 缓存 / 异常收敛
├── agent/       状态机
│   ├── intent.py    7 类意图 + 置信度 + product_id 槽位
│   ├── graph.py     状态机主体：确认门 / 四条兜底 / 工具编排上限 3 轮
│   ├── session.py   会话记忆（进程内存，接口按 Redis 设计）
│   └── cli.py       交互入口
└── llm/client.py    DeepSeek 封装（OpenAI 兼容，支持 json_mode）
```

关键设计决策的出处：

- **检索阈值、doc_type 加权、词法置信门的调参史与理由** → `docs/01_rag_knowledge_base.md` §七，
  以及 `retriever.py` 顶部的 D1~D8 docstring
- **元数据过滤为什么不做独占式** → `docs/01` §三「坑」
- **工具读写分级、确认门** → `docs/02`
- **状态机、置信度兜底、上下文管理** → `docs/03`
- **MySQL/Redis 表结构与缓存键（尚未接入）** → `docs/04`

---

## 3. 实测数据（别信形容词，信这个）

| 检索方式 | hit@5 | 未命中 |
|---|---|---|
| 纯向量 | 90% | 5 |
| 纯 BM25 | 94% | 3 |
| RRF 混合 | 96% | 2 |
| 混合 + 话题商品锚点 | **100%** | 0 |

规模：32 文档 / 3 品类 / **8 个 SKU**（不是 10）/ 6 个工具。
写操作集合：`{apply_after_sale, transfer_to_human}`。

调参时有三个反直觉结论，都在 `docs/01` §七，改参数前先读：
阈值 0.02→0.012、`doc_type` 加权只留 spec 1.3、**带 `where` 时词法门放宽到 1 个实词
（无过滤路径门槛不动）**。

---

## 4. 已知问题清单（按优先级）

### P1 — 影响「能不能演示得住」

1. **评价域没接进路由**：`RAG_MODES` 三个意图全指向 `product_knowledge`，
   `review_knowledge` 库已建好但没人查。"XX 口碑怎么样"目前拿商品详情作答。
   ⚠️ 这条没做完的话，上面 100% 的检索成果在真实链路里吃不到一半。
   两种补法待定：新增第 8 个意图（干净，但动意图集）/ 在 `product_consult` 内
   按"口碑/缺点/评价"二次判定（不动意图集，但塞了规则）。

### P2 — 代码缺陷，都已定位

2. `graph._transfer()` **没写会话历史**：转人工那轮在 `/history` 看不到，
   摘要 `history[-6:]` 也会缺用户最后一句。
3. `graph.py` 里多处 LLM 调用**没有 try/except**（只有 `intent.classify` 有兜底），
   网络抖动会中断会话。
4. `executor` 只读缓存的 key **只哈希了 `arguments`，没含 `user_id`**，
   加了"我的订单列表"这类按用户维度的只读工具会串户。这也是多用户支持的前置。

### P3 — 未完成项（用户在 TODO 里列过）

5. MySQL / Redis 接入（现为 SQLite + 进程内存会话）
6. 多用户支持：工具层硬编码 `user_id="u1001"`，分布在
   `app/agent/graph.py`（201/224/258/276 行）和 `app/tools/mock.py`
7. FastAPI 服务 + Streamlit 演示前端（未开始）
8. `requirements.txt` 含 langchain / langgraph 等**当前代码并未引用**的依赖，
   是早期探索留下的，待裁剪

---

## 5. Git 现状（接手必读）

### 当前状态

```
main      7cb8d5b [origin/main: ahead 4]   ← 领先远端 4 个 commit，尚未推送
backup-before-cleanup  85ae90b             ← 整理前的安全备份，不要删
```

工作区干净。已提交的 4 个新 commit（消息已按要求写好）：

```
7cb8d5b chore: 索引产物与本地工作目录移出版本库
05230df fix:   requirements.txt 转 UTF-8，pip install -r 才可读
396edbe docs:  重写 README 并回写评测结论
7f6477d feat(agent): Agent 层话题商品锚点过滤，检索 hit@5 96%→100%
```

历史最上面 4 个仍是占位消息（`first commit` / `框架起立` 是有含义的，保留）：

```
85ae90b goooooooo   → 实际是：整个 Agent 层 + tools 层 + llm 层 + eval
e191c1a goooooo     → 实际是：retriever 混合检索 + RRF 融合
4155381 goooo       → 实际是：检索管线微调 + docs/04 数据规范
fb89231 gooo        → 实际是：loader/chunker/indexer 完成 + README 首版
c2df46b 框架起立
0b5163c first commit
```

拟改成的名字（写在这里，省得重新推导）：

| 原 | 新 |
|---|---|
| `gooo` | `feat(retrieval): 完成 loader/chunker/indexer 检索管线` |
| `goooo` | `fix(retrieval): 修正检索管线细节并补充 docs/04 数据规范` |
| `goooooo` | `feat(retrieval): 实现向量+BM25 混合检索与 RRF 融合` |
| `goooooooo` | `feat(agent): 实现工具层、Agent 状态机、LLM 客户端与检索评测` |

### ⚠️ 为什么停在半路（重要，别重复踩）

尝试用 `git rebase -i` 改写这 4 条消息，**失败了三次**，原因不是命令写错：

1. **第一次踩的坑**：新版 git 的 rebase todo 行格式是 `pick <sha> # <message>`
   （sha 和消息之间有 `#`），老写法 `<verb> <sha> <message>` 匹配不上，
   结果 todo 里一个 reword 都没改成，rebase 空跑一遍。
2. **根因**：这台机器上 `git checkout` 切到旧 commit（如 `c2df46b`）时，
   **`app/retrieval/__init__.py` 不会被写回磁盘**（该文件只有两行注释，
   `core.autocrlf=true`）。已单独复现过一次，可 100% 重现。
   于是 rebase 一开始检测到 unstaged deletion 就中止：
   `error: cannot rebase: You have unstaged changes`。
3. 每次中止都留下了 rebase 中间态，传染给下一次尝试。

已解决的部分：文件已用 `git checkout -- app/retrieval/__init__.py` 恢复，
五个离线自测复跑全 PASS，工作区干净。

### 继续整理的三条路线（任选，按推荐度排序）

**A. 不改写历史，直接推（最省事）**
```bash
git push origin main          # fast-forward，不需要 force
```
代价：远端历史里那 4 个 `goo` 消息留着。仓库是自己用的话无实质影响。

**B. 先 `autocrlf=false` 再 rebase（想改干净的话）**
```bash
git config core.autocrlf false
git reset --hard backup-before-cleanup   # 或直接在干净工作区上
# 再 rebase，todo 解析要用 `pick <sha> # <message>` 的格式
git push --force-with-lease origin main
```
改完记得确认 `app/retrieval/__init__.py` 还在。**操作前先做本地副本备份。**

**C. 不管了，直接 force push 现状**：不建议，历史没整理干净还付了 force 的代价。

### 无论走哪条，验证清单

```bash
ls app/retrieval/__init__.py                       # 必须在
python -m app.retrieval.loader                     # 必须 PASS
git status --short                                 # 必须干净
```

---

## 6. 建议的下一步顺序

1. **先决定 Git 走哪条路线**（§5），这一步不复杂但会卡住所有人
2. **P1 评价域路由** —— 这是当前收益最高的功能缺口，做完整个链路才算闭环
3. **P2 三个缺陷**（`_transfer` 写历史、graph LLM 兜底、缓存 key 含 user_id）——
   加起来不到半小时，直接影响"能不能演示得住"
4. MySQL / Redis 接入 → 多用户支持
