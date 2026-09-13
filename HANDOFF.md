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

**Git 已收尾**（2026-09-13）：`main` = `origin/main` = `77f8492`，全部推送完毕；
历史里 4 条 `goo` 占位消息**决定保留不改**。详见 §5。

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

## 5. Git 现状（已收尾）

### 当前状态

```
main      77f8492 [origin/main]   ← 与远端一致，已全部推送
backup-before-cleanup  85ae90b    ← 整理前的安全备份，不要删
```

工作区干净。2026-09-13 走的是**路线 A：不改写历史，直接 fast-forward 推**：

```
85ae90b..77f8492  main -> main
```

已在远端的 5 个 commit：

```
77f8492 docs:  增加交接文档，记录项目现状与 Git 未完成事项
7cb8d5b chore: 索引产物与本地工作目录移出版本库
05230df fix:   requirements.txt 转 UTF-8，pip install -r 才可读
396edbe docs:  重写 README 并回写评测结论
7f6477d feat(agent): Agent 层话题商品锚点过滤，检索 hit@5 96%→100%
```

### 关于 4 条 `goo` 消息：决定保留

历史顶部还有 4 条占位消息，**决定不改写了**——仓库不对外展示，改名的收益抵不上
rebase 在这台机器上的风险（见下）。它们实际对应的内容：

```
85ae90b goooooooo   → 实际是：整个 Agent 层 + tools 层 + llm 层 + eval
e191c1a goooooo     → 实际是：retriever 混合检索 + RRF 融合
4155381 goooo       → 实际是：检索管线微调 + docs/04 数据规范
fb89231 gooo        → 实际是：loader/chunker/indexer 完成 + README 首版
c2df46b 框架起立
0b5163c first commit
```

万一以后要改，映射表在这里（省得重新推导）：

| 原 | 新 |
|---|---|
| `gooo` | `feat(retrieval): 完成 loader/chunker/indexer 检索管线` |
| `goooo` | `fix(retrieval): 修正检索管线细节并补充 docs/04 数据规范` |
| `goooooo` | `feat(retrieval): 实现向量+BM25 混合检索与 RRF 融合` |
| `goooooooo` | `feat(agent): 实现工具层、Agent 状态机、LLM 客户端与检索评测` |

### ⚠️ 真要走改写历史，先读这段

当时试过 `git rebase -i`，**失败了三次**，原因不是命令写错：

1. **todo 行格式变了**：新版 git 是 `pick <sha> # <message>`（sha 与消息之间有 `#`），
   老写法 `<verb> <sha> <message>` 匹配不上 —— 结果一个 reword 都没改成，rebase 空跑一遍。
2. **根因（可 100% 复现）**：这台机器 `core.autocrlf=true`，`git checkout` 切到旧 commit
   （如 `c2df46b`）时 **`app/retrieval/__init__.py` 不会被写回磁盘**。rebase 一开始检测到
   unstaged deletion 就中止：`error: cannot rebase: You have unstaged changes`。
3. 每次中止都留下 rebase 中间态，传染给下一次尝试。恢复办法：
   `git checkout -- app/retrieval/__init__.py`。

**绕开办法：别用 rebase。** 用 `git reset --soft backup-before-cleanup` 再按逻辑单元重新提交——
这条路完全不动工作区，不触发上面第 2 条。

### push 不上去时先看这里

直连 github.com 会被 reset，报错长这样：

```
fatal: unable to access 'https://github.com/...': schannel: failed to receive handshake, SSL/TLS connection failed
```

**这个报错完全看不出是代理问题**——本机 `http.proxy` / `https.proxy` 都是空的，
git 默认直连。要走 Clash：

```bash
git -c http.proxy=http://127.0.0.1:7890 -c https.proxy=http://127.0.0.1:7890 push origin main
```

先确认 Clash 活着（返回 200 才往下走）：

```bash
curl -x http://127.0.0.1:7890 -o /dev/null -w "%{http_code}\n" https://github.com
```

（想一劳永逸：`git config http.proxy http://127.0.0.1:7890`，本仓库尚未配置。）

### 验证清单

```bash
ls app/retrieval/__init__.py                       # 必须在
python -m app.retrieval.loader                     # 必须 PASS
git status --short                                 # 必须干净
```

---

## 6. 建议的下一步顺序

1. **P1 评价域路由** —— 这是当前收益最高的功能缺口，做完整个链路才算闭环
2. **P2 三个缺陷**（`_transfer` 写历史、graph LLM 兜底、缓存 key 含 user_id）——
   加起来不到半小时，直接影响"能不能演示得住"
3. MySQL / Redis 接入 → 多用户支持
