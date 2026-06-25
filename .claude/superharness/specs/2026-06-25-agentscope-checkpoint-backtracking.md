# AgentScope 可回溯能力（Checkpointing / Time-Travel）设计

**日期:** 2026-06-25
**状态:** 设计已批准（待实现）
**参考:** LangGraph checkpointer / time-travel

## 1. 目标

为 AgentScope 框架引入“可回溯能力”，参考 LangGraph 的 checkpoint 机制：能把
agent 运行过程中的状态按步落盘，并支持回到任意历史落点重新执行。用
`session_id`（即 LangGraph 的 `thread_id`）唯一绑定一条会话的回溯历史。

“线性回溯”是对外的默认表现；底层是支持 fork 的分叉树，线性只是树上从 head 沿
`parent` 链回溯到 root 的默认遍历顺序。

## 2. 已定决策

| # | 决策 | 取值 |
|---|---|---|
| 1 | 实现层级 | 分层：核心层定义 `Checkpointer` 抽象，App 层提供 Redis 实现 |
| 2 | 回溯粒度 | 每个 ReAct 迭代（= LangGraph super-step / `cur_iter`） |
| 3 | 回退语义 | 完整时间旅行：`parent_checkpoint_id` 构成 DAG 树，支持 fork |
| 4 | checkpoint 内容 | `AgentState` 全量快照 + 完整消息快照（自包含） |
| 5 | 对外接口 | 核心 Python API（仿 LangGraph），本期不做 HTTP / Agent 工具 |
| 6 | 留存策略 | 可配置（上限 N / TTL），默认全保留 |
| 7 | 自动落点 | 传入 `checkpointer` 即每迭代自动 `put` |
| 8 | 本期范围 | 仅核心层；App `RedisCheckpointer` 与消息表一致性为后续阶段 |

## 3. 概念映射（对齐 LangGraph）

| LangGraph | AgentScope |
|---|---|
| `thread_id` | `session_id`（复用现有） |
| `checkpoint_id` | 新增，每个落点唯一 id |
| `parent_checkpoint_id` | 构成回溯树 DAG |
| super-step | 一次 ReAct 迭代（`cur_iter`） |
| channel state | `AgentState` 全量快照 |
| `get_state` / `get_state_history` / time-travel | `agent.get_state` / `get_state_history` / `rewind` |

## 4. 现状（落地基础）

- `AgentState`（`src/agentscope/state/_state.py`）= 每会话可变运行态：
  `session_id`、`summary`、`context: list[Msg]`、`reply_id`、`cur_iter`、
  `permission_context`、`tool_context`、`tasks_context`、`middle_context`。
- `SessionRecord` 持有 `state: AgentState`，由 `StorageBase`（Redis 实现）持久化；
  `update_session_state(...)` 当前**直接覆盖单槽**，无历史。
- 消息单独持久化：`upsert_message` / `list_messages`。
- `ChatService._run_impl` 是热路径：每 turn 跑完 `agent.reply_stream` 后落 state，
  全程在 session 分布式锁内。

## 5. 设计（本期：核心层）

### 5.1 新模块 `src/agentscope/checkpoint/`

**`Checkpoint`**（pydantic `BaseModel`）：

```text
checkpoint_id: str               # _generate_id()
thread_id: str                   # = session_id
parent_checkpoint_id: str | None # 构成 DAG；root 为 None
step: int                        # 对应 cur_iter / super-step 序号
created_at: float
source: Literal["iteration", "fork", "manual"]
state: AgentState                # 全量快照
messages: list[Msg]              # 完整消息快照（自包含，fork 分支天然隔离）
metadata: dict                   # 触发信息等
```

**`CheckpointerBase`（ABC）** — 仿 `BaseCheckpointSaver`：

```text
async put(checkpoint) -> str                  # 存储，返回 checkpoint_id
async get(thread_id, checkpoint_id) -> Checkpoint | None
async get_latest(thread_id) -> Checkpoint | None   # 当前 head
async list(thread_id, *, limit=None, before=None) -> list[Checkpoint]
                                              # = get_state_history，默认沿主线
async delete_thread(thread_id) -> None
```

- 留存策略由 `RetentionPolicy(max_per_thread: int | None = None,
  ttl_seconds: int | None = None)` 配置，默认两者皆 `None`（全保留），在 `put`
  内执行裁剪。
- 内置 **`MemoryCheckpointer`**（进程内实现，供裸用 SDK / 测试，呼应已有的
  in-memory message bus）。

### 5.2 Agent 集成

- `Agent.__init__` 增加可选参数 `checkpointer: CheckpointerBase | None = None`。
- `AgentState` 增加 `checkpoint_id: str | None` 字段（当前 head 指针），随会话
  持久化，使恢复时知道“站在树的哪个节点”。
- ReAct 循环每个迭代结束（`cur_iter` 自增处）：若 `checkpointer` 存在，构造一个
  `Checkpoint`（`parent = state.checkpoint_id`，快照当前 `AgentState` + 消息），
  调 `put`，并把返回的新 id 写回 `state.checkpoint_id`。

### 5.3 时间旅行 API（Agent 上的核心 Python API）

```text
await agent.get_state() -> Checkpoint            # 当前落点
await agent.get_state_history() -> list[Checkpoint]
                                                 # 主线历史 = 线性回溯视图
await agent.rewind(checkpoint_id) -> None        # 装回该点 state+messages，head 指向它
```

- `rewind` 后再 `reply`：新 checkpoint 的 `parent` = 该点 → **自动 fork 出子分支**，
  旧分支保留。这正是 LangGraph 的时间旅行语义。
- 线性回溯 = `get_state_history` 默认沿 `parent` 链从 head 回到 root，呈现单条主线。

## 6. 后续阶段（不在本期）

- **App `RedisCheckpointer(CheckpointerBase)`**：key `checkpoint:{thread}:{cpid}`，
  每 thread 一个有序索引 + head 指针；`ChatService` 装配 Agent 时注入；写入在现有
  session 分布式锁内完成；`RetentionPolicy` 从部署配置注入。
- **App 消息表一致性**：rewind/fork 时以 `checkpoint.messages` 为权威，重置或旁路
  现有 `list_messages` 单线追加流。

## 7. 风险

| 风险 | 缓解 |
|---|---|
| 每迭代全量快照 → 存储膨胀 | retention 配置；后续可内容寻址去重 |
| 大 `context` 每迭代序列化 → 写放大 | 接受为本期代价；后续可增量/压缩 |
| App 消息表与 checkpoint 快照一致性 | 后续阶段以 `checkpoint.messages` 为权威重置 |

## 8. 验收（核心层）

- `MemoryCheckpointer` 可 put/get/get_latest/list/delete_thread，且 `list` 默认
  返回从 head 到 root 的主线。
- 带 `checkpointer` 的 Agent 跑一轮多迭代 reply 后，能 `get_state_history` 看到每
  迭代一个 checkpoint，`parent` 链正确。
- `rewind` 到中间某点后再 `reply`，生成的新 checkpoint `parent` 指向该点，旧分支
  仍可 `get` 到（fork 验证）。
- retention `max_per_thread=N` 时，主线超过 N 丢最旧。

> 全程遵循 superharness：严格 TDD（先写失败测试）、系统化调试、完成前用真实命令
> 输出验证。实现请运行 `/superharness:go 实现核心层 checkpoint 能力`。
