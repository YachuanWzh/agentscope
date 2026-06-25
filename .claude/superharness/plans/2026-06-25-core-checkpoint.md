# Plan — 核心层 checkpoint 能力

**Spec:** `.claude/superharness/specs/2026-06-25-agentscope-checkpoint-backtracking.md`
**硬约束:** 纯增量/可选。`checkpointer=None`（默认）时 Agent 行为与现状**逐字节一致**；
不改任何现有测试预期。
**测试命令:** `PYTHONPATH=src python -m pytest <path> -q`（site-packages 版过期，必须用本地 src）

## 设计要点（落定）
- 新模块 `src/agentscope/checkpoint/`：`Checkpoint`、`RetentionPolicy`、
  `CheckpointerBase`(ABC)、`MemoryCheckpointer`。
- `Checkpoint`: checkpoint_id / thread_id(=session_id) / parent_checkpoint_id /
  step / created_at / source / state:AgentState / messages:list[Msg] / metadata。
  快照需**深拷贝隔离**（put 时 deep copy，免后续 mutation 污染）。
- `MemoryCheckpointer`: 进程内 dict 存储 + 每 thread head 指针；`list` 沿 parent 链
  从 from_checkpoint_id（默认 head）回溯到 root，newest-first；retention 裁剪。
- `AgentState` 增加 `checkpoint_id: str | None = None`（head 指针，可选字段）。
- `Agent.__init__` 增加 `checkpointer: CheckpointerBase | None = None`。
- ReAct 循环 `cur_iter += 1` 后：若 checkpointer 存在，深拷贝当前 state put 一个
  子 checkpoint（parent=state.checkpoint_id），新 id 写回 state.checkpoint_id。
- 时间旅行 API：`get_state()` / `get_state_history()` / `rewind(checkpoint_id)`。
  rewind 把 checkpoint.state 装回 agent.state（深拷贝），并设 state.checkpoint_id。
  rewind 后再 reply → 新 checkpoint parent=该点 → 自动 fork。

## 任务（每个走 RED→GREEN→REFACTOR→commit）

1. **Checkpoint + RetentionPolicy 模型** — 字段、默认、深拷贝快照不共享引用。
   测试 `tests/checkpoint_model_test.py`。
2. **CheckpointerBase + MemoryCheckpointer** — put/get/get_latest/list/delete_thread；
   list 默认沿 head→root 主线；retention max_per_thread/ttl。
   测试 `tests/checkpoint_memory_test.py`。
3. **AgentState.checkpoint_id 字段** — 可选，默认 None，旧记录反序列化不破。
   测试 `tests/checkpoint_agent_state_test.py`（或并入 agent 测试）。
4. **Agent 集成（自动落点）** — checkpointer 参数；多迭代 reply 后每迭代一个
   checkpoint，parent 链正确；checkpointer=None 时零行为变化（回归断言）。
   测试 `tests/checkpoint_agent_test.py`。
5. **时间旅行 API** — get_state / get_state_history / rewind；rewind→reply fork 验证。
   并入 `tests/checkpoint_agent_test.py`。
6. **导出** — `checkpoint/__init__.py` + 顶层 `agentscope/__init__` 暴露。

## 验证
- 新增测试全绿。
- `agent_basic_test.py` + `compress_context_test.py` 等 agent 相关全绿（回归）。
- 全量 `tests/` 跑通（对比基线）。
