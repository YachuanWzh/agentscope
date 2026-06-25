# AgentScope 的 Plan 模式（任务规划系统）大白话解读

## 一、这个"Plan 模式"到底是什么？

简单说：**AgentScope 没有在代码里单独做一个叫 Plan Mode 的东西**。它真正做规划的方式，是通过一套**任务管理工具（Task 工具）**来实现的。

这套工具一共四个：`TaskCreate`（创建任务）、`TaskGet`（查看任务详情）、`TaskList`（列出所有任务）、`TaskUpdate`（更新任务状态）。

当 Agent 需要做一件复杂的事情时，它会像人一样，先把大任务拆成小任务，放到一个任务清单里，然后一个一个去执行，执行过程中不停地更新状态。这个任务清单会一直跟着 Agent，哪怕对话中断了再恢复，清单还在。

这个东西在代码里的注释就直接叫它 "Planning tools — always on"（规划工具 — 永远开着）。

---

## 二、为什么这么设计？

### 核心问题：LLM 干着干着就"忘了"自己在干嘛

大模型有一个致命弱点：**上下文窗口是有限的**。当一个任务特别复杂，Agent 来回推理几十轮之后，最早的那些对话记录可能就被挤出了上下文窗口。Agent 就会"失忆"，忘记自己原来要干嘛、做到哪了。

就像你让一个人去装修一间房子，他干了两天之后，忘了当初房主说的是要刷白墙还是蓝墙，忘了厨房的水管已经改过了。

### 设计目标：给 Agent 一个"外部记忆"

所以这套任务系统的核心目标就一个：**把"计划"从 Agent 的脑子里（上下文窗口）搬到外面（状态存储）**。

Agent 推理的时候，这些任务的状态是存在 `AgentState` 里的，不占 LLM 的上下文窗口的 token 额度。只有 Agent 主动调用 `TaskList` 或 `TaskGet` 工具时，才会把当前状态读出来看到。

### 具体为什么这么设计，拆开来看：

**1. 任务清单独立于对话上下文**

对话可能因为上下文窗口满了被压缩（compress_context），但任务清单不会被压缩。任务清单存在 `AgentState.tasks_context` 里，是一个独立的数据结构，每次 Agent 对话结束后被持久化到数据库，下次对话开始时再加载回来。

**2. 任务之间可以设置依赖关系**

现实中的复杂任务，B 必须在 A 完成之后才能开始。比如"先创建数据库表"才能"写数据访问层代码"。这套系统支持用 `blocks` 和 `blocked_by` 来表达这种依赖，Agent 知道哪些任务被卡住了，不能开始做。

**3. 永远打开，不需要手动开关**

这四个任务工具在 `get_toolkit()` 函数里被直接加到工具列表（tool_groups 之外的 "always-on" 工具），Agent 任何时候都能用它们，不需要用户去配置或开启。设计者的态度很明确：规划能力是 Agent 的基本功，不是可选功能。

**4. 给前端看的，不是只给 Agent 看的**

任务清单还会通过 WebSocket 推送到前端 UI，在 `TaskPanel` 组件里渲染成一列小标签，用户能一眼看到 Agent 现在在干嘛、完成到哪了、什么被卡住了。这让用户对 Agent 的工作进度有掌控感。

**5. 上下文压缩时，任务信息会被保留**

当对话太长需要压缩的时候，系统会让模型生成一个结构化的摘要，这个摘要的格式是 `SummarySchema`，里面专门有 `task_overview`（任务概览）、`current_state`（当前进度）、`next_steps`（下一步要做什么）这些字段。也就是说，即使对话历史被压缩了，任务的核心信息还是会被保留下来。

---

## 三、怎么实现的？（核心代码逻辑）

### 3.1 数据模型：Task 长什么样？

```python
class Task(BaseModel):
    subject: str         # 任务标题，比如"修复登录页面的bug"
    description: str     # 详细说明
    state: "pending" | "in_progress" | "completed"  # 当前状态
    id: str              # 任务编号，自动递增 "1", "2", "3"...
    owner: str | None    # 谁在做这个任务
    blocks: list[str]    # 这个任务阻塞了哪些任务（别人要等我）
    blocked_by: list[str] # 这个任务被哪些任务阻塞（我要等别人）
    metadata: dict       # 自定义的附加信息
    created_at: str      # 创建时间
```

任务存在 `TaskContext` 里：

```python
class TaskContext(BaseModel):
    tasks: list[Task] = []  # 就是一个任务列表
```

`TaskContext` 又是 `AgentState` 的一部分：

```python
class AgentState(BaseModel):
    tasks_context: TaskContext = TaskContext()  # 任务上下文
    context: list[Msg] = []                      # 对话上下文
    summary: str = ""                            # 压缩后的摘要
    tool_context: ToolContext = ToolContext()     # 工具上下文（文件缓存等）
    permission_context: PermissionContext = ...  # 权限上下文
    ...
```

### 3.2 四个工具的实现细节

**TaskCreate —— 创建任务**
- 做的事情很简单：从现有任务列表里找到最大的数字 ID，+1 作为新 ID
- 创建一个 `Task` 对象，`state` 默认是 `"pending"`
- 把这个 `Task` 追加到 `_agent_state.tasks_context.tasks` 列表里
- 返回一句 "Task (id=3) created successfully: 修复登录页面的bug"

**TaskList —— 列出所有任务**
- 遍历 `_agent_state.tasks_context.tasks`
- 把每个任务格式化成一行："3 [pending] 修复登录页面的bug [blocked by 1, 2]"
- 如果列表是空的，返回 "No tasks available."

**TaskGet —— 查看某个任务的详情**
- 按 ID 从列表里找到那个任务
- 返回完整信息：标题、状态、描述、owner、被谁阻塞、阻塞了谁

**TaskUpdate —— 更新任务**
- 按 ID 找到任务
- 可以修改：标题、描述、状态、owner、metadata
- 特别的操作：`add_blocks` 和 `add_blocked_by` 会自动维护双向关系（如果 A blocks B，那 B 的 blocked_by 里自动加上 A）
- 状态改成 `"deleted"` 时：真的从列表里删掉，并且清理所有其他任务中对这个 ID 的引用

### 3.3 工具怎么注册到 Agent 身上的？

在 `get_toolkit()` 函数里（`app/_service/_toolkit.py`），有这么一行注释和一行代码：

```python
# Planning tools — always on.
tools += [TaskCreate(), TaskList(), TaskGet(), TaskUpdate()]
```

这四个工具被直接加到基础工具列表里，不属于任何 tool group，所以永远可用，不需要 `ResetTools` 去激活。

### 3.4 工具怎么和 Agent 的状态绑定的？

关键在于 **`is_state_injected = True`**。这四个工具都继承自 `_TaskToolBase`，这个基类设置了：

```python
class _TaskToolBase(ToolBase):
    is_concurrency_safe = True   # 可以并行调用
    is_state_injected = True     # 自动注入 Agent 状态
```

当 `is_state_injected = True` 时，Agent 在执行工具的时候，会自动把 `_agent_state` 参数传进去。工具拿到这个状态对象后，直接在上面增删改查。改完之后，对话服务会把整个 `AgentState` 序列化存回数据库。

### 3.5 任务怎么在对话之间保持？

一条完整的链路是这样的：

```
用户发消息
  → ChatService 从数据库加载 SessionRecord（包含 AgentState + 任务列表）
  → get_toolkit() 组装工具（包含四个 Task 工具）
  → Agent._reply_impl() 进入 ReAct 循环
  → Agent 调用 TaskCreate/TaskUpdate → 直接修改 AgentState.tasks_context
  → 对话结束 → ChatService 把 AgentState 存回数据库
  → 下次对话 → 从数据库加载 → 任务列表又回来了
```

### 3.6 ReAct 循环里任务工具的角色

Agent 的核心工作流是 ReAct（Reasoning + Acting）循环：

1. `_check_next_action()` — 判断下一步该干嘛
   - 有工具调用要执行 → 进入 Acting
   - 没工具调用要执行 → 进入 Reasoning
   - 在等外部事件 → Exit，挂起等待

2. `_reasoning()` — 调用 LLM，让模型思考下一步
   - 如果上下文太长，先调 `compress_context()` 压缩
   - 模型返回文本 or 工具调用

3. `_acting()` — 执行工具调用（TaskCreate、TaskUpdate、Bash、Read 等等）

任务工具在这个循环里跟其他工具（读文件、写文件、跑脚本）是**平起平坐**的。Agent 可能在同一个 Reasoning 步骤里既创建任务又读文件。

### 3.7 上下文压缩时，任务信息怎么保留？

当对话上下文太长了（超过模型输入限制的 80%），系统会触发压缩：

1. 系统把对话历史 + 一段提示词发给一个压缩模型
2. 提示词要求模型按照 `SummarySchema` 的格式输出：
   - `task_overview`：用户到底要我干嘛
   - `current_state`：已经做完了什么
   - `important_discoveries`：过程中发现了什么重要的事
   - `next_steps`：下一步具体要做什么
   - `context_to_preserve`：用户有什么特殊偏好需要记住
3. 压缩模型生成的摘要替代了原来的长对话历史
4. 摘要被放进 `<system-info>` 标签，作为系统信息前置到新的上下文中

这样，即使原始对话被丢掉了，任务的"灵魂"还在。

---

## 四、设计亮点总结

| 设计决策 | 为什么这样 |
|---------|-----------|
| 任务独立于对话上下文 | LLM 上下文会满会被压缩，任务清单不能丢 |
| 永远打开，不需要配置 | 规划是基本能力，不是可选功能 |
| 支持依赖关系 | 现实任务有先后顺序，被阻塞的任务不应该开始 |
| 状态自动注入（is_state_injected） | 工具直接操作 Agent 内部状态，不用绕弯子 |
| 持久化到数据库 | 对话中断恢复后，任务清单还在 |
| 推送到前端 UI | 让用户看得见 Agent 的工作进展 |
| 压缩时保留任务信息 | 上下文被压缩时，任务的核心信息通过结构化摘要保留 |
| 不需要单独的"Plan Mode" | 直接在正常对话流程中穿插规划，不搞特殊模式切换 |

---

## 五、任务是怎么被拆解成小任务的？（核心机制详解）

这是整个规划系统最核心的问题：**Agent 怎么知道要把一个大任务拆成哪几个小任务？拆几个？怎么确定先后顺序？**

答案一句话：**全是 LLM 自己决定的，没有一行代码在做"拆解逻辑"。**

但这不代表"随便拆"。整个拆解过程是由 **工具描述 + 系统提示词 + LLM 推理能力** 三样东西配合完成的。

### 5.1 核心流程：从用户请求到任务列表

```
用户说："帮我给这个项目加上用户登录功能"
  │
  ▼
┌──────────────────────────────────────────────────────┐
│ ① Agent._reply_impl() 进入 ReAct 循环               │
│                                                      │
│ ② _reasoning() 触发                                  │
│    → _prepare_model_input() 组装送给 LLM 的内容：     │
│      - 系统提示词（system_prompt）                    │
│      - 历史对话（context）                            │
│      - 压缩摘要（summary，如果有的话）                │
│      - 工具列表（tools），包含 TaskCreate 的详细描述   │
│                                                      │
│ ③ LLM 读到 TaskCreate 的 description，里面写着：      │
│    "Use this tool proactively:                        │
│     - Complex multi-step tasks (3+ distinct steps)    │
│     - Non-trivial and complex tasks                   │
│     - ..."                                           │
│    LLM 自己判断：登录功能确实复杂，应该拆！            │
│                                                      │
│ ④ LLM 输出多个 ToolCallBlock：                        │
│    - tool_call: TaskCreate("设计数据库用户表")         │
│    - tool_call: TaskCreate("写注册接口")               │
│    - tool_call: TaskCreate("写登录接口")               │
│    - tool_call: TaskCreate("写前端登录页面")           │
│    - tool_call: TaskUpdate(设置依赖关系)               │
│    - tool_call: TaskUpdate(标记任务1为 in_progress)    │
│                                                      │
│ ⑤ _acting() 逐个执行这些工具调用                       │
│    → TaskCreate.call() 把 Task 对象 append 到         │
│      AgentState.tasks_context.tasks 里                │
│    → TaskUpdate.call() 修改状态和依赖关系              │
│                                                      │
│ ⑥ 下一轮 reasoning，LLM 调 TaskList 看看有什么要做    │
│    → 读到：任务1 [in_progress], 任务2-4 [pending]      │
│    → 决定执行任务1：调 Bash 创建数据库迁移文件        │
│    → ...                                             │
└──────────────────────────────────────────────────────┘
```

### 5.2 拆解指令藏在工具描述里

LLM 之所以知道"什么时候该拆任务"，是因为 `TaskCreate` 的描述里写得很清楚。这个描述会作为 tool schema 的一部分，在每次 `_reasoning()` 时随 `_prepare_model_input()` 一起发给 LLM：

```
_prapare_model_input() 发给 LLM 的 tools 参数里包含：

{
  "name": "TaskCreate",
  "description": "Use this tool to create a structured task list for
your current session. This helps you track progress, organize complex
tasks, and demonstrate thoroughness to the user.

## When to Use This Tool

- Complex multi-step tasks (3+ distinct steps)
- Non-trivial and complex tasks that require careful planning
- Plan mode — when using plan mode, create a task list
- User explicitly requests todo list
- User provides multiple tasks (numbered or comma-separated)
- After receiving new instructions — immediately capture requirements

## When NOT to Use

- Only a single, straightforward task
- Trivial task where tracking provides no benefit
- Less than 3 trivial steps
- Purely conversational or informational

## Task Fields
- subject: brief, actionable title in imperative form
- description: what needs to be done

All tasks are created with status 'pending'.
...",
  "parameters": { "subject": "...", "description": "...", "metadata": "..." }
}
```

LLM 看到这段描述后，会自己判断当前请求是否满足"需要拆"的条件。**这本质上是 prompt engineering，不是代码逻辑。**

### 5.3 拆解粒度是怎么控制的？

拆多细、拆几个，完全由 LLM 的理解能力决定，约束来自三个方面：

**① 工具描述里的指导原则**

`TaskCreate` 的描述里暗示了粒度：
- "3 or more distinct steps" → 最少拆 3 个
- "clear, specific subjects" → 名字要具体，不能模糊
- "single, straightforward task → skip" → 太简单的别拆
- "trivial and tracking provides no benefit → skip" → 不值得追踪的别拆

**② 系统提示词的角色**

每个 Agent 创建时都有一个 `system_prompt`（默认为 `"You're a helpful assistant."`），用户可以自己定制。比如可以写：

```
你是一个资深软件工程师。接到复杂任务时，先拆解成可执行的子任务，
每个子任务应该是独立的、可验证的，再逐个执行。
```

这个系统提示词会和工具描述一起发给 LLM，共同影响拆解行为。

**③ LLM 自身的推理能力**

最终用什么粒度、拆成几个、怎么排序，都是 LLM "自己想的"。就像一个人类工程师接到需求后会自己判断"这事大概分 3 步"还是"这事得分 10 步"。不同的模型（GPT-4、Claude、Qwen 等）拆出来的结果可能完全不同。

### 5.4 依赖关系是怎么确定的？

拆完任务后，LLM 还需要确定哪些任务之间有先后依赖。这是通过 `TaskUpdate` 的 `add_blocks` / `add_blocked_by` 参数来设置的：

```
LLM 的推理：先有数据库表 → 才能写接口 → 才能写前端页面

于是输出：
  TaskUpdate(task_id="2", add_blocked_by=["1"])  // 注册接口等表
  TaskUpdate(task_id="3", add_blocked_by=["1"])  // 登录接口等表
  TaskUpdate(task_id="4", add_blocked_by=["2","3"]) // 前端等接口
```

这个依赖关系存在 `Task` 对象的 `blocks` 和 `blocked_by` 字段里，`TaskUpdate._update_block_relation()` 会自动维护双向关系：

```python
# _update_block_relation 的核心逻辑
if task.id == block_id and blocked_by_id not in task.blocks:
    task.blocks.append(blocked_by_id)       # A 阻塞了 B

if task.id == blocked_by_id and block_id not in task.blocked_by:
    task.blocked_by.append(block_id)        # B 被 A 阻塞
```

### 5.5 拆解发生在哪个时刻？

拆解不是一次性完成的。它可以在任何时候发生：

| 时机 | 场景 |
|------|------|
| **对话刚开始** | 用户的需求很复杂，LLM 第一轮 reasoning 就先拆任务 |
| **执行中途** | 做到一半发现某个任务比想象中复杂，再拆一次 |
| **完成一个任务后** | 完成任务后 `TaskUpdate` 返回提示 "Call TaskList to find your next task"，LLM 查看任务列表时可能会发现缺口，补充新任务 |
| **出错后** | 某个任务执行失败，LLM 可能需要创建新的修复任务 |

**TaskCreate 描述里明确写了这个模式：**

```
- After completing a task — Mark it as completed and add any new
  follow-up tasks discovered during implementation
```

这意味着任务列表是**动态增长**的，不是一开始就定死的。像一个活的 TODO 列表。

### 5.6 谁来驱动"下一步做什么"？

任务创建好之后，LLM 怎么知道该做哪一个？有几个机制：

**① TaskUpdate 的提示**

当一个任务被标记为 `completed` 时，`TaskUpdate` 的返回值里会附带一句提示：

```python
if _agent_state.tasks_context.tasks[index].state == "completed":
    res += "\n\nTask completed. Call TaskList now to find your next " \
           "available task or see if your work unblocked others."
```

**② TaskList 的输出排序**

`TaskList` 的描述建议 LLM "Prefer working on tasks in ID order"，因为 ID 是创建顺序，通常前面的任务是基础性的。但这只是建议，LLM 可以根据实际情况调整。

**③ 阻塞机制强制顺序**

如果一个任务被阻塞（`blocked_by` 不为空），LLM 从 `TaskList` 或 `TaskGet` 的输出里能看到 `[blocked by #1, #2]`，自然就不会去开始它。

### 5.7 大白话总结

> **拆任务这件事，不是程序代码在拆，是 LLM 在"想"。程序只提供了四样东西：**
>
> 1. **一支笔**（TaskCreate）—— 让 LLM 能把想法写下来
> 2. **一张纸**（AgentState.tasks_context）—— 写完的东西不会丢
> 3. **一本使用说明书**（工具描述里的 When to Use / When NOT to Use）—— 告诉 LLM 什么时候该用这支笔
> 4. **一个橡皮擦**（TaskUpdate）—— 状态变了可以改
>
> **真正做"项目管理"的那个大脑，是 LLM 自己。**

---

## 六、一个完整的使用例子

假设用户说："帮我给这个项目加上用户登录功能"。

**Agent 的思考过程可能是这样：**

1. 调 LLM 推理 → LLM 读到 TaskCreate 描述，判断"这事需要拆" → 输出 6 个工具调用
2. 调 `TaskCreate` 创建四个任务：
   - 任务1：设计数据库用户表
   - 任务2：写注册接口
   - 任务3：写登录接口
   - 任务4：写前端登录页面
3. 调 `TaskUpdate` 设置依赖关系：任务2、3、4 都 blocked_by 任务1（先有表才能写接口）
4. 调 `TaskUpdate` 把任务1 标记为 `in_progress`
5. 调 Bash 创建数据库迁移文件...
6. 完成后调 `TaskUpdate` 把任务1 标记为 `completed` → 返回值提示 "Call TaskList to find your next task"
7. 调 `TaskList` 看看接下来能做啥 → 任务2 和任务3 的阻塞解除了
8. 继续...

**用户在前端看到的是：** 一个任务面板，实时显示 4 个任务的状态变化，知道 Agent 在干什么。

---

## 七、和 Claude Code 的"Plan Mode"有什么区别？

Claude Code 有一个专门的 `EnterPlanMode` / `ExitPlanMode` 工具对，进入 Plan Mode 后 Agent 只能探索和规划，不能修改代码。这是一种**显式的模式切换**。

AgentScope 没有这种显式切换。它的哲学是：**规划和执行不需要分开，Agent 在同一个 ReAct 循环里随时可以创建任务。** 你不需要先"进入规划模式"，规划完了再"退出规划模式"。Agent 一边规划一边干活，就像一个有经验的人一样。

如果需要"只读探索"的能力，AgentScope 用权限系统（`PermissionMode.EXPLORE`）来实现，由 SubAgent 带着只读权限去探索代码库，而不是切换主 Agent 的模式。

---

## 八、核心文件清单

| 文件 | 作用 |
|------|------|
| `src/agentscope/tool/_task/_create_task.py` | TaskCreate 工具实现 |
| `src/agentscope/tool/_task/_list_task.py` | TaskList 工具实现 |
| `src/agentscope/tool/_task/_get_task.py` | TaskGet 工具实现 |
| `src/agentscope/tool/_task/_update_task.py` | TaskUpdate 工具实现 |
| `src/agentscope/tool/_task/_task_tool_base.py` | 四个工具的共同基类 |
| `src/agentscope/state/_task.py` | Task 数据模型 |
| `src/agentscope/state/_state.py` | AgentState、TaskContext 定义 |
| `src/agentscope/app/_service/_toolkit.py` | 工具组装（Planning tools — always on） |
| `src/agentscope/agent/_agent.py` | Agent 核心 ReAct 循环 |
| `src/agentscope/agent/_config.py` | SummarySchema（上下文压缩时保留任务信息） |
| `examples/web_ui/frontend/src/components/chat/TaskPanel.tsx` | 前端任务面板组件 |
