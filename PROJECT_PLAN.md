# AI Mock Interview Agent — 完整开发规划

## 一、项目定位

**一句话描述**：基于 Multi-Agent 架构的智能面试模拟系统，输入 JD 即可获得一场自适应、可评分的 AI 模拟面试。

**核心卖点（面试时怎么讲）**：
- Multi-Agent 协作：4 个专职 Agent 各司其职，通过 LangGraph 编排
- 自适应追问：根据回答质量动态决定是否深挖、换题、调整难度
- 结构化评估：基于 Rubric 的评分体系，输出可量化的面试报告
- 工程化交付：WebSocket 实时对话 + Docker 一键部署 + 前端可视化

---

## 二、完整技术栈

| 层级 | 技术选型 | 选型理由 |
|------|---------|---------|
| **语言** | Python 3.11+ | 生态最好，Agent 框架首选 |
| **Web 框架** | FastAPI | 异步支持好，WebSocket 原生支持，自动生成 API 文档 |
| **Agent 编排** | LangGraph 0.2+ | 支持有状态的循环图，条件分支，适合面试的多轮循环场景 |
| **LLM 接入** | OpenAI SDK（兼容 DeepSeek/Qwen 等） | 统一接口，通过 base_url 切换模型提供商 |
| **向量数据库** | ChromaDB | 轻量级，本地部署无依赖，够用 |
| **数据校验** | Pydantic v2 | Structured Output 的基础，LLM 输出强类型化 |
| **实时通信** | WebSocket（FastAPI 原生） | 面试对话的实时双向通信 |
| **前端界面** | Gradio | 比 Streamlit 更适合对话场景，自带聊天组件 |
| **容器化** | Docker + Docker Compose | 一键部署，环境隔离 |
| **测试** | pytest + pytest-asyncio | 异步测试支持 |
| **文档** | MkDocs（可选） | 自动生成项目文档站 |

---

## 三、Multi-Agent 架构设计

### 3.1 四个 Agent 的职责

```
┌─────────────────────────────────────────────────────────┐
│                    LangGraph Orchestrator                │
│                                                         │
│  ┌──────────────┐    ┌──────────────────┐               │
│  │ JD Analyst   │───>│ Question Planner │               │
│  │ Agent        │    │ Agent            │               │
│  └──────────────┘    └────────┬─────────┘               │
│                               │                         │
│                               v                         │
│                      ┌────────────────┐                 │
│                      │  Interviewer   │<──┐             │
│                      │  Agent         │───┘ (循环)      │
│                      └────────┬───────┘                 │
│                               │ (面试结束)              │
│                               v                         │
│                      ┌────────────────┐                 │
│                      │  Evaluator     │                 │
│                      │  Agent         │                 │
│                      └────────────────┘                 │
└─────────────────────────────────────────────────────────┘
```

**Agent 1 — JD Analyst（JD 分析师）**
- 输入：原始 JD 文本
- 输出：结构化技能矩阵（Pydantic Model）
  - 必须技能 vs 加分技能
  - 技能分类（编程语言 / 框架 / 系统设计 / 软技能）
  - 每个技能的预估权重
- 技术点：Structured Output（function calling / JSON mode）

**Agent 2 — Question Planner（出题策划师）**
- 输入：技能矩阵 + 面试题库（RAG 检索）
- 输出：面试题单（5-8 题），包含：
  - 题目内容、考察技能点、难度等级、参考答案要点、追问方向
- 技术点：RAG 检索 + Rerank + LLM 生成混合出题

**Agent 3 — Interviewer（面试官）**
- 输入：题单 + 候选人实时回答
- 行为：
  - 按题单提问
  - 根据回答质量实时决策：追问 / 给提示 / 下一题 / 结束
  - 控制面试节奏（时间、题量）
- 技术点：LangGraph 条件边 + 状态机循环

**Agent 4 — Evaluator（评估师）**
- 输入：完整面试对话记录
- 输出：结构化评分报告
  - 各技能维度评分（1-10）
  - 总体评级（A/B/C/D）
  - 优势与改进建议
  - 关键回答的逐条点评
- 技术点：Scoring Rubric + Structured Output + Few-shot

### 3.2 LangGraph 状态图

```
                    ┌──────────┐
                    │  START   │
                    └────┬─────┘
                         │
                         v
                ┌────────────────┐
                │  analyze_jd    │  (JD Analyst Agent)
                └────────┬───────┘
                         │
                         v
                ┌────────────────┐
                │  plan_questions│  (Question Planner Agent)
                └────────┬───────┘
                         │
                         v
                ┌────────────────┐
            ┌──>│  ask_question  │  (Interviewer Agent)
            │   └────────┬───────┘
            │            │
            │            v
            │   ┌────────────────┐
            │   │  wait_answer   │  <── 用户通过 WebSocket 回答
            │   └────────┬───────┘
            │            │
            │            v
            │   ┌────────────────┐
            │   │ assess_answer  │  (Interviewer Agent 评估单题)
            │   └────────┬───────┘
            │            │
            │            v
            │   ┌────────────────┐
            │   │   route_next   │  ── 条件分支 ──┐
            │   └────────────────┘               │
            │       │        │                   │
            │  [追问] │   [下一题]              [结束面试]
            │       │        │                   │
            └───────┘        └───────┐           │
                                     │           │
                             ┌───────┘           │
                             │                   │
                             └──>  ┌─────────────v──┐
                                   │  evaluate_all  │  (Evaluator Agent)
                                   └────────┬───────┘
                                            │
                                            v
                                   ┌────────────────┐
                                   │  generate_report│
                                   └────────┬───────┘
                                            │
                                            v
                                      ┌──────────┐
                                      │   END    │
                                      └──────────┘
```

### 3.3 核心状态定义（InterviewState）

```python
from typing import TypedDict, Literal
from pydantic import BaseModel

class SkillItem(BaseModel):
    name: str
    category: Literal["language", "framework", "system_design", "soft_skill", "domain"]
    weight: float  # 0.0 - 1.0
    is_required: bool

class SkillMatrix(BaseModel):
    position_title: str
    skills: list[SkillItem]
    experience_level: Literal["intern", "junior", "mid", "senior"]

class QuestionItem(BaseModel):
    id: int
    content: str
    skill_tags: list[str]
    difficulty: Literal["easy", "medium", "hard"]
    reference_points: list[str]  # 参考答案要点
    follow_up_directions: list[str]  # 可追问方向

class AnswerAssessment(BaseModel):
    question_id: int
    score: int  # 1-10
    covered_points: list[str]
    missed_points: list[str]
    should_follow_up: bool
    follow_up_reason: str | None

class InterviewState(TypedDict):
    jd_text: str
    skill_matrix: SkillMatrix | None
    question_plan: list[QuestionItem]
    current_question_index: int
    follow_up_count: int  # 当前题的追问次数
    max_follow_ups: int   # 每题最多追问次数
    conversation_history: list[dict]  # 完整对话记录
    assessments: list[AnswerAssessment]
    interview_complete: bool
    final_report: dict | None
```

---

## 四、项目目录结构

```
ai-mock-interview/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI 入口
│   ├── config.py                  # 配置管理（LLM keys, model names）
│   │
│   ├── agents/                    # Multi-Agent 核心
│   │   ├── __init__.py
│   │   ├── state.py               # InterviewState 定义
│   │   ├── graph.py               # LangGraph 图编排（核心！）
│   │   ├── jd_analyst.py          # JD 分析 Agent
│   │   ├── question_planner.py    # 出题策划 Agent
│   │   ├── interviewer.py         # 面试官 Agent
│   │   └── evaluator.py           # 评估 Agent
│   │
│   ├── rag/                       # RAG 检索模块
│   │   ├── __init__.py
│   │   ├── vector_store.py        # ChromaDB 封装
│   │   ├── embeddings.py          # Embedding 服务
│   │   ├── retriever.py           # 混合检索（向量 + BM25）
│   │   └── reranker.py            # 重排序
│   │
│   ├── llm/                       # LLM 服务层
│   │   ├── __init__.py
│   │   ├── client.py              # OpenAI 兼容客户端封装
│   │   └── prompts.py             # 所有 Prompt 模板集中管理
│   │
│   ├── models/                    # Pydantic 数据模型
│   │   ├── __init__.py
│   │   ├── jd.py                  # JD 相关模型
│   │   ├── question.py            # 题目相关模型
│   │   ├── interview.py           # 面试会话模型
│   │   └── report.py              # 评估报告模型
│   │
│   ├── api/                       # API 路由
│   │   ├── __init__.py
│   │   ├── router.py              # 路由注册
│   │   ├── interview_ws.py        # WebSocket 面试端点
│   │   └── interview_rest.py      # REST 端点（上传 JD、获取报告等）
│   │
│   └── services/                  # 业务服务层
│       ├── __init__.py
│       ├── session_manager.py     # 面试会话管理
│       └── report_generator.py    # 报告生成与格式化
│
├── data/
│   ├── question_bank/             # 面试题库（JSON/Markdown）
│   │   ├── python.json
│   │   ├── system_design.json
│   │   ├── machine_learning.json
│   │   └── ...
│   └── sample_jds/                # 示例 JD（用于测试和演示）
│       ├── ai_engineer.txt
│       ├── backend_developer.txt
│       └── ...
│
├── frontend/
│   └── gradio_app.py              # Gradio 前端界面
│
├── scripts/
│   ├── init_vector_store.py       # 初始化向量数据库（导入题库）
│   └── evaluate.py                # 评估脚本（跑测试用例验证评分一致性）
│
├── tests/
│   ├── test_jd_analyst.py
│   ├── test_question_planner.py
│   ├── test_interviewer.py
│   ├── test_evaluator.py
│   └── test_integration.py        # 端到端集成测试
│
├── docs/
│   ├── architecture.md            # 架构说明文档
│   └── images/                    # 架构图等
│
├── .env.example                   # 环境变量模板
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml                 # 项目配置 + 依赖管理
├── requirements.txt               # pip 依赖
└── README.md
```

---

## 五、分阶段开发计划（共 3 周）

### Phase 1：地基搭建（第 1 周前半，Day 1-3）

**目标**：项目能跑起来，LLM 能调通，基础架构就位。

| 任务 | 产出 | 预计耗时 |
|------|------|---------|
| 1.1 初始化项目结构 | 目录 + pyproject.toml + .gitignore + git init | 1h |
| 1.2 配置管理 | config.py 用 pydantic-settings 读 .env | 1h |
| 1.3 LLM 客户端封装 | 统一的 async 调用接口，支持普通/流式/structured output | 3h |
| 1.4 FastAPI 骨架 | main.py + 健康检查 + CORS + 路由挂载 | 1h |
| 1.5 Pydantic 模型定义 | 所有核心数据模型 | 2h |
| 1.6 Docker 配置 | Dockerfile + docker-compose.yml 能跑通 | 2h |

**验收标准**：`docker compose up` 能启动，`/health` 返回 200，LLM 调用能正常返回。

---

### Phase 2：RAG 题库系统（第 1 周后半，Day 4-6）

**目标**：面试题库入库，能根据技能检索出相关题目。

| 任务 | 产出 | 预计耗时 |
|------|------|---------|
| 2.1 收集面试题库 | 至少 3 个方向（Python/系统设计/ML），每个 20+ 题 | 3h |
| 2.2 题库数据格式设计 | JSON schema：题目/标签/难度/参考答案/追问方向 | 1h |
| 2.3 ChromaDB 封装 | vector_store.py：增删改查 + collection 管理 | 2h |
| 2.4 Embedding 服务 | 支持 OpenAI embedding 或本地模型 | 1h |
| 2.5 混合检索实现 | 向量检索 + BM25 关键词检索 + RRF 融合 | 3h |
| 2.6 初始化脚本 | init_vector_store.py：一键导入题库 | 1h |

**验收标准**：输入 "Python 异步编程"，能检索出相关面试题，排序合理。

---

### Phase 3：Multi-Agent 核心（第 2 周，Day 7-11）⭐ 最重要

**目标**：四个 Agent 能协作完成一次完整的面试流程。

| 任务 | 产出 | 预计耗时 |
|------|------|---------|
| 3.1 JD Analyst Agent | 输入 JD → 输出 SkillMatrix（structured output） | 3h |
| 3.2 Question Planner Agent | SkillMatrix + RAG 检索 → 面试题单 | 4h |
| 3.3 Interviewer Agent | 面试对话循环 + 追问决策逻辑 | 5h |
| 3.4 Evaluator Agent | 对话记录 → 结构化评分报告 | 4h |
| 3.5 LangGraph 编排 | graph.py：状态图定义 + 条件边 + 中断点 | 4h |
| 3.6 Prompt 工程 | 每个 Agent 的 system prompt + few-shot examples | 3h |

**验收标准**：在终端里能跑通完整流程——输入 JD，自动出题，逐题问答，输出评分报告。

**这个阶段的关键设计决策**：

```python
# graph.py 的核心路由逻辑（面试时必须能讲清楚）
def route_after_assessment(state: InterviewState) -> str:
    latest = state["assessments"][-1]
    
    # 回答太差且还没追问过 → 追问
    if latest.score < 4 and state["follow_up_count"] < state["max_follow_ups"]:
        return "ask_follow_up"
    
    # 回答尚可但有遗漏点 → 追问一次
    if latest.score < 7 and latest.missed_points and state["follow_up_count"] == 0:
        return "ask_follow_up"
    
    # 还有下一题 → 继续
    if state["current_question_index"] < len(state["question_plan"]) - 1:
        return "next_question"
    
    # 全部问完 → 评估
    return "evaluate"
```

---

### Phase 4：WebSocket 实时对话（第 2 周末，Day 12-13）

**目标**：面试过程通过 WebSocket 实时进行。

| 任务 | 产出 | 预计耗时 |
|------|------|---------|
| 4.1 WebSocket 端点 | interview_ws.py：连接管理 + 消息协议 | 3h |
| 4.2 会话管理 | session_manager.py：面试状态持久化 | 2h |
| 4.3 流式输出 | Agent 回复通过 WebSocket 流式推送 | 2h |
| 4.4 消息协议设计 | 定义 client/server 消息类型和格式 | 1h |

**WebSocket 消息协议**：

```python
# Client → Server
{"type": "start_interview", "jd_text": "..."}
{"type": "answer", "content": "我认为..."}
{"type": "end_interview"}

# Server → Client  
{"type": "status", "stage": "analyzing_jd", "message": "正在分析岗位要求..."}
{"type": "question", "data": {"id": 1, "content": "请介绍...", "skill": "Python"}}
{"type": "question_stream", "chunk": "请"}  # 流式输出
{"type": "follow_up", "data": {"content": "你能再展开讲讲..."}}
{"type": "report", "data": { ... }}
```

**验收标准**：用 WebSocket 客户端工具能完成一次完整面试。

---

### Phase 5：Gradio 前端界面（第 3 周前半，Day 14-16）

**目标**：做一个可演示、有视觉冲击力的面试界面。

| 任务 | 产出 | 预计耗时 |
|------|------|---------|
| 5.1 面试主界面 | 聊天窗口 + JD 输入区 + 状态指示 | 3h |
| 5.2 面试流程控制 | 开始/暂停/结束按钮 + 进度显示 | 2h |
| 5.3 评分报告展示 | 雷达图 + 分数卡片 + 详细点评 | 3h |
| 5.4 UI 美化 | 自定义 CSS + 响应式布局 | 2h |

**验收标准**：录一个 2 分钟的 Demo GIF，流程顺畅，UI 美观。

---

### Phase 6：评估体系 + 文档（第 3 周后半，Day 17-19）

**目标**：让项目可量化、可复现、可展示。

| 任务 | 产出 | 预计耗时 |
|------|------|---------|
| 6.1 评分 Rubric 设计 | 各维度评分标准（面试时必讲） | 2h |
| 6.2 评估脚本 | 用 5-10 组 case 跑一致性测试 | 3h |
| 6.3 README 撰写 | 项目介绍 + Quick Start + 架构图 + 演示 GIF | 3h |
| 6.4 架构文档 | 详细技术文档 + LangGraph 状态图 | 2h |

**验收标准**：一个新人看 README 能在 5 分钟内跑起来。

---

## 六、面试高频追问 & 你的应答策略

| 面试官会问 | 你应该怎么答 |
|-----------|-------------|
| 为什么用 LangGraph 而不是简单的 chain？ | 面试场景有**循环**（多轮追问）和**条件分支**（追问/下一题/结束），普通 chain 是 DAG 无法处理循环，LangGraph 支持有环图和人在环中的中断 |
| ChromaDB 够用吗？生产环境怎么办？ | 本项目题库规模在万级以内，ChromaDB 完全够用。如果要上生产，可以平滑迁移到 Milvus/Qdrant，因为我的 vector_store.py 做了抽象层 |
| 你的追问逻辑怎么设计的？ | 基于 score + missed_points + follow_up_count 三个维度做路由决策，每题最多追问 2 次，避免死循环。追问方向在出题阶段就预设了，不是临场瞎问 |
| LLM 评分不稳定怎么办？ | 三个手段：(1) Structured Output 强制 JSON 格式 (2) Scoring Rubric 写进 prompt 做 few-shot (3) 评估脚本验证评分一致性（同一回答跑 5 次看方差） |
| 如何处理用户回答超长/跑题？ | Interviewer Agent 的 prompt 中有角色设定——控制节奏，会打断跑题、引导回正题，这也是 Agent 比纯 chain 的优势 |
| 多 Agent 之间怎么通信？ | 通过 LangGraph 的 InterviewState 共享状态，每个 Agent 只读自己需要的字段、写自己负责的字段，无直接耦合 |

---

## 七、关键成功指标

- [ ] 输入一个 JD，3 秒内完成分析并开始出题
- [ ] 单次面试 5-8 题，15-20 分钟完成
- [ ] 追问逻辑合理，不会出现无意义重复追问
- [ ] 评分报告覆盖所有考察技能维度
- [ ] Docker 一键部署，README 清晰可复现
- [ ] Demo GIF 展示完整面试流程
