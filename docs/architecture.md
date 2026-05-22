# 架构文档 — AI Mock Interview Agent

## 一、系统总览

本系统基于 Multi-Agent 架构，使用 LangGraph 编排 4 个专职 Agent 完成一次完整的模拟面试。

```
用户输入 JD
    │
    v
┌─────────────────────────────────────────────────────┐
│                 LangGraph StateGraph                │
│                                                     │
│  [JD Analyst] → [Question Planner] → [Interviewer]  │
│                                         ↑    │      │
│                                         └────┘      │
│                                 (循环:追问/下一题)    │
│                                         │           │
│                                    [Evaluator]      │
└─────────────────────────────────────────────────────┘
    │
    v
结构化面试报告
```

## 二、Agent 职责划分

### Agent 1: JD Analyst（JD 分析师）

**输入**: 原始 JD 文本
**输出**: `SkillMatrix` (Pydantic Model)

```json
{
  "position_title": "AI 应用开发工程师（实习）",
  "experience_level": "intern",
  "skills": [
    {"name": "Python", "category": "language", "weight": 0.9, "is_required": true},
    {"name": "LangGraph", "category": "framework", "weight": 0.8, "is_required": true},
    {"name": "系统设计", "category": "system_design", "weight": 0.5, "is_required": false}
  ]
}
```

**关键技术**: Structured Output — LLM 输出必须符合 JSON Schema，通过 `response_format=json_object` + schema 注入实现。

### Agent 2: Question Planner（出题策划师）

**输入**: SkillMatrix + RAG 检索结果
**输出**: `QuestionPlan` (5-8 题)

**流程**:
1. 从 SkillMatrix 提取 top-6 权重最高的技能
2. 用技能名作为 query，通过 HybridRetriever 检索相关题目
3. 将技能矩阵 + 检索结果拼接到 Prompt
4. LLM 生成结构化题单（可基于检索结果改编，也可全新生成）

**出题策略**:
- 难度分布: ~30% easy + ~50% medium + ~20% hard
- 从易到难排列（warm-up first）
- 每题附带参考答案要点和追问方向

### Agent 3: Interviewer（面试官）

最复杂的 Agent，由 3 个 LangGraph 节点组成：

| 节点 | 职责 | 对用户可见 |
|------|------|-----------|
| `ask_question` | 自然语言提问 | ✅ |
| `ask_follow_up` | 针对遗漏点追问 | ✅ |
| `assess_answer` | 内部评估回答质量 | ❌ |

**追问决策（路由函数）**:

```python
def route_after_assessment(state) -> str:
    latest = state["assessments"][-1]
    
    if latest.should_follow_up:          # 评估认为需要追问
        return "ask_follow_up"
    
    if current_idx < total - 1:          # 还有下一题
        return "ask_question"
    
    return "evaluate"                    # 全部问完 → 评估
```

### Agent 4: Evaluator（评估师）

**输入**: 完整对话记录 + 各题评估结果 + 技能矩阵
**输出**: `InterviewReport`

独立于面试官，从"评审委员会"视角综合评估。

## 三、LangGraph 状态图

```
                    ┌──────────┐
                    │ __start__│
                    └────┬─────┘
                         │
                         v
                ┌────────────────┐
                │  analyze_jd    │
                └────────┬───────┘
                         │
                         v
                ┌────────────────┐
                │plan_questions  │
                └────────┬───────┘
                         │
                         v
                ┌────────────────┐
            ┌──>│ ask_question   │
            │   └────────┬───────┘
            │            │
            │            v
            │   ┌────────────────┐
            │   │ assess_answer  │ ← interrupt_before (等待用户输入)
            │   └────────┬───────┘
            │            │
            │    ┌───────┼───────────┐
            │    │       │           │
            │  [追问]  [下一题]    [结束]
            │    │       │           │
            │    v       v           v
            │  ┌──────┐ ┌────────┐ ┌──────────────┐
            │  │follow│ │advance │ │evaluate_     │
            │  │_up   │ │question│ │interview     │
            │  └──┬───┘ └───┬────┘ └──────┬───────┘
            │     │         │             │
            │     └──>assess│             v
            │         _answer        ┌──────────┐
            └───────────────┘        │ __end__  │
                                     └──────────┘
```

### 关键设计: interrupt_before

```python
graph.compile(interrupt_before=["assess_answer"])
```

图在 `assess_answer` 节点前暂停，等待外部注入候选人的回答。这是 LangGraph 的 **human-in-the-loop** 模式：

1. Graph 运行到 `ask_question` → 生成面试官的提问
2. Graph 暂停 → 提问通过 WebSocket/Gradio 展示给用户
3. 用户输入回答 → 通过 `update_state()` 注入 `current_candidate_answer`
4. Graph 恢复 → `assess_answer` 评估回答 → 路由到下一步

### 状态定义 (InterviewState)

```python
class InterviewState(TypedDict, total=False):
    jd_text: str                                      # 输入
    skill_matrix: SkillMatrix                         # JD Analyst 写入
    question_plan: list[QuestionItem]                  # Planner 写入
    current_question_index: int                        # 面试进度
    follow_up_count: int                               # 当前题追问次数
    max_follow_ups: int                                # 每题最大追问
    conversation_history: Annotated[list, reducer]     # 对话记录（追加）
    current_candidate_answer: str                      # 外部注入的回答
    assessments: Annotated[list, operator.add]         # 评估结果（追加）
    final_report: InterviewReport                      # Evaluator 写入
    interview_complete: bool                           # 控制流标志
```

## 四、RAG 混合检索

### 检索流程

```
Query ("Python 异步编程 Agent")
    │
    ├── 向量检索 (ChromaDB)
    │   └── cosine similarity top-30
    │
    ├── BM25 关键词检索
    │   └── token 匹配 top-30
    │
    └── RRF 融合排序
        score(d) = Σ w_i / (k + rank_i(d))
        └── 返回 top-10
```

### 数据入库格式

每道题的索引文本:
```
[Python] 请解释 Python 中的 GIL 是什么？
技能标签: Python, 并发编程, 多线程
难度: medium
参考要点: GIL 是 CPython 中的互斥锁; 对 CPU 密集型影响大; ...
```

元数据字段: `question_id`, `category`, `difficulty`, `skill_tags`, `reference_points`, `follow_up_directions`

## 五、WebSocket 通信协议

### Client → Server

| type | 用途 |
|------|------|
| `answer` | 提交候选人回答 |
| `end_interview` | 提前结束面试 |

### Server → Client

| type | 用途 |
|------|------|
| `status` | 状态更新 (analyzing_jd / processing / ...) |
| `question` | 新问题 (含 index, total, skill_tags, difficulty) |
| `follow_up` | 追问 (含 follow_up_number) |
| `report` | 最终评估报告 |
| `error` | 错误信息 |

## 六、Structured Output 策略

所有 Agent 的结构化输出通过 `LLMClient.chat_structured()` 实现：

1. 从 Pydantic Model 自动提取 JSON Schema
2. 将 Schema 注入 System Prompt 末尾
3. 使用 `response_format={"type": "json_object"}` 强制 JSON
4. 对 LLM 输出进行 `model_validate_json()` 验证
5. 如果 `json_object` 模式不支持，自动降级为 prompt-based 提取

## 七、评分一致性保障

1. **Rubric 标准化**: 1-10 分 Rubric 写入每个评分 Prompt
2. **低温采样**: 评分类调用使用 `temperature=0.2`
3. **结构化输出**: 强制 JSON 格式，消除格式不一致
4. **评估脚本**: `scripts/evaluate.py` 对同一回答多次评分，检查方差
