# AI Mock Interview Agent

基于 **Multi-Agent 架构**的智能面试模拟系统。输入一段 JD（职位描述），即可获得一场自适应、可评分的 AI 模拟面试。

## 核心亮点

- **Multi-Agent 协作**：4 个专职 Agent（JD 分析师 / 出题策划师 / 面试官 / 评估师）通过 LangGraph 编排协作
- **自适应追问**：根据回答质量动态决定是否深挖、换题、调整难度
- **混合检索 RAG**：向量搜索 + BM25 关键词搜索 + RRF 融合排序，从 62 道题库中精准匹配
- **结构化评估**：基于 Rubric 的评分体系，输出可量化的面试报告（各技能维度评分 + 等级 + 建议）
- **工程化交付**：FastAPI + WebSocket 实时对话 + Gradio 可视化界面 + Docker 一键部署

## 技术栈

| 层级 | 技术 |
|------|------|
| Agent 编排 | LangGraph (StateGraph + 条件边 + interrupt) |
| LLM 接入 | OpenAI SDK（兼容 DeepSeek / Qwen 等） |
| RAG 检索 | ChromaDB + BM25 (rank-bm25) + RRF 融合 |
| Web 框架 | FastAPI + WebSocket |
| 数据校验 | Pydantic v2 (Structured Output) |
| 前端界面 | Gradio |
| 容器化 | Docker + Docker Compose |

## 架构概览

```
                    ┌──────────┐
                    │  START   │
                    └────┬─────┘
                         │
                         v
                ┌────────────────┐
                │  JD Analyst    │  解析 JD → 技能矩阵
                │  Agent         │  (Structured Output)
                └────────┬───────┘
                         │
                         v
                ┌────────────────┐
                │  Question      │  技能矩阵 + RAG → 面试题单
                │  Planner Agent │  (混合检索 + LLM 生成)
                └────────┬───────┘
                         │
                         v
                ┌────────────────┐
            ┌──>│  Interviewer   │  提问 → 等待回答 → 评估
            │   │  Agent         │  → 追问/下一题/结束
            │   └────────┬───────┘
            │            │
            └────────────┘  (循环：追问 or 下一题)
                         │
                         v  (全部问完)
                ┌────────────────┐
                │  Evaluator     │  完整对话 → 结构化报告
                │  Agent         │  (技能评分 + 等级 + 建议)
                └────────┬───────┘
                         │
                         v
                    ┌──────────┐
                    │   END    │
                    └──────────┘
```

## Quick Start

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd ai-mock-interview
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key 和模型配置
```

### 4. 初始化题库向量数据库

```bash
python -m scripts.init_vector_store
```

### 5. 启动

**方式 A：Gradio 界面（推荐演示）**

```bash
python -m frontend.gradio_app
# 打开 http://127.0.0.1:7860
```

**方式 B：FastAPI + WebSocket**

```bash
uvicorn app.main:app --port 8000
# API 文档: http://127.0.0.1:8000/docs
# WebSocket: ws://127.0.0.1:8000/api/ws/interview/{session_id}
```

**方式 C：CLI 命令行**

```bash
python -m scripts.run_interview_cli
```

**方式 D：Docker**

```bash
docker compose up --build
```

## 项目结构

```
ai-mock-interview/
├── app/
│   ├── agents/            # Multi-Agent 核心
│   │   ├── state.py       # LangGraph 共享状态 (InterviewState)
│   │   ├── graph.py       # LangGraph 图编排（核心）
│   │   ├── jd_analyst.py  # JD 分析 Agent
│   │   ├── question_planner.py  # 出题策划 Agent
│   │   ├── interviewer.py # 面试官 Agent（提问/追问/评估）
│   │   └── evaluator.py   # 评估 Agent
│   ├── rag/               # RAG 检索模块
│   │   ├── vector_store.py    # ChromaDB 封装
│   │   ├── embeddings.py      # Embedding 服务
│   │   └── retriever.py       # 混合检索 (向量+BM25+RRF)
│   ├── llm/               # LLM 服务层
│   │   ├── client.py      # 统一 async 客户端 (chat/stream/structured)
│   │   └── prompts.py     # 全部 Prompt 集中管理
│   ├── models/            # Pydantic 数据模型
│   ├── api/               # FastAPI 路由 (REST + WebSocket)
│   └── services/          # 会话管理 + LangGraph 驱动
├── data/
│   ├── question_bank/     # 面试题库 (62 题 × 3 方向)
│   └── sample_jds/        # 示例 JD
├── frontend/
│   └── gradio_app.py      # Gradio 可视化界面
├── scripts/               # 工具脚本
├── docs/                  # 技术文档
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 面试题库

| 方向 | 题数 | 覆盖主题 |
|------|------|---------|
| Python | 22 | GIL、装饰器、异步编程、内存管理、元类、并发、类型系统等 |
| System Design | 20 | URL 短链、消息队列、缓存、微服务、CAP、Docker、CI/CD 等 |
| ML & AI | 20 | Transformer、RAG、Agent、LangGraph、RLHF、向量数据库等 |

## 评估体系

详见 [评分 Rubric](docs/scoring_rubric.md)

- 单题评分 1-10 分，基于参考答案要点覆盖率
- 追问规则：score < 4 必追问，4-6 酌情追问，≥ 7 直接下一题
- 综合评分：技能权重加权平均
- 等级：A (≥8) / B (≥6.5) / C (≥5) / D (<5)

运行评估脚本验证评分一致性：

```bash
python -m scripts.evaluate
```

## 技术架构详解

详见 [架构文档](docs/architecture.md)

## License

MIT
