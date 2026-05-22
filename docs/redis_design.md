# Redis 运行时增强设计

本项目的 Redis 不是替代原有核心存储，而是放在 FastAPI 和 LangGraph 面试流程外侧的运行时增强层。核心状态仍然由 `SessionManager`、LangGraph checkpoint、SQLite memory 和 ChromaDB/RAG 负责；Redis 负责短生命周期、高并发、可过期的后端能力。

## 设计目标

- 面试配额控制：限制公开体验环境里创建会话、上传简历、提交回答的频率，避免 LLM API 被刷。
- 并发安全：同一个会话同时提交两次回答时，用 Redis 分布式锁保护跨进程场景。
- WebSocket 在线状态：连接建立后写入带 TTL 的在线标记，断开时删除，心跳时续期。
- 会话快照缓存：把 session 元信息、轻量图状态、最终报告写入 Redis，便于后台查看或后续扩展恢复体验。
- 保持简单可讲：采用 `INCR + EXPIRE`、`SET NX PX`、TTL key 这些面试中常见且容易解释的 Redis 模式。

## 接入位置

| 模块 | Redis 能力 | 说明 |
| --- | --- | --- |
| `app/cache/redis_client.py` | 客户端单例 | 通过 `REDIS_ENABLED` 控制开关，连接失败时 fail-open |
| `app/cache/rate_limiter.py` | 固定窗口限流 | `INCR` 统计窗口内请求数，`EXPIRE` 自动清理 |
| `app/cache/locks.py` | 分布式锁 | `SET key value NX PX ttl` 获取锁，释放前校验 token |
| `app/cache/websocket_presence.py` | 在线状态 | `ws:session:{session_id}:online`，带 TTL |
| `app/cache/session_cache.py` | 会话/报告缓存 | 保存轻量 session snapshot 和 report |
| `app/api/interview_rest.py` | REST 限流 | start、resume、answer 接口触发限流 |
| `app/api/interview_ws.py` | WS 在线/限流 | 连接标记在线，ping/answer 续期，answer 做限流 |
| `app/services/session_manager.py` | 锁与快照 | answer 外层加 Redis 锁，图执行后写快照 |

## Key 设计

| Key | 示例 | 生命周期 | 用途 |
| --- | --- | --- | --- |
| `rl:{name}:{identity_hash}:{window}` | `rl:interview:start:abc123:29670000` | 约一个窗口 | API 固定窗口限流 |
| `lock:session:{id}:answer` | `lock:session:s1:answer` | 60 秒 | 防止同一会话并发处理回答 |
| `ws:session:{id}:online` | `ws:session:s1:online` | 90 秒 | WebSocket 在线状态 |
| `session:{id}:meta` | `session:s1:meta` | 1 天 | 会话基础信息 |
| `session:{id}:snapshot` | `session:s1:snapshot` | 1 天 | 当前轮次、题目数、最近面试官消息 |
| `session:{id}:report` | `session:s1:report` | 7 天 | 最终报告缓存 |

## Fail-open 策略

Redis 是增强层，不是主链路强依赖：

- Redis 未安装、未启用或连接失败时，限流、在线状态、缓存直接跳过。
- 分布式锁不可用时，仍保留原来的 `asyncio.Lock` 保护单进程并发。
- 只有 Redis 可用且锁已经被其他请求占用时，才返回“正在处理上一条回答”的冲突错误。

这个策略适合课程项目和简历项目：既能展示 Redis 后端思想，又不会为了引入 Redis 破坏原本稳定的 AI 面试流程。

## Docker Compose 使用

`docker-compose.yml` 已加入 Redis 7：

```bash
docker compose build
docker compose up -d
docker compose ps
```

Compose 中 API 服务会自动设置：

```env
REDIS_ENABLED=true
REDIS_URL=redis://redis:6379/0
```

普通本地 Python 运行默认保持：

```env
REDIS_ENABLED=false
REDIS_URL=redis://localhost:6379/0
```

需要本地直接启用时，先启动 Redis，再把 `.env` 中 `REDIS_ENABLED` 改为 `true`。
