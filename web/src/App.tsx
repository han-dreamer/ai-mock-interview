import {
  Activity,
  AlertCircle,
  ArrowLeft,
  Bot,
  Brain,
  CheckCircle2,
  CircleStop,
  ClipboardList,
  FileText,
  History,
  KeyRound,
  Loader2,
  LogOut,
  MessageSquareText,
  Play,
  RotateCcw,
  Send,
  Sparkles,
  Upload,
  User,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent, RefObject } from "react";
import {
  API_BASE_URL,
  checkHealth,
  exportInterviewReport,
  getInterviewReport,
  getMe,
  listInterviewSessions,
  login,
  register,
  startInterview,
  websocketUrl,
} from "./api";
import type {
  ChatItem,
  CurrentTurn,
  InterviewMode,
  InterviewReport,
  InterviewSessionSummary,
  ResumeParseResult,
  ServerMessage,
  UserPublic,
} from "./types";

const sampleJd = `岗位：AI 应用开发工程师

岗位职责：
1. 参与 AI Agent、RAG、智能问答和工作流编排等大模型应用的设计与开发；
2. 使用 Python / FastAPI 构建后端服务，支持流式对话、文件上传、异步任务和 WebSocket 实时交互；
3. 负责向量检索、Prompt Engineering、结构化输出、模型调用稳定性和效果评估；
4. 与产品和前端协作，将原型能力落地为可演示、可部署的 AI 应用。

岗位要求：
- 熟悉 Python、FastAPI、LangChain/LangGraph 或同类 Agent 框架；
- 理解 RAG 的切分、召回、重排、评估和工程化优化；
- 熟悉 REST API、WebSocket、数据库和基础部署流程；
- 有 AI 应用项目经验，能解释架构设计、异常处理和效果评估方法。`;

type View = "setup" | "history" | "interview" | "report";
type ConnectionState = "idle" | "connecting" | "open" | "closed" | "error";
type AuthMode = "login" | "register";

const uid = () => `${Date.now()}-${Math.random().toString(16).slice(2)}`;
const AUTH_TOKEN_KEY = "ai-mock-auth-token";

const modeName = (mode: InterviewMode) =>
  mode === "professional" ? "专业面试模式" : "练习模式";

const difficultyName = (difficulty?: string) => {
  const map: Record<string, string> = {
    easy: "基础",
    medium: "中等",
    hard: "进阶",
  };
  return difficulty ? (map[difficulty] ?? difficulty) : "";
};

const statusText = (stage: string, fallback: string) => {
  const map: Record<string, string> = {
    analyzing_jd: "正在分析 JD、简历和历史记忆，准备个性化问题。",
    questions_ready: "问题已准备好，面试即将开始。",
    resumed: "已恢复到当前面试进度。",
    waiting: "正在等待下一轮问题。",
    processing: "正在评估你的回答，并判断是否需要追问。",
    evaluating: "正在结束面试并生成评估报告。",
    pong: "连接正常。",
  };
  return map[stage] ?? fallback;
};

export function App() {
  const [view, setView] = useState<View>("setup");
  const [jdText, setJdText] = useState(sampleJd);
  const [mode, setMode] = useState<InterviewMode>("professional");
  const [maxFollowUps, setMaxFollowUps] = useState(2);
  const [authToken, setAuthToken] = useState(() => localStorage.getItem(AUTH_TOKEN_KEY) ?? "");
  const [currentUser, setCurrentUser] = useState<UserPublic | null>(null);
  const [authMode, setAuthMode] = useState<AuthMode>("login");
  const [authUsername, setAuthUsername] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [authError, setAuthError] = useState("");
  const [isAuthenticating, setIsAuthenticating] = useState(false);
  const [isCheckingAuth, setIsCheckingAuth] = useState(true);
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [resume, setResume] = useState<ResumeParseResult | null>(null);
  const [sessionId, setSessionId] = useState("");
  const [historyItems, setHistoryItems] = useState<InterviewSessionSummary[]>([]);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [messages, setMessages] = useState<ChatItem[]>([]);
  const [currentTurn, setCurrentTurn] = useState<CurrentTurn | null>(null);
  const [answer, setAnswer] = useState("");
  const [status, setStatus] = useState("填写 JD，可选上传简历后开始模拟面试。");
  const [stage, setStage] = useState("ready");
  const [error, setError] = useState("");
  const [isStarting, setIsStarting] = useState(false);
  const [isAwaitingAnswer, setIsAwaitingAnswer] = useState(false);
  const [connection, setConnection] = useState<ConnectionState>("idle");
  const [report, setReport] = useState<InterviewReport | null>(null);
  const [health, setHealth] = useState<{ status: string; model?: string; debug?: boolean } | null>(
    null,
  );

  const socketRef = useRef<WebSocket | null>(null);
  const transcriptRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    checkHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  useEffect(() => {
    let active = true;
    if (!authToken) {
      setCurrentUser(null);
      setIsCheckingAuth(false);
      return;
    }

    setIsCheckingAuth(true);
    getMe(authToken)
      .then((user) => {
        if (active) {
          setCurrentUser(user);
        }
      })
      .catch(() => {
        if (active) {
          localStorage.removeItem(AUTH_TOKEN_KEY);
          setAuthToken("");
          setCurrentUser(null);
        }
      })
      .finally(() => {
        if (active) {
          setIsCheckingAuth(false);
        }
      });

    return () => {
      active = false;
    };
  }, [authToken]);

  useEffect(() => {
    transcriptRef.current?.scrollTo({
      top: transcriptRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, status]);

  useEffect(() => {
    return () => socketRef.current?.close();
  }, []);

  const canStart = Boolean(currentUser && authToken && jdText.trim().length >= 10 && !isStarting);

  async function loadHistory() {
    if (!authToken) return;
    setIsHistoryLoading(true);
    setHistoryError("");
    try {
      setHistoryItems(await listInterviewSessions(authToken));
    } catch (err) {
      setHistoryError(err instanceof Error ? err.message : "加载历史记录失败。");
    } finally {
      setIsHistoryLoading(false);
    }
  }

  async function handleAuthSubmit(event: FormEvent) {
    event.preventDefault();
    setAuthError("");
    setIsAuthenticating(true);
    try {
      const response =
        authMode === "login"
          ? await login({ username: authUsername.trim(), password: authPassword })
          : await register({
              username: authUsername.trim(),
              password: authPassword,
              displayName: displayName.trim(),
            });
      localStorage.setItem(AUTH_TOKEN_KEY, response.access_token);
      setAuthToken(response.access_token);
      setCurrentUser(response.user);
      setAuthPassword("");
      setView("setup");
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : "认证失败，请稍后重试。");
    } finally {
      setIsAuthenticating(false);
    }
  }

  function handleLogout() {
    socketRef.current?.close();
    localStorage.removeItem(AUTH_TOKEN_KEY);
    setAuthToken("");
    setCurrentUser(null);
    setAuthPassword("");
    setAuthError("");
    resetAll();
  }

  async function handleStart() {
    if (!canStart) return;
    setIsStarting(true);
    setError("");
    setStatus(resumeFile ? "正在上传并解析简历，请稍候。" : "正在创建面试会话。");
    try {
      const response = await startInterview({
        jdText: jdText.trim(),
        maxFollowUps,
        mode,
        token: authToken,
        resumeFile,
      });
      setSessionId(response.session_id);
      setResume(response.resume ?? null);
      setMessages([
        {
          id: uid(),
          role: "system",
          content:
            response.mode === "professional"
              ? "专业面试已创建。系统会结合 JD、简历和长期记忆，先进行项目深挖，再进入技术广度考察。"
              : "练习面试已创建。系统会根据 JD 生成问题，并在报告中给出参考答案和学习建议。",
        },
      ]);
      setView("interview");
      connectSocket(response.session_id);
      void loadHistory();
    } catch (err) {
      setError(err instanceof Error ? err.message : "创建面试失败");
      setStatus("创建会话失败，请检查登录状态、后端服务或模型配置。");
    } finally {
      setIsStarting(false);
    }
  }

  function connectSocket(id = sessionId) {
    if (!id) return;
    socketRef.current?.close();
    setConnection("connecting");
    setStatus("正在连接实时面试通道。");
    const ws = new WebSocket(websocketUrl(id, authToken));
    socketRef.current = ws;

    ws.onopen = () => {
      setConnection("open");
      setStatus("连接成功，面试官正在准备第一轮问题。");
      ws.send(JSON.stringify({ type: "start_interview" }));
    };

    ws.onmessage = (event) => {
      const payload = JSON.parse(event.data) as ServerMessage;
      handleServerMessage(payload);
    };

    ws.onerror = () => {
      setConnection("error");
      setError("WebSocket 连接异常，请稍后重连或检查后端服务。");
    };

    ws.onclose = () => {
      setConnection((prev) => (prev === "error" ? "error" : "closed"));
    };
  }

  async function openHistoryView() {
    setView("history");
    await loadHistory();
  }

  async function openHistoricalReport(item: InterviewSessionSummary) {
    setError("");
    try {
      const data = await getInterviewReport(authToken, item.session_id);
      socketRef.current?.close();
      setSessionId(item.session_id);
      setMode(item.mode);
      setMessages([]);
      setReport(data);
      setConnection("idle");
      setView("report");
    } catch (err) {
      setHistoryError(err instanceof Error ? err.message : "加载报告失败。");
    }
  }

  async function downloadReport(sessionId: string, format: "markdown" | "pdf") {
    setError("");
    setHistoryError("");
    try {
      const blob = await exportInterviewReport(authToken, sessionId, format);
      if (format === "pdf") {
        openPrintableReport(blob, sessionId);
        return;
      }
      downloadBlob(blob, `interview-report-${sessionId.slice(0, 8)}.md`);
    } catch (err) {
      const message = err instanceof Error ? err.message : "导出报告失败。";
      setError(message);
      setHistoryError(message);
    }
  }

  function resumeHistoricalInterview(item: InterviewSessionSummary) {
    socketRef.current?.close();
    setSessionId(item.session_id);
    setMode(item.mode);
    setResume(null);
    setMessages([
      {
        id: uid(),
        role: "system",
        content: "正在恢复历史面试，会从后端 checkpoint 读取当前进度。",
      },
    ]);
    setCurrentTurn(null);
    setAnswer("");
    setReport(null);
    setError("");
    setStatus("正在恢复历史面试。");
    setStage("resuming");
    setView("interview");
    connectSocket(item.session_id);
  }

  function handleServerMessage(payload: ServerMessage) {
    if (payload.type === "status") {
      setStage(payload.stage);
      setStatus(statusText(payload.stage, payload.message));
      return;
    }

    if (payload.type === "question") {
      setIsAwaitingAnswer(true);
      setCurrentTurn({
        questionIndex: payload.question_index,
        totalQuestions: payload.total_questions,
        skillTags: payload.skill_tags,
        difficulty: payload.difficulty,
      });
      setStatus("请作答当前问题。");
      setMessages((items) => [
        ...items,
        {
          id: uid(),
          role: "interviewer",
          content: payload.content,
          meta: `第 ${payload.question_index}/${payload.total_questions} 题 · ${difficultyName(
            payload.difficulty,
          )}`,
        },
      ]);
      return;
    }

    if (payload.type === "follow_up") {
      setIsAwaitingAnswer(true);
      setCurrentTurn({
        questionIndex: payload.question_index,
        followUpNumber: payload.follow_up_number,
      });
      setStatus("面试官正在追问，请继续补充回答。");
      setMessages((items) => [
        ...items,
        {
          id: uid(),
          role: "interviewer",
          content: payload.content,
          meta: `第 ${payload.question_index} 题追问 · 第 ${payload.follow_up_number} 次`,
        },
      ]);
      return;
    }

    if (payload.type === "interview_end") {
      setIsAwaitingAnswer(false);
      setStatus("面试已结束，正在整理评估报告。");
      setStage("evaluating");
      return;
    }

    if (payload.type === "report") {
      setReport(payload.data);
      setStatus("报告已生成。");
      setStage("completed");
      setView("report");
      void loadHistory();
      return;
    }

    if (payload.type === "error") {
      setError(payload.message);
      setStatus("处理过程中出现错误，请查看提示后重试。");
    }
  }

  function submitAnswer(event: FormEvent) {
    event.preventDefault();
    const content = answer.trim();
    if (!content || !socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) return;
    socketRef.current.send(JSON.stringify({ type: "answer", content }));
    setMessages((items) => [...items, { id: uid(), role: "candidate", content }]);
    setAnswer("");
    setIsAwaitingAnswer(false);
    setStatus("已提交回答，正在评估。");
    setStage("processing");
  }

  function stopInterview() {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ type: "stop" }));
      setStatus("正在提前结束面试并生成阶段性报告。");
      setIsAwaitingAnswer(false);
    }
  }

  function resetAll() {
    socketRef.current?.close();
    setView("setup");
    setSessionId("");
    setMessages([]);
    setCurrentTurn(null);
    setAnswer("");
    setStatus("填写 JD，可选上传简历后开始模拟面试。");
    setStage("ready");
    setError("");
    setReport(null);
    setResume(null);
    setConnection("idle");
  }

  const progress = useMemo(() => {
    if (!currentTurn?.questionIndex || !currentTurn.totalQuestions) return 0;
    return Math.round((currentTurn.questionIndex / currentTurn.totalQuestions) * 100);
  }, [currentTurn]);

  return (
    <main className="app-shell">
      <div className="ambient ambient-left" />
      <div className="ambient ambient-right" />
      <header className="topbar">
        <button className="brand" onClick={resetAll} title="回到开始页">
          <span className="brand-mark">
            <Brain size={18} />
          </span>
          <span>
            <strong>AI Mock Interview</strong>
            <small>JD 与简历驱动的 Agent 面试系统</small>
          </span>
        </button>
        <div className="topbar-actions">
          {currentUser && (
            <span className="api-pill" title={currentUser.username}>
              <User size={14} />
              {currentUser.display_name || currentUser.username}
            </span>
          )}
          <StatusPill connection={connection} />
          <span className="api-pill" title={API_BASE_URL}>
            <Activity size={14} />
            {health?.status === "ok" ? health.model ?? "后端在线" : "检查后端"}
          </span>
          {currentUser && (
            <button className="icon-text-button" onClick={handleLogout} type="button">
              <LogOut size={14} />
              退出
            </button>
          )}
        </div>
      </header>

      {currentUser && (
        <nav className="view-tabs" aria-label="工作台视图">
          <button className={view === "setup" ? "active" : ""} onClick={() => setView("setup")} type="button">
            <Play size={15} />
            开始面试
          </button>
          <button className={view === "history" ? "active" : ""} onClick={openHistoryView} type="button">
            <History size={15} />
            历史记录
          </button>
        </nav>
      )}

      {!currentUser && (
        <AuthPage
          mode={authMode}
          setMode={setAuthMode}
          username={authUsername}
          setUsername={setAuthUsername}
          password={authPassword}
          setPassword={setAuthPassword}
          displayName={displayName}
          setDisplayName={setDisplayName}
          isSubmitting={isAuthenticating}
          isChecking={isCheckingAuth}
          error={authError}
          onSubmit={handleAuthSubmit}
        />
      )}

      {currentUser && view === "setup" && (
        <SetupPage
          jdText={jdText}
          setJdText={setJdText}
          mode={mode}
          setMode={setMode}
          maxFollowUps={maxFollowUps}
          setMaxFollowUps={setMaxFollowUps}
          currentUser={currentUser}
          resumeFile={resumeFile}
          setResumeFile={setResumeFile}
          isStarting={isStarting}
          canStart={canStart}
          error={error}
          onStart={handleStart}
        />
      )}

      {currentUser && view === "interview" && (
        <InterviewPage
          mode={mode}
          sessionId={sessionId}
          resume={resume}
          status={status}
          stage={stage}
          error={error}
          messages={messages}
          currentTurn={currentTurn}
          progress={progress}
          answer={answer}
          setAnswer={setAnswer}
          isAwaitingAnswer={isAwaitingAnswer}
          connection={connection}
          transcriptRef={transcriptRef}
          onSubmitAnswer={submitAnswer}
          onStop={stopInterview}
          onReconnect={() => connectSocket()}
          onBack={() => setView("setup")}
        />
      )}

      {currentUser && view === "history" && (
        <HistoryPage
          items={historyItems}
          isLoading={isHistoryLoading}
          error={historyError}
          onRefresh={loadHistory}
          onViewReport={openHistoricalReport}
          onResume={resumeHistoricalInterview}
          onExport={downloadReport}
        />
      )}

      {currentUser && view === "report" && report && (
        <ReportPage
          report={report}
          mode={mode}
          sessionId={sessionId}
          messages={messages}
          onRestart={resetAll}
          onBackToInterview={() => setView("interview")}
          onExport={downloadReport}
        />
      )}
    </main>
  );
}

function AuthPage(props: {
  mode: AuthMode;
  setMode: (value: AuthMode) => void;
  username: string;
  setUsername: (value: string) => void;
  password: string;
  setPassword: (value: string) => void;
  displayName: string;
  setDisplayName: (value: string) => void;
  isSubmitting: boolean;
  isChecking: boolean;
  error: string;
  onSubmit: (event: FormEvent) => void;
}) {
  const isRegister = props.mode === "register";
  return (
    <section className="auth-layout">
      <div className="auth-copy">
        <span className="eyebrow">用户工作区</span>
        <h1>先登录，再开始你的专属模拟面试</h1>
        <p>
          每个账号会拥有独立的面试会话、短期检查点和长期记忆空间，后续可以继续扩展历史记录与报告管理。
        </p>
        <div className="signal-strip">
          <span>JWT 身份认证</span>
          <span>账号隔离会话</span>
          <span>长期记忆绑定用户</span>
        </div>
      </div>

      <form className="auth-panel" onSubmit={props.onSubmit}>
        <div className="panel-heading">
          <KeyRound size={20} />
          <div>
            <h2>{isRegister ? "创建账号" : "登录账号"}</h2>
            <p>{isRegister ? "注册后会自动进入面试工作台。" : "使用你的账号进入个人面试空间。"}</p>
          </div>
        </div>

        <div className="mode-switch" aria-label="认证模式">
          <button
            className={!isRegister ? "active" : ""}
            onClick={() => props.setMode("login")}
            type="button"
          >
            登录
          </button>
          <button
            className={isRegister ? "active" : ""}
            onClick={() => props.setMode("register")}
            type="button"
          >
            注册
          </button>
        </div>

        <label className="field">
          <span>用户名</span>
          <input
            autoComplete="username"
            minLength={3}
            maxLength={64}
            required
            value={props.username}
            onChange={(event) => props.setUsername(event.target.value)}
            placeholder="例如 guguohan"
          />
        </label>

        {isRegister && (
          <label className="field">
            <span>显示名称</span>
            <input
              autoComplete="name"
              maxLength={64}
              value={props.displayName}
              onChange={(event) => props.setDisplayName(event.target.value)}
              placeholder="可选，用于页面右上角展示"
            />
          </label>
        )}

        <label className="field">
          <span>密码</span>
          <input
            autoComplete={isRegister ? "new-password" : "current-password"}
            minLength={isRegister ? 6 : 1}
            maxLength={128}
            required
            type="password"
            value={props.password}
            onChange={(event) => props.setPassword(event.target.value)}
            placeholder={isRegister ? "至少 6 位" : "输入密码"}
          />
        </label>

        {props.error && (
          <div className="error-callout">
            <AlertCircle size={16} />
            {props.error}
          </div>
        )}

        <button className="primary-action" disabled={props.isSubmitting || props.isChecking} type="submit">
          {props.isSubmitting || props.isChecking ? <Loader2 className="spin" size={18} /> : <KeyRound size={18} />}
          {props.isChecking ? "正在检查登录状态" : isRegister ? "注册并进入" : "登录"}
        </button>
      </form>
    </section>
  );
}

function SetupPage(props: {
  jdText: string;
  setJdText: (value: string) => void;
  mode: InterviewMode;
  setMode: (value: InterviewMode) => void;
  maxFollowUps: number;
  setMaxFollowUps: (value: number) => void;
  currentUser: UserPublic;
  resumeFile: File | null;
  setResumeFile: (file: File | null) => void;
  isStarting: boolean;
  canStart: boolean;
  error: string;
  onStart: () => void;
}) {
  return (
    <section className="setup-layout">
      <div className="setup-copy">
        <span className="eyebrow">AI 模拟面试工作台</span>
        <h1>把 JD 和简历变成一场可追问的技术面试</h1>
        <p>
          系统会分析岗位要求和候选人经历，结合 RAG 题库、长期记忆和多 Agent
          工作流生成问题，并在实时对话后输出结构化评估报告。
        </p>
        <div className="signal-strip">
          <span>JD / 简历驱动</span>
          <span>RAG 个性化出题</span>
          <span>WebSocket 实时面试</span>
          <span>结构化评估报告</span>
        </div>
      </div>

      <div className="setup-panel">
        <div className="panel-heading">
          <ClipboardList size={20} />
          <div>
            <h2>创建面试会话</h2>
            <p>
              当前账号：{props.currentUser.display_name || props.currentUser.username}。粘贴目标岗位 JD，专业模式下建议上传简历。
            </p>
          </div>
        </div>

        <label className="field">
          <span>岗位 JD</span>
          <textarea
            value={props.jdText}
            onChange={(event) => props.setJdText(event.target.value)}
            rows={9}
            placeholder="粘贴目标岗位描述，包含职责、技能要求和经验要求。"
          />
        </label>

        <div className="grid-2">
          <label className="field">
            <span>每题最多追问</span>
            <input
              type="number"
              min={0}
              max={5}
              value={props.maxFollowUps}
              onChange={(event) => props.setMaxFollowUps(Number(event.target.value))}
            />
          </label>
          <div className="identity-note">
            <User size={16} />
            <span>面试记录会自动绑定当前登录账号。</span>
          </div>
        </div>

        <div className="mode-switch" aria-label="面试模式">
          <button
            className={props.mode === "professional" ? "active" : ""}
            onClick={() => props.setMode("professional")}
            type="button"
          >
            <Sparkles size={16} />
            专业面试
          </button>
          <button
            className={props.mode === "practice" ? "active" : ""}
            onClick={() => props.setMode("practice")}
            type="button"
          >
            <MessageSquareText size={16} />
            练习模式
          </button>
        </div>

        <label className="upload-box">
          <Upload size={20} />
          <input
            type="file"
            accept=".pdf,.png,.jpg,.jpeg"
            onChange={(event) => props.setResumeFile(event.target.files?.[0] ?? null)}
          />
          <span>{props.resumeFile ? props.resumeFile.name : "上传简历，可选 PDF / PNG / JPG"}</span>
        </label>

        {props.error && (
          <div className="error-callout">
            <AlertCircle size={16} />
            {props.error}
          </div>
        )}

        <button className="primary-action" disabled={!props.canStart} onClick={props.onStart}>
          {props.isStarting ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
          {props.isStarting ? "正在启动" : "开始模拟面试"}
        </button>
      </div>
    </section>
  );
}

function HistoryPage(props: {
  items: InterviewSessionSummary[];
  isLoading: boolean;
  error: string;
  onRefresh: () => void;
  onViewReport: (item: InterviewSessionSummary) => void;
  onResume: (item: InterviewSessionSummary) => void;
  onExport: (sessionId: string, format: "markdown" | "pdf") => void;
}) {
  return (
    <section className="history-layout">
      <div className="history-header">
        <div>
          <span className="eyebrow">面试记录</span>
          <h1>历史面试与报告管理</h1>
          <p>查看当前账号下的面试进度、报告结果，或继续未完成的模拟面试。</p>
        </div>
        <button className="secondary-button" onClick={props.onRefresh} type="button">
          {props.isLoading ? <Loader2 className="spin" size={16} /> : <RotateCcw size={16} />}
          刷新
        </button>
      </div>

      {props.error && (
        <div className="error-callout">
          <AlertCircle size={16} />
          {props.error}
        </div>
      )}

      {props.isLoading && !props.items.length ? (
        <div className="empty-state">
          <Loader2 className="spin" size={18} />
          正在加载历史记录
        </div>
      ) : null}

      {!props.isLoading && !props.items.length ? (
        <div className="empty-state">
          <History size={18} />
          还没有历史面试，先创建一场新的模拟面试。
        </div>
      ) : null}

      {props.items.length ? (
        <div className="history-list">
          {props.items.map((item) => (
            <article className="history-item" key={item.session_id}>
              <div className="history-main">
                <div className="history-title-row">
                  <h2>{item.title || "未命名面试"}</h2>
                  <span className={`session-status status-${item.status}`}>
                    {sessionStatusName(item.status)}
                  </span>
                </div>
                <p>{item.jd_preview}</p>
                <div className="history-meta">
                  <span>{modeName(item.mode)}</span>
                  <span>{item.assessment_count} 次回答</span>
                  {item.question_count ? <span>{item.question_count} 道题</span> : null}
                  <span>{formatSessionTime(item.updated_at || item.created_at)}</span>
                </div>
              </div>

              <div className="history-score">
                {item.overall_score ? (
                  <>
                    <strong>{item.overall_score.toFixed(1)}</strong>
                    <span>{item.grade ?? "报告"}</span>
                  </>
                ) : (
                  <>
                    <strong>{item.current_question_index || "-"}</strong>
                    <span>当前题</span>
                  </>
                )}
              </div>

              <div className="history-actions">
                {item.has_report ? (
                  <>
                    <button className="primary-small" onClick={() => props.onViewReport(item)} type="button">
                      <FileText size={16} />
                      查看报告
                    </button>
                    <button
                      className="secondary-button"
                      onClick={() => props.onExport(item.session_id, "pdf")}
                      type="button"
                    >
                      <FileText size={16} />
                      打印 PDF
                    </button>
                  </>
                ) : null}
                {item.status !== "completed" && item.status !== "failed" ? (
                  <button className="secondary-button" onClick={() => props.onResume(item)} type="button">
                    <Play size={16} />
                    继续面试
                  </button>
                ) : null}
              </div>
            </article>
          ))}
        </div>
      ) : null}
    </section>
  );
}

function sessionStatusName(status: InterviewSessionSummary["status"]) {
  const map: Record<InterviewSessionSummary["status"], string> = {
    pending: "待开始",
    analyzing: "分析中",
    interviewing: "面试中",
    evaluating: "评估中",
    completed: "已完成",
    failed: "失败",
  };
  return map[status] ?? status;
}

function formatSessionTime(value?: string | null) {
  if (!value) return "时间未知";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function openPrintableReport(blob: Blob, sessionId: string) {
  const url = URL.createObjectURL(blob);
  const win = window.open(url, `interview-report-${sessionId.slice(0, 8)}`);
  if (!win) {
    downloadBlob(blob, `interview-report-${sessionId.slice(0, 8)}.html`);
    return;
  }
  win.addEventListener("load", () => {
    win.focus();
    win.print();
  });
}

function InterviewPage(props: {
  mode: InterviewMode;
  sessionId: string;
  resume: ResumeParseResult | null;
  status: string;
  stage: string;
  error: string;
  messages: ChatItem[];
  currentTurn: CurrentTurn | null;
  progress: number;
  answer: string;
  setAnswer: (value: string) => void;
  isAwaitingAnswer: boolean;
  connection: ConnectionState;
  transcriptRef: RefObject<HTMLDivElement | null>;
  onSubmitAnswer: (event: FormEvent) => void;
  onStop: () => void;
  onReconnect: () => void;
  onBack: () => void;
}) {
  return (
    <section className="workspace">
      <aside className="sidebar">
        <button className="ghost-button" onClick={props.onBack}>
          <ArrowLeft size={16} />
          返回设置
        </button>

        <div className="side-section">
          <span className="side-label">会话</span>
          <code>{props.sessionId.slice(0, 8) || "待创建"}</code>
        </div>

        <div className="side-section">
          <span className="side-label">模式</span>
          <strong>{modeName(props.mode)}</strong>
        </div>

        <div className="progress-box">
          <div className="progress-head">
            <span>面试进度</span>
            <strong>{props.progress}%</strong>
          </div>
          <div className="progress-track">
            <div style={{ width: `${props.progress}%` }} />
          </div>
          {props.currentTurn?.questionIndex && (
            <p>
              当前第 {props.currentTurn.questionIndex}
              {props.currentTurn.totalQuestions ? ` / ${props.currentTurn.totalQuestions}` : ""} 题
            </p>
          )}
        </div>

        <div className="side-section">
          <span className="side-label">当前状态</span>
          <p>{props.status}</p>
          <small>{props.stage}</small>
        </div>

        {props.currentTurn?.skillTags?.length ? (
          <div className="side-section">
            <span className="side-label">考察技能</span>
            <div className="tag-list">
              {props.currentTurn.skillTags.map((tag) => (
                <span key={tag}>{tag}</span>
              ))}
            </div>
          </div>
        ) : null}

        {props.resume && (
          <div className="resume-mini">
            <FileText size={16} />
            <span>简历已解析，将用于项目深挖和岗位匹配提问。</span>
          </div>
        )}
      </aside>

      <div className="interview-panel">
        <div className="interview-header">
          <div>
            <h2>实时面试</h2>
            <p>
              请像真实面试一样作答。系统会根据你的回答判断是否追问、进入下一题或生成报告。
            </p>
          </div>
          <div className="interview-actions">
            {props.connection !== "open" && (
              <button className="secondary-button" onClick={props.onReconnect}>
                <RotateCcw size={16} />
                重连
              </button>
            )}
            <button className="danger-button" onClick={props.onStop}>
              <CircleStop size={16} />
              结束面试
            </button>
          </div>
        </div>

        {props.error && (
          <div className="error-callout">
            <AlertCircle size={16} />
            {props.error}
          </div>
        )}

        <div className="transcript" ref={props.transcriptRef}>
          {props.messages.map((message) => (
            <article className={`message ${message.role}`} key={message.id}>
              <div className="avatar">
                {message.role === "candidate" ? <User size={16} /> : <Bot size={16} />}
              </div>
              <div className="bubble">
                {message.meta && <span className="message-meta">{message.meta}</span>}
                <p>{message.content}</p>
              </div>
            </article>
          ))}
          {!props.isAwaitingAnswer && props.connection === "open" && (
            <div className="thinking-line">
              <Loader2 className="spin" size={16} />
              {props.status}
            </div>
          )}
        </div>

        <form className="composer" onSubmit={props.onSubmitAnswer}>
          <textarea
            value={props.answer}
            onChange={(event) => props.setAnswer(event.target.value)}
            placeholder={props.isAwaitingAnswer ? "输入你的回答..." : "等待面试官提问..."}
            rows={3}
            disabled={!props.isAwaitingAnswer || props.connection !== "open"}
          />
          <button
            className="send-button"
            disabled={!props.answer.trim() || !props.isAwaitingAnswer || props.connection !== "open"}
            type="submit"
            title="发送回答"
          >
            <Send size={18} />
          </button>
        </form>
      </div>
    </section>
  );
}

function ReportPage(props: {
  report: InterviewReport;
  mode: InterviewMode;
  sessionId: string;
  messages: ChatItem[];
  onRestart: () => void;
  onBackToInterview: () => void;
  onExport: (sessionId: string, format: "markdown" | "pdf") => void;
}) {
  const skillScores = props.report.skill_scores ?? [];
  const grade = props.report.grade ?? "?";
  const overall = props.report.overall_score ?? 0;

  return (
    <section className="report-layout">
      <div className="report-hero">
        <button className="ghost-button" onClick={props.onBackToInterview}>
          <ArrowLeft size={16} />
          返回面试记录
        </button>
        <div className="grade-lockup">
          <span className={`grade grade-${grade}`}>{grade}</span>
          <div>
            <span className="eyebrow">面试报告</span>
            <h1>{overall.toFixed(1)} / 10</h1>
            <p>{props.report.overall_assessment}</p>
          </div>
        </div>
        <div className="report-actions">
          <button
            className="secondary-button"
            onClick={() => props.onExport(props.sessionId, "markdown")}
          >
            <FileText size={16} />
            导出 Markdown
          </button>
          <button className="secondary-button" onClick={() => props.onExport(props.sessionId, "pdf")}>
            <FileText size={16} />
            打印 / 保存 PDF
          </button>
          <button className="primary-small" onClick={props.onRestart}>
            <RotateCcw size={16} />
            重新开始
          </button>
        </div>
      </div>

      <div className="report-grid">
        {props.report.hiring_recommendation && (
          <section className="report-section span-2">
            <h2>面试结论</h2>
            <p className="recommendation">{props.report.hiring_recommendation}</p>
          </section>
        )}

        {props.report.round_scores?.length ? (
          <section className="report-section">
            <h2>轮次表现</h2>
            <div className="round-list">
              {props.report.round_scores.map((round) => (
                <div className="round-item" key={round.round_name}>
                  <div>
                    <strong>{round.round_name}</strong>
                    <p>{round.summary}</p>
                  </div>
                  <span>{round.score.toFixed(1)}</span>
                </div>
              ))}
            </div>
          </section>
        ) : null}

        <section className="report-section">
          <h2>技能评分</h2>
          <div className="skill-list">
            {skillScores.map((skill) => (
              <div className="skill-item" key={skill.skill_name}>
                <div className="skill-head">
                  <strong>{skill.skill_name}</strong>
                  <span>{skill.score}/10</span>
                </div>
                <div className="score-track">
                  <div style={{ width: `${Math.max(8, skill.score * 10)}%` }} />
                </div>
                <p>{skill.evidence}</p>
              </div>
            ))}
          </div>
        </section>

        <ReportList title="优势表现" items={props.report.strengths ?? []} positive />
        <ReportList title="改进建议" items={props.report.improvements ?? []} />

        {props.report.missed_knowledge?.length ? (
          <section className="report-section span-2">
            <h2>遗漏知识点</h2>
            <div className="missed-list">
              {props.report.missed_knowledge.map((item) => (
                <details key={item.question}>
                  <summary>
                    <span>{item.question}</span>
                    <strong>{item.score}/10</strong>
                  </summary>
                  <ul>
                    {item.missed_points.map((point) => (
                      <li key={point}>{point}</li>
                    ))}
                  </ul>
                  <p>{item.reference_answer}</p>
                </details>
              ))}
            </div>
          </section>
        ) : null}

        {props.report.study_suggestions?.length ? (
          <section className="report-section span-2">
            <h2>学习建议</h2>
            <div className="suggestion-list">
              {props.report.study_suggestions.map((item) => (
                <span key={item}>
                  <CheckCircle2 size={16} />
                  {item}
                </span>
              ))}
            </div>
          </section>
        ) : null}
      </div>
    </section>
  );
}

function ReportList(props: {
  title: string;
  items: { point: string; evidence: string }[];
  positive?: boolean;
}) {
  return (
    <section className="report-section">
      <h2>{props.title}</h2>
      <div className="highlight-list">
        {props.items.map((item) => (
          <article className={props.positive ? "positive" : ""} key={`${item.point}-${item.evidence}`}>
            <strong>{item.point}</strong>
            <p>{item.evidence}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function StatusPill({ connection }: { connection: ConnectionState }) {
  const label: Record<ConnectionState, string> = {
    idle: "未连接",
    connecting: "连接中",
    open: "实时连接",
    closed: "已断开",
    error: "连接异常",
  };
  return (
    <span className={`status-pill ${connection}`}>
      <span />
      {label[connection]}
    </span>
  );
}
