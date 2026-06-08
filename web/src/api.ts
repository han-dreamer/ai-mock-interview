import type { AuthTokenResponse, InterviewMode, StartInterviewResponse, UserPublic } from "./types";

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000/api";

export const HEALTH_URL = API_BASE_URL.replace(/\/api$/, "/health");

function absoluteApiBase(): string {
  if (/^https?:\/\//i.test(API_BASE_URL)) {
    return API_BASE_URL;
  }
  return new URL(API_BASE_URL, window.location.origin).toString().replace(/\/$/, "");
}

export function websocketUrl(sessionId: string, token?: string): string {
  const wsBase = absoluteApiBase().replace(/^http:/, "ws:").replace(/^https:/, "wss:");
  const jwt = token?.trim();
  const suffix = jwt ? `?token=${encodeURIComponent(jwt)}` : "";
  return `${wsBase}/ws/interview/${sessionId}${suffix}`;
}

async function parseJson<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = payload?.detail ?? payload?.message ?? response.statusText;
    throw new Error(formatApiError(detail));
  }
  return payload as T;
}

function formatApiError(detail: unknown): string {
  if (typeof detail !== "string") {
    return JSON.stringify(detail);
  }
  const map: Record<string, string> = {
    "Authentication required.": "请先登录后再继续操作。",
    "Invalid authentication token.": "登录状态无效，请重新登录。",
    "Authentication token has expired.": "登录已过期，请重新登录。",
    "User not found or inactive.": "账号不存在或已被停用。",
    "Invalid username or password.": "用户名或密码错误。",
    "Username already exists.": "用户名已存在。",
    "Session access denied": "当前账号无权访问这个面试会话。",
  };
  return map[detail] ?? detail;
}

function authHeaders(token: string): Record<string, string> {
  return {
    Authorization: `Bearer ${token}`,
  };
}

export async function checkHealth(): Promise<{ status: string; model?: string; debug?: boolean }> {
  const response = await fetch(HEALTH_URL);
  return parseJson(response);
}

export async function register(input: {
  username: string;
  password: string;
  displayName: string;
}): Promise<AuthTokenResponse> {
  const response = await fetch(`${API_BASE_URL}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      username: input.username,
      password: input.password,
      display_name: input.displayName,
    }),
  });
  return parseJson(response);
}

export async function login(input: {
  username: string;
  password: string;
}): Promise<AuthTokenResponse> {
  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      username: input.username,
      password: input.password,
    }),
  });
  return parseJson(response);
}

export async function getMe(token: string): Promise<UserPublic> {
  const response = await fetch(`${API_BASE_URL}/auth/me`, {
    headers: authHeaders(token),
  });
  return parseJson(response);
}

export async function startInterview(input: {
  jdText: string;
  maxFollowUps: number;
  mode: InterviewMode;
  token: string;
  resumeFile?: File | null;
}): Promise<StartInterviewResponse> {
  if (input.resumeFile) {
    const form = new FormData();
    form.append("jd_text", input.jdText);
    form.append("resume_file", input.resumeFile);
    form.append("max_follow_ups", String(input.maxFollowUps));
    form.append("mode", input.mode);
    const response = await fetch(`${API_BASE_URL}/interview/start-with-resume`, {
      method: "POST",
      headers: authHeaders(input.token),
      body: form,
    });
    return parseJson(response);
  }

  const response = await fetch(`${API_BASE_URL}/interview/start`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(input.token),
    },
    body: JSON.stringify({
      jd_text: input.jdText,
      max_follow_ups: input.maxFollowUps,
      mode: input.mode,
    }),
  });
  return parseJson(response);
}
