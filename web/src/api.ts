import type { InterviewMode, StartInterviewResponse } from "./types";

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000/api";

export const HEALTH_URL = API_BASE_URL.replace(/\/api$/, "/health");

function absoluteApiBase(): string {
  if (/^https?:\/\//i.test(API_BASE_URL)) {
    return API_BASE_URL;
  }
  return new URL(API_BASE_URL, window.location.origin).toString().replace(/\/$/, "");
}

export function websocketUrl(sessionId: string, accessCode?: string): string {
  const wsBase = absoluteApiBase().replace(/^http:/, "ws:").replace(/^https:/, "wss:");
  const token = accessCode?.trim();
  const suffix = token ? `?access_token=${encodeURIComponent(token)}` : "";
  return `${wsBase}/ws/interview/${sessionId}${suffix}`;
}

async function parseJson<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = payload?.detail ?? payload?.message ?? response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload as T;
}

export async function checkHealth(): Promise<{ status: string; model?: string; debug?: boolean }> {
  const response = await fetch(HEALTH_URL);
  return parseJson(response);
}

export async function startInterview(input: {
  jdText: string;
  maxFollowUps: number;
  mode: InterviewMode;
  userId: string;
  accessCode?: string;
  resumeFile?: File | null;
}): Promise<StartInterviewResponse> {
  const accessHeaders: Record<string, string> = {};
  if (input.accessCode?.trim()) {
    accessHeaders["X-Access-Code"] = input.accessCode.trim();
  }
  if (input.resumeFile) {
    const form = new FormData();
    form.append("jd_text", input.jdText);
    form.append("resume_file", input.resumeFile);
    form.append("max_follow_ups", String(input.maxFollowUps));
    form.append("mode", input.mode);
    form.append("user_id", input.userId);
    const response = await fetch(`${API_BASE_URL}/interview/start-with-resume`, {
      method: "POST",
      headers: accessHeaders,
      body: form,
    });
    return parseJson(response);
  }

  const response = await fetch(`${API_BASE_URL}/interview/start`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...accessHeaders,
    },
    body: JSON.stringify({
      jd_text: input.jdText,
      max_follow_ups: input.maxFollowUps,
      mode: input.mode,
      user_id: input.userId,
    }),
  });
  return parseJson(response);
}
