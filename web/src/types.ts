export type InterviewMode = "practice" | "professional";

export interface UserPublic {
  id: string;
  username: string;
  display_name: string;
  role: string;
  created_at?: string | null;
  last_login_at?: string | null;
}

export interface AuthTokenResponse {
  access_token: string;
  token_type: "bearer";
  user: UserPublic;
}

export interface StartInterviewResponse {
  session_id: string;
  message: string;
  websocket_url: string;
  mode: InterviewMode;
  resume?: ResumeParseResult;
}

export interface InterviewSessionSummary {
  session_id: string;
  title: string;
  mode: InterviewMode;
  status: "pending" | "analyzing" | "interviewing" | "evaluating" | "completed" | "failed";
  graph_started: boolean;
  has_report: boolean;
  question_count: number;
  assessment_count: number;
  current_question_index: number;
  max_follow_ups: number;
  jd_preview: string;
  overall_score?: number | null;
  grade?: "A" | "B" | "C" | "D" | null;
  error_message?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  completed_at?: string | null;
}

export interface ResumeParseResult {
  normalized_text?: string;
  candidate_name?: string;
  contact?: Record<string, unknown>;
  skills?: string[];
  projects?: unknown[];
  education?: unknown[];
  work_experience?: unknown[];
  [key: string]: unknown;
}

export interface SkillScore {
  skill_name: string;
  score: number;
  evidence: string;
}

export interface ReportHighlight {
  point: string;
  evidence: string;
}

export interface RoundScore {
  round_name: string;
  score: number;
  grade: "A" | "B" | "C" | "D";
  summary: string;
}

export interface MissedKnowledge {
  question: string;
  score: number;
  missed_points: string[];
  reference_answer: string;
}

export interface InterviewReport {
  skill_scores?: SkillScore[];
  overall_score?: number;
  grade?: "A" | "B" | "C" | "D";
  strengths?: ReportHighlight[];
  improvements?: ReportHighlight[];
  overall_assessment?: string;
  round_scores?: RoundScore[];
  technical_depth_score?: number;
  technical_breadth_score?: number;
  hiring_recommendation?: string;
  total_questions?: number;
  missed_knowledge?: MissedKnowledge[];
  study_suggestions?: string[];
}

export type ServerMessage =
  | {
      type: "status";
      stage: string;
      message: string;
    }
  | {
      type: "question";
      question_index: number;
      total_questions: number;
      content: string;
      skill_tags: string[];
      difficulty: string;
    }
  | {
      type: "follow_up";
      question_index: number;
      follow_up_number: number;
      content: string;
    }
  | {
      type: "interview_end";
      message: string;
    }
  | {
      type: "report";
      data: InterviewReport;
    }
  | {
      type: "error";
      message: string;
    }
  | {
      type: "stream_chunk";
      chunk: string;
      done: boolean;
    };

export interface ChatItem {
  id: string;
  role: "interviewer" | "candidate" | "system";
  content: string;
  meta?: string;
}

export interface CurrentTurn {
  questionIndex?: number;
  totalQuestions?: number;
  skillTags?: string[];
  difficulty?: string;
  followUpNumber?: number;
}
