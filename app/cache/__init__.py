"""Redis-backed runtime helpers.

Redis is intentionally used as an optional acceleration and coordination layer:
rate limits, short-lived locks, WebSocket presence, and lightweight snapshots.
Core interview state still lives in LangGraph checkpoints and the in-process
SessionManager during a live run.
"""
