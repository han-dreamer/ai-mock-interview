"""Inspect long-term memory stored by the interview system.

Usage:
    python scripts/inspect_memory.py --user-id local-user
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.memory.service import get_memory_service  # noqa: E402


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", default="local-user")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument(
        "--query",
        default="",
        help="Optional semantic query for long-term memory recall.",
    )
    args = parser.parse_args()

    memory = get_memory_service()
    context = await memory.abuild_context(args.user_id, semantic_query=args.query)
    print(memory.format_context(context))

    print("\n## Skill Memories")
    for skill in memory.store.list_skill_memories(args.user_id, limit=args.limit):
        weak = "; ".join(skill.weak_points[:4]) or "n/a"
        print(
            f"- {skill.skill_name}: recent={skill.recent_score:.1f}, "
            f"avg={skill.avg_score:.1f}, attempts={skill.attempts}, "
            f"level={skill.mastery_level}, priority={skill.next_practice_priority:.2f}, "
            f"weak={weak}"
        )


if __name__ == "__main__":
    asyncio.run(main())
