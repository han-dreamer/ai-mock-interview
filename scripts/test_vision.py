"""Quick test script to verify the vision (multi-modal) capability.

Usage:
    .venv\\Scripts\\python scripts/test_vision.py <image_path>
    .venv\\Scripts\\python scripts/test_vision.py            # runs config check only

Tests:
  1. Config validation — ensure vision settings are loaded
  2. If an image path is given, runs a real vision OCR extraction
"""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.llm.client import get_llm_client


def check_config():
    print("=" * 60)
    print("Vision Configuration Check")
    print("=" * 60)
    print(f"  vision_model     : {settings.vision_model}")
    print(f"  vision_api_key   : {'***' + settings.effective_vision_api_key[-6:] if settings.effective_vision_api_key else '(empty)'}")
    print(f"  vision_base_url  : {settings.effective_vision_base_url}")
    print(f"  llm_model        : {settings.llm_model}")
    print(f"  llm_base_url     : {settings.llm_base_url}")
    print()

    if not settings.effective_vision_api_key:
        print("WARNING: No API key configured for vision. Set VISION_API_KEY or LLM_API_KEY in .env")
        return False

    print("Config OK")
    return True


async def test_vision_ocr(image_path: str):
    path = Path(image_path)
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return

    print(f"\nTesting vision OCR on: {path.name} ({path.stat().st_size} bytes)")
    print("-" * 60)

    llm = get_llm_client()
    result = await llm.chat_vision(
        image_path=path,
        prompt="Extract all text from this image. Preserve the structure.",
    )

    print(f"Extracted text ({len(result)} chars):")
    print("-" * 60)
    print(result[:2000])
    if len(result) > 2000:
        print(f"\n... ({len(result) - 2000} more chars)")
    print("-" * 60)
    print("Vision OCR test PASSED")


async def test_resume_parse(image_path: str):
    from app.utils.file_parser import parse_resume_file

    path = Path(image_path)
    print(f"\nTesting full resume parse pipeline on: {path.name}")
    print("-" * 60)

    text = await parse_resume_file(path)
    print(f"Parsed text ({len(text)} chars):")
    print(text[:1500])
    if len(text) > 1500:
        print(f"\n... ({len(text) - 1500} more chars)")
    print("-" * 60)
    print("Resume parse test PASSED")


if __name__ == "__main__":
    ok = check_config()
    if not ok:
        sys.exit(1)

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        asyncio.run(test_vision_ocr(image_path))
        asyncio.run(test_resume_parse(image_path))
    else:
        print("\nNo image file provided. Run with an image path to test OCR:")
        print("  .venv\\Scripts\\python scripts/test_vision.py path/to/resume.png")
