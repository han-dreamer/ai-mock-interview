"""Unified async LLM client supporting chat / streaming / structured / vision output."""

from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path
from typing import Any, AsyncIterator, TypeVar, get_args, get_origin

from openai import AsyncOpenAI
from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _build_field_description(model: type[BaseModel], indent: int = 2) -> str:
    """Build a human-readable field list, recursively expanding nested models."""
    prefix = " " * indent
    lines: list[str] = []
    for name, field_info in model.model_fields.items():
        annotation = field_info.annotation
        desc = field_info.description or ""
        required = field_info.is_required()
        req_tag = "required" if required else "optional"

        inner_model = _get_inner_basemodel(annotation)
        if inner_model:
            is_list = get_origin(annotation) is list or (
                get_origin(annotation) is not None
                and any(get_origin(a) is list for a in get_args(annotation) if a is not type(None))
            )
            if is_list or (get_origin(annotation) is list):
                nested_desc = _build_field_description(inner_model, indent + 4)
                lines.append(f'{prefix}"{name}": [ {nested_desc}, ... ] ({req_tag})')
            else:
                nested_desc = _build_field_description(inner_model, indent + 4)
                lines.append(f'{prefix}"{name}": {nested_desc} ({req_tag})')
        else:
            type_str = _annotation_to_str(annotation)
            line = f'{prefix}"{name}": {type_str} ({req_tag})'
            if desc:
                line += f"  // {desc}"
            lines.append(line)

    return "{\n" + "\n".join(lines) + "\n" + " " * (indent - 2) + "}"


def _get_inner_basemodel(annotation: Any) -> type[BaseModel] | None:
    """If annotation is or contains a BaseModel subclass, return it."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    origin = get_origin(annotation)
    if origin is list:
        args = get_args(annotation)
        if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
            return args[0]
    # Handle Union types (e.g. X | None)
    args = get_args(annotation)
    if args:
        for a in args:
            if isinstance(a, type) and issubclass(a, BaseModel):
                return a
            if get_origin(a) is list:
                inner_args = get_args(a)
                if inner_args and isinstance(inner_args[0], type) and issubclass(inner_args[0], BaseModel):
                    return inner_args[0]
    return None


def _annotation_to_str(annotation: Any) -> str:
    """Convert a type annotation to a readable string like 'list[str]'."""
    origin = get_origin(annotation)
    if origin is list:
        args = get_args(annotation)
        inner = _annotation_to_str(args[0]) if args else "any"
        return f"list[{inner}]"
    if origin is None:
        if annotation is None:
            return "null"
        name = getattr(annotation, "__name__", str(annotation))
        return name.replace("NoneType", "null")
    # Union types (e.g. str | None)
    args = get_args(annotation)
    if args:
        parts = [_annotation_to_str(a) for a in args]
        return " | ".join(parts)
    return str(annotation)


class LLMClient:
    """Thin wrapper around the OpenAI-compatible async client.

    Supports any provider (OpenAI / DeepSeek / Qwen / Ollama …) via base_url.
    Includes multi-modal vision support via a separate vision client.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        self._api_key = api_key or settings.llm_api_key
        self._base_url = base_url or settings.llm_base_url
        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature

        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )

        self._vision_client = AsyncOpenAI(
            api_key=settings.effective_vision_api_key,
            base_url=settings.effective_vision_base_url,
        )
        self.vision_model = settings.vision_model

    # ------------------------------------------------------------------
    # Normal chat completion
    # ------------------------------------------------------------------
    async def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
        )
        content = response.choices[0].message.content or ""
        logger.debug("LLM response (%d chars): %s…", len(content), content[:120])
        return content

    # ------------------------------------------------------------------
    # Streaming chat completion
    # ------------------------------------------------------------------
    async def chat_stream(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ------------------------------------------------------------------
    # Multi-modal vision completion
    # ------------------------------------------------------------------
    async def chat_vision(
        self,
        image_path: str | Path,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Send an image + text prompt to a vision-capable LLM.

        Encodes the image as base64 and uses the OpenAI vision API format.
        Uses the dedicated vision client/model from config.
        """
        image_path = Path(image_path)
        suffix = image_path.suffix.lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
        mime_type = mime_map.get(suffix, "image/png")

        image_bytes = image_path.read_bytes()
        b64_data = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64_data}"

        logger.info(
            "Vision request: %s (%d bytes, %s)",
            image_path.name, len(image_bytes), model or self.vision_model,
        )

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ],
        })

        response = await self._vision_client.chat.completions.create(
            model=model or self.vision_model,
            messages=messages,
            temperature=temperature if temperature is not None else 0.1,
        )

        content = response.choices[0].message.content or ""
        logger.info("Vision response: %d chars", len(content))
        return content

    # ------------------------------------------------------------------
    # Structured output — force JSON conforming to a Pydantic model
    # ------------------------------------------------------------------
    MAX_STRUCTURED_RETRIES = 3

    async def chat_structured(
        self,
        messages: list[dict],
        response_model: type[T],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> T:
        """Return a validated Pydantic instance parsed from LLM JSON output.

        Retries up to MAX_STRUCTURED_RETRIES times if the LLM returns invalid
        JSON (e.g. echoes back the schema definition instead of real data).
        """
        field_desc = _build_field_description(response_model)

        system_suffix = (
            "\n\n## OUTPUT FORMAT\n"
            "You MUST respond with ONLY a valid JSON object (no markdown, no extra text).\n"
            f"The JSON must have these fields:\n{field_desc}\n\n"
            "LANGUAGE RULE: For every user-visible free-text value in the JSON, "
            "use Simplified Chinese. Keep technical names such as FastAPI, LangGraph, "
            "RAG, WebSocket, StateGraph, React, Redis, Docker and API in English when "
            "appropriate, but write explanations, evidence, summaries, suggestions, "
            "questions and follow-up directions in Chinese.\n\n"
            "IMPORTANT: Fill in actual values based on your analysis. "
            "Do NOT return the schema definition or type descriptions — return real data only."
        )

        last_error: Exception | None = None

        for attempt in range(1, self.MAX_STRUCTURED_RETRIES + 1):
            patched = list(messages)
            if attempt > 1:
                retry_suffix = (
                    system_suffix
                    + f"\n\nWARNING: Your previous response was invalid JSON. "
                    f"You returned a schema/type definition instead of actual data. "
                    f"Please return a JSON object with REAL VALUES this time. "
                    f"For example, 'score' should be a number like 7, not a type description."
                )
            else:
                retry_suffix = system_suffix

            if patched and patched[0]["role"] == "system":
                patched[0] = {
                    **patched[0],
                    "content": patched[0]["content"] + retry_suffix,
                }
            else:
                patched.insert(0, {"role": "system", "content": retry_suffix})

            try:
                response = await self._client.chat.completions.create(
                    model=model or self.model,
                    messages=patched,
                    temperature=temperature if temperature is not None else self.temperature,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content or "{}"
            except Exception:
                logger.warning("json_object mode not supported, falling back to plain chat")
                raw = await self.chat(patched, model=model, temperature=temperature)

            raw = self._clean_json_response(raw)

            if self._looks_like_schema(raw):
                logger.warning(
                    "Attempt %d/%d: LLM returned schema definition instead of data, retrying...",
                    attempt, self.MAX_STRUCTURED_RETRIES,
                )
                continue

            try:
                result = response_model.model_validate_json(raw)
                if attempt > 1:
                    logger.info("Structured output succeeded on retry #%d", attempt)
                return result
            except Exception as e:
                last_error = e
                logger.warning(
                    "Attempt %d/%d: validation failed (%s), retrying...",
                    attempt, self.MAX_STRUCTURED_RETRIES, e,
                )

        raise last_error or ValueError("chat_structured failed after all retries")

    @staticmethod
    def _looks_like_schema(raw: str) -> bool:
        """Detect if the LLM returned a JSON Schema definition instead of data."""
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return False
        if not isinstance(obj, dict):
            return False
        schema_keys = {"type", "properties", "description", "$defs", "$schema", "definitions"}
        return bool(schema_keys & set(obj.keys()) and "type" in obj and obj["type"] == "object")

    @staticmethod
    def _clean_json_response(raw: str) -> str:
        """Strip markdown fences and extract the JSON object from LLM output."""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        # Some models wrap JSON in extra text; try to extract the outermost { }
        if not raw.startswith("{"):
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                raw = match.group(0)

        return raw


_default_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
