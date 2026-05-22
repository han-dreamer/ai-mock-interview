"""LangChain-compatible embeddings for RAGAS evaluation."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from langchain_core.embeddings import Embeddings
from openai import AsyncOpenAI, OpenAI

from app.config import settings


class OpenAICompatibleTextEmbeddings(Embeddings):
    """Small embedding adapter for OpenAI-compatible text embedding endpoints.

    RAGAS can pass generated intermediate questions through LangChain's embedding
    abstraction. Some OpenAI-compatible providers are strict about the `input`
    payload and reject non-plain text values after LangChain preprocessing. This
    adapter keeps the request shape simple: `input` is always `list[str]`.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 10,
        timeout: float = 180,
        max_retries: int = 2,
    ) -> None:
        self.model = model or settings.embedding_model
        self.batch_size = batch_size
        self._client = OpenAI(
            api_key=api_key or settings.effective_embedding_api_key,
            base_url=base_url or settings.effective_embedding_base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._async_client = AsyncOpenAI(
            api_key=api_key or settings.effective_embedding_api_key,
            base_url=base_url or settings.effective_embedding_base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        normalized = [_normalize_text(text) for text in texts]
        embeddings: list[list[float]] = []
        for batch in _batched(normalized, self.batch_size):
            response = self._client.embeddings.create(
                model=self.model,
                input=batch,
            )
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        normalized = [_normalize_text(text) for text in texts]
        tasks = [
            self._async_client.embeddings.create(model=self.model, input=batch)
            for batch in _batched(normalized, self.batch_size)
        ]
        responses = await asyncio.gather(*tasks)
        embeddings: list[list[float]] = []
        for response in responses:
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

    async def aembed_query(self, text: str) -> list[float]:
        result = await self.aembed_documents([text])
        return result[0]


def _normalize_text(value: Any) -> str:
    if isinstance(value, str):
        text = value
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        text = "\n".join(value)
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)

    text = re.sub(r"\s+", " ", text).strip()
    return text or " "


def _batched(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]
