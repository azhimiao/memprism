"""Explicit HTTP embeddings adapter. No endpoint discovery or fake fallback.

Wire format: POST the configured full embeddings URL with model, input,
dimensions and encoding_format=float. Credentials are read only when called.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Protocol, Sequence
from urllib.parse import urlsplit

import httpx


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    endpoint: str
    dimensions: int
    model: str = "text-embedding-v4"
    revision: str = "1"
    api_key_env: str = "AML_EMBEDDING_API_KEY"
    batch_size: int = 10
    timeout_seconds: float = 30.0
    max_retries: int = 0

    def __post_init__(self) -> None:
        url = urlsplit(self.endpoint)
        if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("embedding endpoint must be an explicit HTTPS URL without credentials/query/fragment")
        if not self.provider.strip() or not self.model.strip() or not self.revision.strip():
            raise ValueError("provider, model and revision must be explicit")
        if isinstance(self.dimensions, bool) or not isinstance(self.dimensions, int) or self.dimensions < 1:
            raise ValueError("embedding dimensions must be a positive integer")
        if not 1 <= self.batch_size <= 10 or not 0 <= self.max_retries <= 2:
            raise ValueError("batch_size must be 1..10 and max_retries 0..2")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("embedding timeout must be finite and positive")

    @property
    def identity(self) -> dict[str, object]:
        return {"provider": self.provider, "endpoint": self.endpoint, "model": self.model,
                "dimensions": self.dimensions, "revision": self.revision,
                "representation": "whole_raw_text_l2_float64_v1"}

    @property
    def version(self) -> str:
        return hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()


@dataclass
class EmbeddingBatch:
    vectors: list[list[float]]
    usage: dict[str, object] = field(default_factory=dict)


class EmbeddingFailure(RuntimeError):
    def __init__(self, code: str, usage: dict[str, object] | None = None):
        super().__init__(code)
        self.code = code
        self.usage = usage or {}


class EmbeddingProvider(Protocol):
    config: EmbeddingConfig
    implementation_kind: str

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch: ...


def provider_identity(provider: EmbeddingProvider) -> dict[str, object]:
    return {**provider.config.identity, "implementation_kind": provider.implementation_kind}


def provider_version(provider: EmbeddingProvider) -> str:
    return hashlib.sha256(json.dumps(provider_identity(provider), sort_keys=True).encode()).hexdigest()


def normalize_vector(vector: object, dimensions: int) -> list[float]:
    if not isinstance(vector, (list, tuple)) or len(vector) != dimensions:
        raise EmbeddingFailure("dimension_mismatch")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in vector):
        raise EmbeddingFailure("invalid_vector")
    try:
        values = [float(v) for v in vector]
    except (OverflowError, ValueError):
        raise EmbeddingFailure("invalid_vector") from None
    if any(not math.isfinite(v) for v in values):
        raise EmbeddingFailure("invalid_vector")
    # Scaling avoids overflow/underflow for finite, but very large/small inputs.
    scale = max(abs(v) for v in values)
    if scale == 0:
        raise EmbeddingFailure("zero_vector")
    scaled = [v / scale for v in values]
    norm = math.sqrt(math.fsum(v * v for v in scaled))
    return [v / norm for v in scaled]


class HttpEmbeddingProvider:
    implementation_kind = "http"

    def __init__(self, config: EmbeddingConfig, *, transport: httpx.BaseTransport | None = None):
        self.config = config
        self._transport = transport

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        return self.embed_guarded(texts)

    def embed_guarded(self, texts: Sequence[str], *, before_dispatch=None) -> EmbeddingBatch:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingFailure("invalid_input")
        key = os.environ.get(self.config.api_key_env)
        if not key:
            raise EmbeddingFailure("missing_api_key")
        calls, chars, reported_tokens = 0, 0, 0
        complete_usage = True
        vectors: list[list[float]] = []

        def usage(complete: bool) -> dict[str, object]:
            return {"requests": calls, "input_characters_sent": chars,
                    "reported_input_tokens": reported_tokens, "usage_complete": complete,
                    "monetary_cost": None}

        # No implicit proxy, redirects, SDK retries, or logging of request/response bodies.
        with httpx.Client(timeout=self.config.timeout_seconds, transport=self._transport,
                          trust_env=False, follow_redirects=False) as client:
            for start in range(0, len(texts), self.config.batch_size):
                batch = list(texts[start:start + self.config.batch_size])
                response = None
                for attempt in range(self.config.max_retries + 1):
                    if before_dispatch is not None:
                        before_dispatch()
                    calls += 1
                    chars += sum(len(text) for text in batch)
                    try:
                        response = client.post(self.config.endpoint,
                            headers={"Authorization": f"Bearer {key}"},
                            json={"model": self.config.model, "input": batch,
                                  "dimensions": self.config.dimensions, "encoding_format": "float"})
                    except httpx.TimeoutException:
                        code, retryable = "embedding_timeout", True
                    except httpx.HTTPError:
                        code, retryable = "embedding_transport_error", False
                    else:
                        if response.status_code == 200:
                            break
                        code = f"embedding_http_{response.status_code}"
                        retryable = response.status_code == 429 or response.status_code >= 500
                    complete_usage = False
                    if not retryable or attempt == self.config.max_retries:
                        raise EmbeddingFailure(code, usage(False))
                    time.sleep(0.1 * (2 ** attempt))
                try:
                    data = response.json()
                    returned_model = data.get("model", self.config.model) if isinstance(data, dict) else None
                    # OpenRouter's published catalog explicitly binds this route
                    # to this Hugging Face ID; providers use either wire name.
                    openrouter_qwen_alias = (
                        self.config.provider == "openrouter"
                        and self.config.endpoint == "https://openrouter.ai/api/v1/embeddings"
                        and self.config.model == "qwen/qwen3-embedding-8b"
                        and returned_model == "Qwen/Qwen3-Embedding-8B"
                    )
                    if not isinstance(data, dict) or (returned_model != self.config.model and not openrouter_qwen_alias):
                        raise EmbeddingFailure("response_model_mismatch")
                    items = data.get("data")
                    if not isinstance(items, list) or len(items) != len(batch):
                        raise EmbeddingFailure("response_count_mismatch")
                    indexed: dict[int, list[float]] = {}
                    for item in items:
                        index = item.get("index") if isinstance(item, dict) else None
                        if isinstance(index, bool) or not isinstance(index, int) or index not in range(len(batch)) or index in indexed:
                            raise EmbeddingFailure("invalid_response_index")
                        indexed[index] = normalize_vector(item.get("embedding"), self.config.dimensions)
                    vectors.extend(indexed[i] for i in range(len(batch)))
                    token_usage = data.get("usage", {})
                    tokens = token_usage.get("total_tokens") if isinstance(token_usage, dict) else None
                    if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0:
                        reported_tokens += tokens
                    else:
                        complete_usage = False
                except EmbeddingFailure as exc:
                    raise EmbeddingFailure(exc.code, usage(False)) from None
                except (ValueError, TypeError, KeyError, AttributeError):
                    raise EmbeddingFailure("invalid_embedding_response", usage(False)) from None
        return EmbeddingBatch(vectors, usage(complete_usage))


@dataclass(frozen=True)
class PreparedEmbedding:
    version: str
    dimensions: int
    metadata: dict[str, object]
    vectors: list[list[float]]
    status: str
    error_code: str | None
    usage: dict[str, object]


def prepare_embedding(provider: EmbeddingProvider | None, texts: Sequence[str]) -> PreparedEmbedding:
    if provider is None:
        return PreparedEmbedding("unconfigured", 0, {}, [], "unavailable", "missing_configuration", {})
    config = provider.config
    version, metadata = provider_version(provider), provider_identity(provider)
    usage: dict[str, object] = {}
    try:
        batch = provider.embed(texts)
        usage = batch.usage
        if len(batch.vectors) != len(texts):
            raise EmbeddingFailure("response_count_mismatch", batch.usage)
        vectors = [normalize_vector(v, config.dimensions) for v in batch.vectors]
        return PreparedEmbedding(version, config.dimensions, metadata, vectors, "ready", None, batch.usage)
    except EmbeddingFailure as exc:
        return PreparedEmbedding(version, config.dimensions, metadata, [], "failed", exc.code, exc.usage or usage)
