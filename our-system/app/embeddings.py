"""Explicit OpenAI-compatible and native DashScope v4 HTTP adapters.

No endpoint discovery, input truncation, model fallback or inline credentials.
Native v4 defaults to document semantics; use for_query() for retrieval.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol, Sequence
from urllib.parse import urlsplit

import httpx


_DASHSCOPE_PATH = "/api/v1/services/embeddings/text-embedding/text-embedding"
_DASHSCOPE_HOSTS = {
    "dashscope.aliyuncs.com": "cn-beijing",
    "dashscope-intl.aliyuncs.com": "ap-southeast-1",
    "dashscope-us.aliyuncs.com": "us-east-1",
    "cn-hongkong.dashscope.aliyuncs.com": "cn-hongkong",
}
_WORKSPACE_HOST = re.compile(
    r"[a-z0-9][a-z0-9-]*\.(cn-beijing|ap-southeast-1|us-east-1|"
    r"cn-hongkong|eu-central-1|ap-northeast-1)\.maas\.aliyuncs\.com"
)


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
    region: str | None = None
    max_input_bytes: int = 8192
    min_request_interval_seconds: float = 0.0

    def __post_init__(self) -> None:
        url = urlsplit(self.endpoint)
        if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("embedding endpoint must be an explicit HTTPS URL without credentials/query/fragment")
        if not self.provider.strip() or not self.model.strip() or not self.revision.strip():
            raise ValueError("provider, model and revision must be explicit")
        if isinstance(self.dimensions, bool) or not isinstance(self.dimensions, int) or self.dimensions < 1:
            raise ValueError("embedding dimensions must be a positive integer")
        if (isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int)
                or isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int)
                or not 1 <= self.batch_size <= 10 or not 0 <= self.max_retries <= 2):
            raise ValueError("batch_size must be 1..10 and max_retries 0..2")
        if (isinstance(self.timeout_seconds, bool) or not math.isfinite(self.timeout_seconds)
                or self.timeout_seconds <= 0):
            raise ValueError("embedding timeout must be finite and positive")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env):
            raise ValueError("api_key_env must be an environment variable name")
        if (isinstance(self.min_request_interval_seconds, bool)
                or not math.isfinite(self.min_request_interval_seconds)
                or not 0 <= self.min_request_interval_seconds <= 60):
            raise ValueError("request interval must be finite and in 0..60 seconds")
        if self.provider == "dashscope":
            matched = _WORKSPACE_HOST.fullmatch(url.hostname)
            region = _DASHSCOPE_HOSTS.get(url.hostname) or (matched.group(1) if matched else None)
            if (url.path != _DASHSCOPE_PATH or url.port not in (None, 443) or region is None
                    or url.hostname.startswith("trial.")):
                raise ValueError("dashscope requires an explicit official native embeddings endpoint")
            if self.region is not None and self.region != region:
                raise ValueError("embedding region does not match endpoint")
            object.__setattr__(self, "region", region)
            if self.model != "text-embedding-v4" or self.dimensions not in (64, 128, 256, 512, 768, 1024, 1536, 2048):
                raise ValueError("dashscope v4 requires its exact model name and a supported dimension")
            if (isinstance(self.max_input_bytes, bool) or not isinstance(self.max_input_bytes, int)
                    or not 1 <= self.max_input_bytes <= 8192):
                raise ValueError("v4 conservative UTF-8 input byte limit must be 1..8192")

    @property
    def identity(self) -> dict[str, object]:
        identity = {"provider": self.provider, "endpoint": self.endpoint, "model": self.model,
                "dimensions": self.dimensions, "revision": self.revision,
                "representation": "whole_raw_text_l2_float64_v1"}
        # Preserve the legacy adapter's persisted identity exactly.
        if self.provider == "dashscope":
            identity.update(protocol="dashscope_text_embedding_v1", region=self.region,
                text_types={"ingest": "document", "retrieval": "query"}, output_type="dense",
                preprocessing="exact_utf8_no_truncation_v1", provider_tokenizer="not_exposed",
                provider_max_input_tokens=8192, input_limit_policy="conservative_utf8_bytes_v1",
                max_input_bytes=self.max_input_bytes, instruct=None)
        return identity

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
        if config.provider == "dashscope":
            self.implementation_kind = "dashscope_http_v1"
        self._rate_lock = threading.Lock()
        self._next_dispatch = 0.0

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        return self.embed_guarded(texts)

    def for_query(self) -> QueryEmbeddingProvider | HttpEmbeddingProvider:
        if self.config.provider != "dashscope":
            return self  # OpenAI-compatible embeddings have no separate query role.
        return QueryEmbeddingProvider(self)

    def embed_guarded(self, texts: Sequence[str], *, before_dispatch=None,
                      text_type: str = "document") -> EmbeddingBatch:
        native = self.config.provider == "dashscope"
        if text_type not in {"document", "query"} or (not native and text_type != "document"):
            raise EmbeddingFailure("invalid_text_type")
        if (isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence)
                or not texts or any(not isinstance(text, str) or not text.strip() for text in texts)):
            raise EmbeddingFailure("invalid_input")
        texts = tuple(texts)  # Validate every batch before the first billable dispatch.
        if native:
            try:
                lengths = [len(text.encode("utf-8")) for text in texts]
            except UnicodeEncodeError:
                raise EmbeddingFailure("invalid_input_unicode") from None
            if any(length > self.config.max_input_bytes for length in lengths):
                raise EmbeddingFailure("input_too_long")
        key = os.environ.get(self.config.api_key_env)
        if not key or not key.strip():
            raise EmbeddingFailure("missing_api_key")
        if any(ord(char) < 33 or ord(char) > 126 for char in key):
            raise EmbeddingFailure("invalid_api_key")
        calls, chars, reported_tokens = 0, 0, 0
        complete_usage = True
        vectors: list[list[float]] = []
        request_ids: list[str] = []
        returned_models: list[str] = []

        def usage(complete: bool) -> dict[str, object]:
            result = {"requests": calls, "input_characters_sent": chars,
                    "reported_input_tokens": reported_tokens, "usage_complete": complete,
                    "monetary_cost": None}
            if native:
                result.update(request_ids=list(request_ids), response_models=list(returned_models),
                    model_identity_source="response" if calls and len(returned_models) == calls else "request_only_or_partial",
                    text_type=text_type)
            return result

        def record_usage(data):
            nonlocal reported_tokens, complete_usage
            token_usage = data.get("usage", {}) if isinstance(data, dict) else {}
            tokens = token_usage.get("total_tokens") if isinstance(token_usage, dict) else None
            if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0:
                reported_tokens += tokens
            else:
                complete_usage = False
            if native and isinstance(data, dict):
                request_id = data.get("request_id")
                if isinstance(request_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id):
                    request_ids.append(request_id)
                if data.get("model") == self.config.model:
                    returned_models.append(self.config.model)

        def provider_error(data):
            if not isinstance(data, dict):
                return "invalid_embedding_response", False
            code = data.get("code")
            if code is not None and (not isinstance(code, str) or code):
                # Do not expose the supplier's free-form code/message/body.
                if isinstance(code, str) and (code.startswith("Throttling") or code in {"TooManyRequests", "ServiceUnavailable", "InternalError"}):
                    return "embedding_provider_unavailable", True
                return "embedding_provider_error", False
            status = data.get("status_code", 200)
            if isinstance(status, bool) or not isinstance(status, int) or status != 200:
                return "embedding_provider_error", False
            if "error" in data:
                return "embedding_provider_error", False
            return None

        # No implicit proxy, redirects, SDK retries, or logging of request/response bodies.
        with httpx.Client(timeout=self.config.timeout_seconds, transport=self._transport,
                          trust_env=False, follow_redirects=False) as client:
            for start in range(0, len(texts), self.config.batch_size):
                batch = list(texts[start:start + self.config.batch_size])
                payload = ({"model": self.config.model, "input": {"texts": batch},
                            "parameters": {"dimension": self.config.dimensions,
                                           "text_type": text_type, "output_type": "dense"}}
                           if native else {"model": self.config.model, "input": batch,
                               "dimensions": self.config.dimensions, "encoding_format": "float"})
                response = None
                data = None
                for attempt in range(self.config.max_retries + 1):
                    if native and self.config.min_request_interval_seconds:
                        # Shared by document/query views; bounded, fail-fast admission.
                        with self._rate_lock:
                            now = time.monotonic()
                            if now < self._next_dispatch:
                                raise EmbeddingFailure("embedding_rate_limited", usage(complete_usage))
                            self._next_dispatch = now + self.config.min_request_interval_seconds
                    if before_dispatch is not None:
                        before_dispatch()
                    calls += 1
                    chars += sum(len(text) for text in batch)
                    try:
                        response = client.post(self.config.endpoint,
                            headers={"Authorization": f"Bearer {key}"},
                            json=payload)
                    except httpx.TimeoutException:
                        code, retryable = "embedding_timeout", True
                    except httpx.HTTPError:
                        code, retryable = "embedding_transport_error", False
                    else:
                        if response.status_code == 200:
                            try:
                                data = response.json()
                            except ValueError:
                                raise EmbeddingFailure("invalid_embedding_response", usage(False)) from None
                            record_usage(data)
                            error = provider_error(data) if native else None
                            if error is None:
                                break
                            code, retryable = error
                        else:
                            code = f"embedding_http_{response.status_code}"
                            retryable = response.status_code == 429 or response.status_code >= 500
                    complete_usage = False
                    if not retryable or attempt == self.config.max_retries:
                        raise EmbeddingFailure(code, usage(False))
                    delay = 0.1 * (2 ** attempt)
                    if response is not None:
                        try:
                            delay = max(delay, min(1.0, float(response.headers.get("Retry-After", "0"))))
                        except ValueError:
                            pass
                    time.sleep(delay)
                try:
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
                    output = data.get("output")
                    items = output.get("embeddings") if native and isinstance(output, dict) else data.get("data") if not native else None
                    if not isinstance(items, list) or len(items) != len(batch):
                        raise EmbeddingFailure("response_count_mismatch")
                    indexed: dict[int, list[float]] = {}
                    for item in items:
                        index = item.get("text_index" if native else "index") if isinstance(item, dict) else None
                        if isinstance(index, bool) or not isinstance(index, int) or index not in range(len(batch)) or index in indexed:
                            raise EmbeddingFailure("invalid_response_index")
                        indexed[index] = normalize_vector(item.get("embedding"), self.config.dimensions)
                    vectors.extend(indexed[i] for i in range(len(batch)))
                except EmbeddingFailure as exc:
                    raise EmbeddingFailure(exc.code, usage(False)) from None
                except (ValueError, TypeError, KeyError, AttributeError):
                    raise EmbeddingFailure("invalid_embedding_response", usage(False)) from None
        return EmbeddingBatch(vectors, usage(complete_usage))


class QueryEmbeddingProvider:
    """Immutable query role over the same model identity and dispatch limiter."""

    def __init__(self, provider: HttpEmbeddingProvider):
        self._provider = provider
        self.config = provider.config
        self.implementation_kind = provider.implementation_kind

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        return self.embed_guarded(texts)

    def embed_guarded(self, texts: Sequence[str], *, before_dispatch=None) -> EmbeddingBatch:
        return self._provider.embed_guarded(texts, before_dispatch=before_dispatch, text_type="query")


class DashScopeEmbeddingProvider(HttpEmbeddingProvider):
    """Native text-embedding-v4; intentionally excludes compatible-mode URLs."""

    implementation_kind = "dashscope_http_v1"

    def __init__(self, config: EmbeddingConfig, *, transport: httpx.BaseTransport | None = None):
        if config.provider != "dashscope":
            raise ValueError("DashScopeEmbeddingProvider requires provider=dashscope")
        super().__init__(config, transport=transport)


def create_embedding_provider(config: EmbeddingConfig, *,
                              transport: httpx.BaseTransport | None = None) -> HttpEmbeddingProvider:
    """Factory for API injection; other provider labels retain the legacy wire contract.

    The returned provider embeds documents. Inject provider.for_query() into the
    retrieval service for native v4; both views have the same persisted identity.
    """
    cls = DashScopeEmbeddingProvider if config.provider == "dashscope" else HttpEmbeddingProvider
    return cls(config, transport=transport)


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
