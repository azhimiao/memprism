"""Optional, bounded reranking of an already authorized EvidenceCandidate pool.

Ranking scores live in the trace; candidates and their retrieval scores remain
intact. A failed or incomplete attempt always returns the original order.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from numbers import Real
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

from app.jobs import SnapshotChanged
from app.models import EvidenceCandidate

MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MODEL_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
LOCAL_VERSIONS = {"sentence-transformers": "5.1.2", "transformers": "4.57.1", "torch": "2.8.0"}
# Hugging Face revision metadata: small files use Git blob IDs; LFS files
# expose the actual content SHA-256. No weights are fetched to obtain these.
PINNED_ARTIFACTS = {
    "config.json": ("git_blob", "43ac97f1bef372d4f1d30a32f9a84baffcf6575d"),
    "tokenizer_config.json": ("git_blob", "b59159d44c6bd4b1d9f7ae4f8da028da4238e06e"),
    "special_tokens_map.json": ("git_blob", "2ea7ad0e45a9d1d1591782ba7e29a703d0758831"),
    "model.safetensors": ("sha256", "5daeca2481a76b5976a2bdc32f0a78532b6716da4f8cd3ff59460ef8d2f359b4"),
    "tokenizer.json": ("sha256", "62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626"),
    "sentencepiece.bpe.model": ("sha256", "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865"),
}


class Scorer(Protocol):
    def __call__(self, query: str, documents: Sequence[str]) -> Sequence[float]: ...


PairTokenCounter = Callable[[str, str], int]


class RerankFailure(RuntimeError):
    def __init__(self, code: str, usage: dict | None = None):
        super().__init__(code)
        self.code = code
        self.usage = usage or {}


@dataclass(frozen=True)
class RerankConfig:
    enabled: bool = False
    local_model_path: Path | None = None
    max_length: int = 512
    max_query_tokens: int = 128  # Includes pair special tokens with an empty document.
    device: str = "cpu"  # Explicit CPU or cuda:N; never automatic device selection.
    batch_size: int = 8
    cpu_threads: int = 2
    window_chars: int = 1200
    overlap_chars: int = 64
    max_candidates: int = 200
    max_candidate_chars: int = 64_000
    max_total_chars: int = 256_000
    max_query_chars: int = 4096
    max_windows_per_candidate: int = 64
    max_windows: int = 256
    max_pair_tokens: int = 131_072
    timeout_seconds: float = 2.0
    startup_timeout_seconds: float = 60.0
    shutdown_timeout_seconds: float = 1.0

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be boolean")
        integers = ("max_length", "max_query_tokens", "batch_size", "cpu_threads", "window_chars",
                    "max_candidates", "max_candidate_chars", "max_total_chars", "max_query_chars",
                    "max_windows_per_candidate", "max_windows", "max_pair_tokens")
        for name in integers:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 8 <= self.max_length <= 512 or self.max_query_tokens >= self.max_length:
            raise ValueError("max_length must be 8..512 and exceed max_query_tokens")
        if self.max_candidates > 200 or self.batch_size > 64 or self.cpu_threads > 64:
            raise ValueError("candidate, batch or CPU thread budget exceeded")
        if (isinstance(self.overlap_chars, bool) or not isinstance(self.overlap_chars, int)
                or not 0 <= self.overlap_chars < self.window_chars):
            raise ValueError("overlap_chars must be 0..window_chars-1")
        for name in ("timeout_seconds", "startup_timeout_seconds", "shutdown_timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.device, str) or not re.fullmatch(r"cpu|cuda:\d+", self.device):
            raise ValueError("device must be cpu or cuda:N")
        if self.local_model_path is not None:
            path = Path(self.local_model_path)
            if not path.is_absolute():
                raise ValueError("local_model_path must be absolute")
            object.__setattr__(self, "local_model_path", path)

    @property
    def identity(self) -> dict:
        settings = asdict(self)
        settings["local_model_path"] = str(self.local_model_path) if self.local_model_path is not None else None
        return {**settings, "model": MODEL_ID, "revision": MODEL_REVISION,
                "license": "apache-2.0", "dtype": "float32", "backend": "torch",
                "local_versions": dict(LOCAL_VERSIONS), "activation": "identity_logits",
                "expected_artifacts": dict(PINNED_ARTIFACTS),
                "seed": 0, "deterministic_algorithms": True, "attention": "eager",
                "aggregation": "window_max_source_max_component_max_v1",
                "window_policy": "exact_unicode_cover_shrink_overlap_v1", "retries": 0}


def _byte_pair_tokens(query: str, document: str) -> int:
    """Deterministic offline surrogate, not the local model's tokenizer."""
    return len(query.encode("utf-8")) + len(document.encode("utf-8")) + 4


def _failure_code(exc: Exception, default: str = "scoring_failed") -> str:
    if isinstance(exc, RerankFailure):
        return exc.code
    if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
        return "oom"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ImportError):
        return "dependency_missing"
    return default  # Never expose exception text (which can contain source text).


def _usage() -> dict:
    return {"windows_planned": 0, "pair_tokens_planned": 0, "batches_started": 0,
            "windows_scored": 0, "coverage_complete": False, "usage_complete": True}


def _batch_revocation(dispatched: int, cancelled: int, *, unit: str) -> dict:
    """Queued batches are not sent. A batch already handed to the scorer cannot be recalled."""
    return {"stage": "scorer_batch" if unit == "scorer_batch" else "before_worker_request",
            "batches_dispatched": dispatched, "batches_cancelled": cancelled,
            "queued_not_sent": cancelled > 0, "dispatched_results_discarded": dispatched > 0,
            "inflight_recall": False, "dispatch_unit": unit}


def _check_deadline(deadline: float, cancelled: threading.Event | None = None):
    if (cancelled is not None and cancelled.is_set()) or time.monotonic() >= deadline:
        raise RerankFailure("timeout")


def _validate_scores(values, expected: int) -> list[float]:
    try:
        if len(values) != expected:
            raise RerankFailure("invalid_scores")
        result = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise RerankFailure("invalid_scores")
            result.append(float(value))
        if len(result) != expected:
            raise RerankFailure("invalid_scores")
        return result
    except (TypeError, ValueError, OverflowError):
        raise RerankFailure("invalid_scores") from None


def _evaluate(query: str, documents: Sequence[str], config: RerankConfig, scorer: Scorer,
              counter: PairTokenCounter, deadline: float, usage: dict,
              cancelled: threading.Event | None = None, authorize=None) -> dict:
    def count(document):
        _check_deadline(deadline, cancelled)
        value = counter(query, document)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RerankFailure("invalid_token_count")
        return value

    if count("") > config.max_query_tokens:
        raise RerankFailure("query_budget_exceeded")
    windows = []
    for index, document in enumerate(documents):
        start, count_for_candidate = 0, 0
        while start < len(document):
            end = min(len(document), start + config.window_chars)
            tokens = count(document[start:end])
            # Counts need not be monotone under Unicode/BPE. Recount every actual
            # substring; do not decode tokens into a replacement source string.
            while tokens > config.max_length:
                if end - start == 1:
                    raise RerankFailure("window_cannot_fit")
                end = start + max(1, (end - start) * 3 // 4)
                tokens = count(document[start:end])
            count_for_candidate += 1
            if count_for_candidate > config.max_windows_per_candidate or len(windows) >= config.max_windows:
                raise RerankFailure("window_budget_exceeded")
            if usage["pair_tokens_planned"] + tokens > config.max_pair_tokens:
                raise RerankFailure("token_budget_exceeded")
            windows.append({"candidate_index": index, "start": start, "end": end, "pair_tokens": tokens})
            usage["windows_planned"] += 1
            usage["pair_tokens_planned"] += tokens
            if end == len(document):
                break
            overlap = min(config.overlap_chars, (end - start) // 4)
            start = end - overlap

    # Preflight every window before the first score call. Never rank a partially
    # covered parent as if it had been fully assessed.
    candidate_scores = [-math.inf] * len(documents)
    offsets = range(0, len(windows), config.batch_size)
    for position, offset in enumerate(offsets):
        _check_deadline(deadline, cancelled)
        batch = windows[offset:offset + config.batch_size]
        if authorize is not None:
            try:
                authorize([window["candidate_index"] for window in batch])
            except SnapshotChanged as exc:
                exc.dispatch_trace = _batch_revocation(usage["batches_started"], len(offsets) - position,
                                                       unit="scorer_batch")
                raise
        texts = [documents[w["candidate_index"]][w["start"]:w["end"]] for w in batch]
        usage["batches_started"] += 1
        scores = _validate_scores(scorer(query, texts), len(batch))
        _check_deadline(deadline, cancelled)
        for window, score in zip(batch, scores):
            window["score"] = score
            index = window["candidate_index"]
            candidate_scores[index] = max(candidate_scores[index], score)
        usage["windows_scored"] += len(batch)
    usage["coverage_complete"] = True
    return {"candidate_scores": candidate_scores, "windows": windows, "usage": dict(usage)}


class _InjectedBackend:
    """At most one daemon job. Timeouts cannot accumulate concurrent scorers."""
    def __init__(self, config, scorer, counter):
        self.config, self.scorer, self.counter = config, scorer, counter
        self.lock = threading.Lock()
        self.closed = False
        self.runtime = {}

    def warmup(self):
        if self.closed:
            raise RerankFailure("closed")
        if self.lock.locked():
            raise RerankFailure("busy")

    def run(self, query, documents, authorize=None):
        if self.closed:
            raise RerankFailure("closed")
        if not self.lock.acquire(blocking=False):
            raise RerankFailure("busy")
        done, cancelled, result, usage = threading.Event(), threading.Event(), {}, _usage()
        deadline = time.monotonic() + self.config.timeout_seconds

        def work():
            try:
                result["value"] = _evaluate(query, documents, self.config, self.scorer, self.counter,
                                            deadline, usage, cancelled, authorize)
            except SnapshotChanged as exc:
                result["revoked"] = exc
            except Exception as exc:
                result["error"] = _failure_code(exc)
            finally:
                self.lock.release()
                done.set()

        try:
            threading.Thread(target=work, name="aml-reranking-offline", daemon=True).start()
        except Exception:
            self.lock.release()
            raise RerankFailure("worker_failed") from None
        if not done.wait(max(0, deadline - time.monotonic())):
            # Latch the caller's timeout. A timer wait and monotonic clock can
            # disagree at the boundary; released work must still stop batching.
            cancelled.set()
            raise RerankFailure("timeout", {**usage, "usage_complete": False, "coverage_complete": False})
        if "revoked" in result:
            raise result["revoked"]
        if "error" in result:
            raise RerankFailure(result["error"], dict(usage))
        return result["value"]

    def close(self):
        self.closed = True


def _artifact_identity(path: Path) -> dict:
    required = ("config.json", "tokenizer_config.json", "special_tokens_map.json",
                "model.safetensors", "tokenizer.json")
    if any(not (path / name).is_file() for name in required):
        raise RerankFailure("model_files_missing")
    if (path / "added_tokens.json").exists():
        raise RerankFailure("model_artifact_mismatch")
    names = (*required, "sentencepiece.bpe.model")
    hashes = {}
    for name in names:
        artifact = path / name
        if not artifact.is_file():
            continue
        digest = hashlib.sha256()
        blob = hashlib.sha1(f"blob {artifact.stat().st_size}\0".encode())
        with artifact.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                blob.update(block)
        hashes[name] = digest.hexdigest()
        kind, expected = PINNED_ARTIFACTS[name]
        if (blob.hexdigest() if kind == "git_blob" else hashes[name]) != expected:
            raise RerankFailure("model_artifact_mismatch")
    return {"artifact_sha256": hashes}


def _load_cross_encoder(config: RerankConfig):
    # This function runs only in the spawned local worker. No environment changes
    # or heavyweight imports occur in the application process.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8")
    runtime = {}
    for package, expected in LOCAL_VERSIONS.items():
        try:
            runtime[package] = version(package)
        except PackageNotFoundError:
            raise RerankFailure("dependency_missing") from None
        if runtime[package].split("+")[0] != expected:
            raise RerankFailure("dependency_version_mismatch")
    artifacts = _artifact_identity(config.local_model_path)
    import torch
    from sentence_transformers import CrossEncoder

    torch.set_num_threads(config.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    if config.device.startswith("cuda:"):
        index = int(config.device.split(":")[1])
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RerankFailure("device_unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = CrossEncoder(str(config.local_model_path), num_labels=1, max_length=config.max_length,
                         device=config.device, revision=MODEL_REVISION, local_files_only=True,
                         token=False, trust_remote_code=False, backend="torch",
                         model_kwargs={"torch_dtype": torch.float32, "use_safetensors": True,
                                       "attn_implementation": "eager"},
                         tokenizer_kwargs={"model_max_length": config.max_length, "use_fast": True},
                         activation_fn=torch.nn.Identity())
    model.model.eval()

    def counter(query, document):
        # Count the identical pair used by ST 5.1.2 predict(), including special
        # tokens. Preserve whitespace and explicitly disable truncation here.
        return len(model.tokenizer(query, document, add_special_tokens=True,
                                   truncation=False)["input_ids"])

    def scorer(query, documents):
        with torch.inference_mode():
            return model.predict([(query, text) for text in documents], batch_size=config.batch_size,
                                 show_progress_bar=False, convert_to_numpy=True,
                                 activation_fn=torch.nn.Identity(), apply_softmax=False)

    warm_query, warm_text = "例外 exception", "必须保留原文 Preserve source text."
    if counter(warm_query, warm_text) > config.max_length:
        raise RerankFailure("warmup_input_too_long")
    _validate_scores(scorer(warm_query, [warm_text]), 1)
    return scorer, counter, {**artifacts, "packages": runtime,
                             "tokenizer_class": type(model.tokenizer).__name__, "warmup_complete": True}


def _local_worker(connection, config: RerankConfig):
    try:
        scorer, counter, runtime = _load_cross_encoder(config)
        connection.send({"status": "ready", "runtime": runtime})
        while True:
            request = connection.recv()
            usage = _usage()
            try:
                result = _evaluate(request["query"], request["documents"], config, scorer, counter,
                                   time.monotonic() + config.timeout_seconds, usage)
                connection.send({"status": "ok", "result": result})
            except Exception as exc:
                code = _failure_code(exc)
                connection.send({"status": "error", "reason": code, "usage": usage})
                if code in {"oom", "timeout", "scoring_failed"}:
                    break
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send({"status": "error", "reason": _failure_code(exc, "model_load_failed")})
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


class _LocalBackend:
    def __init__(self, config):
        self.config = config
        self.process = self.connection = None
        self.lock = threading.Lock()
        self.runtime = {}
        self.closed = False
        self.generation = 0
        self.sender = None
        self.receiver = None

    def _stop(self):
        process, connection = self.process, self.connection
        self.process = self.connection = None
        if connection is not None:
            connection.close()
        if process is not None:
            if process.pid is None:  # Process.start() itself can fail.
                process.close()
                return
            if process.is_alive():
                process.terminate()
            process.join(self.config.shutdown_timeout_seconds)
            if process.is_alive():
                process.kill()
                process.join(self.config.shutdown_timeout_seconds)
            if not process.is_alive():
                process.close()
            else:
                # Retain the only process handle; never spawn over an unkillable
                # worker. A subsequent close/run can retry bounded cleanup.
                self.process = process
                raise RerankFailure("worker_cleanup_failed", {"usage_complete": False})
        if self.sender is not None and self.sender.ident is not None:
            self.sender.join(self.config.shutdown_timeout_seconds)
        if self.receiver is not None and self.receiver.ident is not None:
            self.receiver.join(self.config.shutdown_timeout_seconds)

    def _send(self, request, deadline):
        # Pipe.send can block on a large payload if a ready worker stops reading.
        # Keep serialization and pipe writes inside the same caller time budget.
        done, errors, connection = threading.Event(), [], self.connection

        def send():
            try:
                connection.send(request)
            except Exception:
                errors.append(True)
            finally:
                done.set()

        self.sender = threading.Thread(target=send, name="aml-reranking-ipc", daemon=True)
        self.sender.start()
        if not done.wait(max(0, deadline - time.monotonic())):
            raise RerankFailure("timeout", {"usage_complete": False})
        if errors:
            raise RerankFailure("worker_failed", {"usage_complete": False})

    def _receive(self, timeout, code):
        # poll() signals the first readable bytes, not a complete pickle. Bound
        # the entire recv(), including an incomplete/oversized worker response.
        done, response, connection = threading.Event(), {}, self.connection

        def receive():
            try:
                response["value"] = connection.recv()
            except Exception:
                response["error"] = True
            finally:
                done.set()

        self.receiver = threading.Thread(target=receive, name="aml-reranking-ipc-read", daemon=True)
        self.receiver.start()
        if not done.wait(timeout):
            raise RerankFailure(code, {"usage_complete": False})
        if "error" in response:
            raise RerankFailure("worker_failed", {"usage_complete": False}) from None
        return response["value"]

    def _start(self):
        if ((self.sender is not None and self.sender.is_alive())
                or (self.receiver is not None and self.receiver.is_alive())):
            raise RerankFailure("busy")
        if self.process is not None and self.process.is_alive():
            if self.connection is None:
                raise RerankFailure("worker_cleanup_failed")
            return
        self._stop()
        self.runtime = {}
        path = self.config.local_model_path
        if path is None:
            raise RerankFailure("missing_configuration")
        if path.name != MODEL_REVISION:
            raise RerankFailure("model_revision_mismatch")
        if not path.is_dir():
            raise RerankFailure("model_missing")
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_local_worker, args=(child, self.config), daemon=True)
        started = time.monotonic()
        try:
            self.process.start()
            self.generation += 1
        finally:
            child.close()
        response = self._receive(max(0, self.config.startup_timeout_seconds - (time.monotonic() - started)),
                                 "startup_timeout")
        if response.get("status") != "ready":
            raise RerankFailure(response.get("reason", "model_load_failed"))
        self.runtime = {**response["runtime"], "worker_generation": self.generation,
                        "startup_elapsed_ms": (time.monotonic() - started) * 1000}

    def _enter(self):
        if self.closed:
            raise RerankFailure("closed")
        if not self.lock.acquire(blocking=False):
            raise RerankFailure("busy")

    def warmup(self):
        self._enter()
        try:
            self._start()
        except Exception as exc:
            self._stop()
            raise RerankFailure(_failure_code(exc, "worker_failed")) from None
        finally:
            self.lock.release()

    def run(self, query, documents):
        self._enter()
        try:
            self._start()
            deadline = time.monotonic() + self.config.timeout_seconds
            self._send({"query": query, "documents": documents}, deadline)
            response = self._receive(max(0, deadline - time.monotonic()), "timeout")
            if response.get("status") != "ok":
                raise RerankFailure(response.get("reason", "worker_failed"), response.get("usage"))
            return response["result"]
        except Exception as exc:
            self._stop()
            raise RerankFailure(_failure_code(exc, "worker_failed"),
                                exc.usage if isinstance(exc, RerankFailure) else {"usage_complete": False}) from None
        finally:
            self.lock.release()

    def close(self):
        if not self.lock.acquire(blocking=False):
            raise RerankFailure("busy")
        try:
            self.closed = True
            self._stop()
        finally:
            self.lock.release()


def _components(candidates, groups):
    """Union same-source identities and explicit complete dependency groups."""
    positions = {candidate.id: i for i, candidate in enumerate(candidates)}
    roots = list(range(len(candidates)))

    def root(index):
        while roots[index] != index:
            roots[index] = roots[roots[index]]
            index = roots[index]
        return index

    def union(a, b):
        a, b = root(a), root(b)
        roots[max(a, b)] = min(a, b)

    sources = {}
    for i, hit in enumerate(candidates):
        identity = (hit.user_id, hit.source_raw_id, hit.source_version, hit.source_sha256)
        if identity in sources:
            union(i, sources[identity])
        else:
            sources[identity] = i
    if not isinstance(groups, (list, tuple)):
        raise RerankFailure("invalid_groups")
    if len(groups) > 200:
        raise RerankFailure("group_budget_exceeded")
    for group in groups:
        if not isinstance(group, Mapping):
            raise RerankFailure("invalid_groups")
        members = group.get("member_ids")
        if (not isinstance(members, (list, tuple)) or len(members) > 200
                or any(not isinstance(member, str) for member in members)
                or not isinstance(group.get("seed_id"), str)):
            raise RerankFailure("invalid_groups")
        ids = [group["seed_id"], *members]
        if group.get("unavailable_dependencies") or any(member not in positions for member in ids):
            raise RerankFailure("missing_dependency")
        for member in ids:
            union(positions[ids[0]], positions[member])
    components = {}
    for i in range(len(candidates)):
        components.setdefault(root(i), []).append(i)
    return list(components.values())


class EvidenceReranker:
    """Reusable plugin. Call after the source gate and before evidence packing.

    Injectable scorers run locally in one bounded daemon thread; production
    CrossEncoder inference runs in a killable, persistent spawned process.
    """
    def __init__(self, config: RerankConfig | None = None, *, scorer: Scorer | None = None,
                 pair_token_counter: PairTokenCounter | None = None, scorer_id: str = "injected",
                 tokenizer_id: str | None = None):
        self.config = config or RerankConfig()
        if scorer is None and pair_token_counter is not None:
            raise ValueError("pair_token_counter requires an injected scorer")
        if pair_token_counter is not None and not tokenizer_id:
            raise ValueError("an injected tokenizer must have an explicit tokenizer_id")
        if scorer is not None and (not callable(scorer) or not scorer_id.strip()):
            raise ValueError("scorer must be callable and have an identity")
        self.backend = (_InjectedBackend(self.config, scorer, pair_token_counter or _byte_pair_tokens)
                        if scorer is not None else _LocalBackend(self.config))
        self.identity = {**self.config.identity, "implementation_kind": "injected" if scorer is not None else "local_cross_encoder",
                         "scorer_id": scorer_id if scorer is not None else MODEL_ID,
                         "tokenizer_id": (tokenizer_id or "offline_utf8_bytes_plus_4_v1") if scorer is not None
                         else f"{MODEL_ID}@{MODEL_REVISION}"}
        if scorer is not None:
            # A fixture is never reported as the pinned multilingual CE.
            self.identity.update(model=None, revision=None, license=None, local_versions={},
                                 expected_artifacts={}, dtype="scorer_defined", device="scorer_defined", backend="injected",
                                 activation="scorer_defined", seed=None, deterministic_algorithms=None, attention=None)
        self.identity["config_sha256"] = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()

    def warmup(self) -> dict:
        if not self.config.enabled:
            return {"status": "disabled", "identity": deepcopy(self.identity)}
        started = time.monotonic()
        try:
            self.backend.warmup()
            result = {"status": "ready"}
        except RerankFailure as exc:
            result = {"status": "failed", "reason": exc.code}
        return {**result, "identity": deepcopy(self.identity), "runtime": deepcopy(self.backend.runtime),
                "elapsed_ms": (time.monotonic() - started) * 1000}

    def rerank(self, query: str, candidates: Sequence[EvidenceCandidate], *, user_id: str,
               groups: Sequence[Mapping] = (), authorize=None) -> tuple[list[EvidenceCandidate], dict]:
        started = time.monotonic()
        original = list(candidates)
        trace = {"status": "disabled", "reason": None, "fallback": False,
                 "identity": deepcopy(self.identity), "candidate_count": len(original), **_usage()}

        def finish(order):
            trace["elapsed_ms"] = (time.monotonic() - started) * 1000
            trace["runtime"] = deepcopy(self.backend.runtime)
            return order, trace

        if not self.config.enabled:
            return finish(original)
        if not original:
            trace["status"] = "empty"
            return finish(original)
        try:
            if not isinstance(query, str) or not query.strip() or not isinstance(user_id, str) or not user_id.strip():
                raise RerankFailure("invalid_input")
            if len(query) > self.config.max_query_chars:
                raise RerankFailure("query_budget_exceeded")
            if len(original) > self.config.max_candidates:
                raise RerankFailure("candidate_budget_exceeded")
            if any(not isinstance(hit, EvidenceCandidate) for hit in original):
                raise RerankFailure("invalid_candidates")
            if len({hit.id for hit in original}) != len(original):
                raise RerankFailure("duplicate_candidates")
            if any(hit.user_id != user_id or hit.deletion_state != "active" or not hit.source_raw_id
                   or not hit.content.strip() for hit in original):
                raise RerankFailure("invalid_candidates")
            if (any(len(hit.content) > self.config.max_candidate_chars for hit in original)
                    or sum(len(hit.content) for hit in original) > self.config.max_total_chars):
                raise RerankFailure("text_budget_exceeded")
            units = _components(original, groups)
            documents = [hit.content for hit in original]

            def authorize_indexes(indexes):
                if authorize is None:
                    return
                chosen, seen = [], set()
                for index in indexes:
                    if index not in seen:
                        seen.add(index)
                        chosen.append(original[index])
                authorize(chosen)

            if isinstance(self.backend, _InjectedBackend):
                result = self.backend.run(query, documents,
                                          authorize=authorize_indexes if authorize is not None else None)
            else:
                # The local worker receives one payload. Internal window batches
                # are already in flight after send and cannot be cancelled.
                if authorize is not None:
                    try:
                        authorize(original)
                    except SnapshotChanged as exc:
                        exc.dispatch_trace = _batch_revocation(0, 1, unit="single_worker_request")
                        raise
                result = self.backend.run(query, documents)
            scores = _validate_scores(result["candidate_scores"], len(original))
            ranked = sorted(units, key=lambda unit: (-max(scores[i] for i in unit), unit[0]))
            ordered = [original[i] for unit in ranked for i in unit]
            trace.update(result["usage"], status="ok")
            trace["components"] = [{"member_ids": [original[i].id for i in unit],
                                    "score": max(scores[i] for i in unit)} for unit in ranked]
            trace["candidate_scores"] = [{"candidate_id": hit.id, "source_raw_id": hit.source_raw_id,
                                           "source_version": hit.source_version, "source_sha256": hit.source_sha256,
                                           "window_max_score": scores[i], "retrieval_score": hit.score}
                                          for i, hit in enumerate(original)]
            trace["windows"] = [{**window, "candidate_id": original[window["candidate_index"]].id,
                                 "source_start": (original[window["candidate_index"]].span_start or 0) + window["start"],
                                 "source_end": (original[window["candidate_index"]].span_start or 0) + window["end"]}
                                for window in result["windows"]]
            return finish(ordered)
        except RerankFailure as exc:
            trace.update(exc.usage, status="fallback", reason=exc.code, fallback=True, coverage_complete=False)
            return finish(original)

    def close(self):
        self.backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
