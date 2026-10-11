"""Add/Search contracts and the raw-message record.

DerivedMemory is a typed placeholder for later fact governance. Phase A does
not create, store, or return derived rows.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MessageIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str
    timestamp: int | None = None

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must be non-empty")
        return value

    @field_validator("timestamp", mode="before")
    @classmethod
    def timestamp_must_be_unix_ms(cls, value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("timestamp must be an integer unix millisecond")
        if value < 0:
            raise ValueError("timestamp must be >= 0")
        return value


class AddRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    messages: list[MessageIn] = Field(min_length=1)

    @field_validator("request_id", "user_id", "session_id")
    @classmethod
    def identifiers_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("identifier must be non-empty")
        return value


class AddResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    success: Literal[True]
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    top_k: int
    options: list[str] | None = None

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must be non-empty")
        return stripped

    @field_validator("user_id")
    @classmethod
    def user_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("user_id must be non-empty")
        return value

    @field_validator("top_k", mode="before")
    @classmethod
    def top_k_must_be_positive_int(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("top_k must be an integer")
        if value < 1:
            raise ValueError("top_k must be >= 1")
        return value

    @field_validator("options")
    @classmethod
    def options_must_be_text(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if any(not item.strip() for item in value):
            raise ValueError("options must be non-empty strings")
        return value


class SearchHit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    score: float
    created_at: int


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    data: list[SearchHit]


class RawMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    user_id: str
    session_id: str
    request_id: str
    message_index: int
    role: Literal["user", "assistant"]
    content: str
    source_timestamp: int | None
    recorded_at: int
    privacy_level: str = "normal"
    deletion_state: str = "active"


class EvidenceCandidate(BaseModel):
    """One saved raw message selected for a Search response."""

    model_config = ConfigDict(extra="ignore")

    id: str
    user_id: str
    content: str
    score: float
    created_at: int
    deletion_state: str
    source_raw_id: str
    message_index: int
    representation: str = "raw"
    span_start: int | None = None
    span_end: int | None = None
    source_version: int | None = None
    source_sha256: str | None = None
    span_sha256: str | None = None
    fragment_hash: str | None = None
    role: str | None = None
    session_id: str | None = None
    source_timestamp: int | None = None
    recorded_at: int | None = None
    boundary_status: str | None = None
    previous_id: str | None = None
    next_id: str | None = None


class DerivedMemory(BaseModel):
    """Placeholder for a later derived fact or event. Not written in phase A."""

    model_config = ConfigDict(extra="ignore")

    id: str
    user_id: str
    source_raw_id: str
    kind: Literal["fact", "event"]
    subject: str
    predicate: str
    object_text: str
    claim_key: str | None = None
    source_timestamp: int | None = None
    event_time: int | None = None
    recorded_at: int
    confidence: float | None = None
    governance_state: Literal["active", "superseded", "deleted", "redacted"] = "active"
    supersedes_id: str | None = None
