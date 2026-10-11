"""Auditable evidence rendering, not an answer or inferred fact."""
import json


def render_span_source(candidate):
    metadata={"source_id":candidate.source_raw_id,"observed_role":candidate.role,
              "source_version":candidate.source_version,"source_time_unix_ms":candidate.source_timestamp}
    return "[source " + json.dumps(metadata,ensure_ascii=False,sort_keys=True,separators=(",",":")) + "]\n" + candidate.content


def render_retracted_source(*, raw_id: str, content: str, role: str, timestamp: int | None) -> str:
    return json.dumps({"representation":"attributed_raw_v1","source_id":raw_id,
                       "observed_role":role,"source_timestamp":timestamp,
                       "retraction_state":"retracted","text":content},
                      ensure_ascii=False,sort_keys=True,separators=(",",":"))
