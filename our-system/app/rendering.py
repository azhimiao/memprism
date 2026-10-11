"""Auditable evidence rendering, not an answer or inferred fact."""
import json
import re
from datetime import datetime, timedelta, timezone


def should_show_source_dates(query: str) -> bool:
    """Query-only navigation, not an event-time or support classifier."""
    return bool(re.search(r"何时|什么时候|多久|先后|之前|之后|最早|最近|截至|"
                         r"\b(?:when|before|after|how long|earliest|latest|chronolog\w*|"
                         r"how many (?:days|weeks|months|years)|which (?:day|month|year)|"
                         r"what (?:date|month|year))\b", query, re.I))


def render_source_date(candidate):
    """Show explicit source time without inventing event time or changing the body.

    Unknown/out-of-calendar times stay unknown. Integer timedelta arithmetic
    preserves the correct day for negative millisecond timestamps.
    """
    stamp = candidate.source_timestamp
    if type(stamp) is not int:
        return candidate.content
    try:
        date = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=stamp)
    except (OverflowError, ValueError):
        return candidate.content
    return f"[Source recorded: {date.year:04d}-{date.month:02d}-{date.day:02d} UTC]\n" + candidate.content


def render_span_source(candidate):
    metadata={"source_id":candidate.source_raw_id,"observed_role":candidate.role,
              "source_version":candidate.source_version,"source_time_unix_ms":candidate.source_timestamp}
    return "[source " + json.dumps(metadata,ensure_ascii=False,sort_keys=True,separators=(",",":")) + "]\n" + candidate.content


def render_metadata(candidate, *, roles=False, dates=False):
    fields=[]
    if roles and candidate.role in {'user','assistant'}:
        fields.append('role='+candidate.role)
    if dates and type(candidate.source_timestamp) is int:
        try:
            date=datetime(1970,1,1,tzinfo=timezone.utc)+timedelta(milliseconds=candidate.source_timestamp)
            fields.append('source_date='+date.strftime('%Y-%m-%dT%H:%M:%SZ'))
        except (OverflowError,ValueError):
            pass
    return ('['+'; '.join(fields)+']\n' if fields else '')+candidate.content


def render_retracted_source(*, raw_id: str, content: str, role: str, timestamp: int | None) -> str:
    return json.dumps({"representation":"attributed_raw_v1","source_id":raw_id,
                       "observed_role":role,"source_timestamp":timestamp,
                       "retraction_state":"retracted","text":content},
                      ensure_ascii=False,sort_keys=True,separators=(",",":"))
