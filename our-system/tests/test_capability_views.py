"""Synthetic adversarial contracts, not benchmark quality or entailment tests."""
from copy import deepcopy
import ast
import hashlib
import json
import socket
import sqlite3
from pathlib import Path

import pytest

from app import capability_views as cv
from app.models import EvidenceCandidate


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def span(text, *, user="scope-owner", raw_id="raw_fixture", prefix="", stamp=None, role="user", version=1):
    """Independent serialization of the existing segments.py identity contract."""
    start = len(prefix)
    values = dict(user_id=user, source_raw_id=raw_id, source_version=version,
                  source_sha256=sha(prefix + text), span_start=start, span_end=start + len(text),
                  span_sha256=sha(text), fragment_hash=sha("fixture-fragment-policy"))
    identity = [values[key] for key in ("user_id", "source_raw_id", "source_version", "source_sha256",
                                      "span_start", "span_end", "span_sha256", "fragment_hash")]
    return EvidenceCandidate(
        id="span_" + sha(json.dumps(identity, ensure_ascii=False, separators=(",", ":"))),
        content=text, score=0.75, created_at=stamp if stamp is not None else 7,
        deletion_state="active", message_index=0, representation="span_exact_v1", role=role,
        source_timestamp=stamp, recorded_at=7, session_id="session-fixture",
        boundary_status="source_end", **values,
    )


def views_for(candidate):
    return cv.build_views([candidate], "scope-owner")[0]


def get_view(candidate, kind):
    return next(view for view in views_for(candidate) if view["kind"] == kind)


def assert_quotes(candidate, view):
    source = view["source"]
    assert source["span_id"] == candidate.id
    assert source["raw_id"] == candidate.source_raw_id
    assert source["user_id"] == candidate.user_id
    assert source["version"] == candidate.source_version
    assert source["span_sha256"] == candidate.span_sha256
    assert source["source_sha256"] == candidate.source_sha256
    assert source["absolute_start"] == candidate.span_start
    assert source["absolute_end"] == candidate.span_end
    assert source["offset_unit"] == "unicode_codepoint"
    assert len(view["fields"]) <= cv.MAX_FIELDS
    for field in view["fields"].values():
        assert field["field_supported"] == "unknown"
        assert len(field["ranges"]) == len(field["values"]) <= cv.MAX_RANGES_PER_FIELD
        for value, interval in zip(field["values"], field["ranges"]):
            assert 0 <= interval["start"] < interval["end"] <= len(candidate.content)
            assert value == candidate.content[interval["start"]:interval["end"]]
            assert interval["absolute_start"] == candidate.span_start + interval["start"]
            assert interval["absolute_end"] == candidate.span_start + interval["end"]
    assert view["fields"]["statement"]["values"] == [candidate.content]
    assert view["support"] == {
        "source_valid": "unknown", "field_supported": "unknown", "set_sufficient": "unknown",
        "basis": "lexical_navigation", "source_validation_owner": "caller_gate",
    }


@pytest.mark.parametrize("text,subject", [
    ("My friend prefers quiet rooms.", "My friend"),
    ("Mira likes bright rooms.", "Mira"),
    ("我的朋友喜欢安静的房间。", "我的朋友"),
    ("她不喜欢喧闹的房间。", "她"),
])
def test_other_subject_is_quoted_and_never_scope_owner(text, subject):
    candidate = span(text)
    view = get_view(candidate, "preference")
    assert view["fields"]["subject"]["values"] == [subject]
    assert "scope-owner" not in json.dumps(view["fields"], ensure_ascii=False)
    assert_quotes(candidate, view)


@pytest.mark.parametrize("role", ["user", "assistant"])
@pytest.mark.parametrize("text,subject", [
    ("Prefer quiet rooms.", []),
    ("喜欢安静的房间。", []),
    ("I prefer quiet rooms.", ["I"]),
    ("我喜欢安静的房间。", ["我"]),
    ('Mira said: "I prefer quiet rooms."', []),
    ("The assistant recommends quiet rooms.", []),
    ("Mira likes quiet rooms. I like bright rooms.", []),
])
def test_no_subject_inference_from_user_or_role(role, text, subject):
    candidate = span(text, role=role)
    views = views_for(candidate)
    assert views
    for view in views:
        assert view["source"]["role"] == role
        assert view["fields"]["subject"]["values"] == subject
        assert view["fields"]["subject"]["lexical_status"] == ("exact_quote" if subject else "unknown")


@pytest.mark.parametrize("text", [
    "If the room is noisy, Mira does not prefer it unless the windows are shut.",
    "Mira doesn't like the room when the windows are open.",
    "如果房间很吵，我不喜欢它，除非窗户关上。",
    "我没有喜欢这个方案，只有预算足够才考虑。",
])
def test_negative_conditional_preference_never_becomes_positive_fact(text):
    candidate = span(text)
    view = get_view(candidate, "preference")
    assert view["fields"]["negation"]["ranges"]
    assert view["fields"]["condition"]["ranges"]
    assert_quotes(candidate, view)
    assert "positive" not in json.dumps(view)


@pytest.mark.parametrize("text,status", [
    ("Mira visited the studio on 2024-03-09.", "occurrence_cue"),
    ("Mira plans to visit the studio tomorrow.", "planned"),
    ("我计划明天参加会议。", "planned"),
    ("我参加了会议。", "occurrence_cue"),
    ("If Mira visited the studio, she may have met someone.", "unknown"),
    ("If Mira visited the studio, the door was open.", "conditional_unknown"),
    ("Mira never visited the studio.", "negated_unknown"),
    ("我没有参加了会议。", "negated_unknown"),
    ("Mira visited the studio and plans to return.", "mixed_unknown"),
    ('Mira said "I visited the studio".', "unknown"),
    ("Did Mira visit tomorrow?", "unknown"),
    ("Mira visited, except in the simulation.", "qualified_unknown"),
    ("Rule: operators shall execute the workflow.", "unknown"),
    ("I shall execute the workflow.", "unknown"),
    ("The list includes completed tasks.", "unknown"),
    ("Visited the studio yesterday.", "unknown"),
    ("Mira visited. Nola visited.", "unknown"),
    ("我将在2028年3月1日参加会议。", "planned"),
    ("I am scheduled to visit on 2028-03-01.", "planned"),
])
def test_event_occurrence_and_plan_are_distinct_lexical_statuses(text, status):
    candidate = span(text)
    view = get_view(candidate, "event")
    assert view["navigation"]["event_status"] == status
    assert_quotes(candidate, view)


@pytest.mark.parametrize("text", ["Mira visited yesterday.", "我昨天参加了会议。", "Mira plans to visit next week."])
@pytest.mark.parametrize("stamp", [None, 0, 1710000000000])
def test_relative_time_never_uses_arrival_or_host_today(text, stamp):
    candidate = span(text, stamp=stamp)
    views = views_for(candidate)
    for view in views:
        assert view["fields"]["relative_time"]["ranges"]
        nav = view["navigation"]
        assert nav["resolved_relative_date"] is None
        assert nav["relative_time_status"] == ("unresolved_no_source_timestamp" if stamp is None else "unresolved_timezone_or_granularity")
    changed = candidate.model_copy(update={"recorded_at": 9000000000000,
                                           "created_at": stamp if stamp is not None else 9000000000000})
    assert [v["navigation"] for v in views_for(changed)] == [v["navigation"] for v in views]


def test_absolute_dates_are_calendar_checked_without_becoming_event_times():
    candidate = span("Mira visited on 2024-02-29, not 2025-02-29; the rule date is 2026年1月2日.")
    view = get_view(candidate, "event")
    assert view["fields"]["absolute_date"]["values"] == ["2024-02-29", "2025-02-29", "2026年1月2日"]
    assert view["navigation"]["absolute_dates"] == [
        {"range_index": 0, "iso_date": "2024-02-29", "precision": "day"},
        {"range_index": 1, "iso_date": None, "precision": "unknown"},
        {"range_index": 2, "iso_date": "2026-01-02", "precision": "day"},
    ]
    assert "event_time" not in view["navigation"]
    assert view["navigation"]["event_status"] != "occurrence_cue"


@pytest.mark.parametrize("text", ["2024年03-01", "2024-03月01日", "012024-03-01", "2024-03-011"])
def test_malformed_or_digit_embedded_date_is_not_normalized(text):
    view = views_for(span(text))[0]
    assert view["fields"]["absolute_date"]["values"] == []
    assert view["navigation"]["absolute_dates"] == []


@pytest.mark.parametrize("text", [
    "Rule: if enabled, first inspect; then execute step 2, except when locked; otherwise stop. The task is pending.",
    "规则：如果启用，首先检查，然后执行步骤2，除非锁定，否则停止。任务尚未完成。",
])
def test_rule_exceptions_steps_and_task_remain_whole_and_coexist(text):
    candidate = span(text)
    views = views_for(candidate)
    assert {"constraint", "procedure", "task"} <= {view["kind"] for view in views}
    for view in views:
        for name in ("condition", "exception", "step"):
            assert view["fields"][name]["ranges"]
        assert view["navigation"]["update_policy"] == "coexist_no_supersede"
        assert "completed" not in view["navigation"]
        assert_quotes(candidate, view)


def test_non_bmp_offsets_are_python_codepoints_including_absolute_base():
    prefix = "🧭前文𐐷\n"
    candidate = span("Mira likes 😀 rooms, but not if 🧪 is open on 2024-07-01.", prefix=prefix)
    views = views_for(candidate)
    assert candidate.span_start == 5
    for view in views:
        assert_quotes(candidate, view)
        for field in view["fields"].values():
            for value, interval in zip(field["values"], field["ranges"]):
                assert (prefix + candidate.content)[interval["absolute_start"]:interval["absolute_end"]] == value


@pytest.mark.parametrize("updates,reason", [
    ({"user_id": "foreign"}, "foreign_user"),
    ({"deletion_state": "deleted"}, "inactive_candidate"),
    ({"deletion_state": "redacted"}, "inactive_candidate"),
    ({"source_raw_id": ""}, "missing_source_identity"),
    ({"representation": "summary"}, "not_exact_span"),
    ({"representation": "raw"}, "not_exact_span"),
    ({"span_start": -1}, "invalid_offsets"),
    ({"span_start": True}, "invalid_offsets"),
    ({"span_start": 2**53}, "invalid_offsets"),
    ({"span_end": 900}, "invalid_offsets"),
    ({"source_version": None}, "invalid_version"),
    ({"source_version": True}, "invalid_version"),
    ({"source_version": 2}, "span_identity_mismatch"),
    ({"source_sha256": None}, "invalid_hash_metadata"),
    ({"fragment_hash": "invalid"}, "invalid_hash_metadata"),
    ({"span_sha256": "0" * 64}, "span_hash_mismatch"),
    ({"source_sha256": "0" * 64}, "span_identity_mismatch"),
    ({"source_raw_id": "forged_raw"}, "span_identity_mismatch"),
    ({"id": "forged_span"}, "span_identity_mismatch"),
    ({"content": "Mira likes 🌻 rooms."}, "span_hash_mismatch"),
    ({"role": "system"}, "invalid_source_metadata"),
    ({"session_id": ""}, "missing_source_identity"),
    ({"message_index": -1}, "invalid_source_metadata"),
    ({"source_timestamp": "today"}, "invalid_source_metadata"),
    ({"source_timestamp": 2**53}, "invalid_source_metadata"),
    ({"recorded_at": True}, "invalid_source_metadata"),
    ({"source_timestamp": 10}, "inconsistent_source_time_metadata"),
    ({"created_at": 10}, "inconsistent_source_time_metadata"),
])
def test_bad_source_and_metadata_get_no_views(updates, reason):
    candidate = span("Mira likes ☀ rooms.").model_copy(update=updates)
    views, trace = cv.build_views([candidate], "scope-owner")
    assert views == []
    assert trace["decisions"] == [{"candidate_index": 0, "decision": "skip", "reason": reason}]
    assert "content" not in json.dumps(trace)


def test_no_source_and_empty_input_produce_no_navigation_or_fallback():
    assert cv.build_views([], "scope-owner")[0] == []
    bare = {"content": "Mira likes rooms.", "user_id": "scope-owner", "deletion_state": "active", "representation": "span_exact_v1"}
    assert cv.build_views([bare], "scope-owner")[0] == []
    assert cv.rank_views([], [], ("personalization",))[0] == []


def test_malformed_scope_metadata_cannot_crash_or_boost_mixed_input():
    candidate = span("I prefer quiet rooms.")
    malformed = candidate.model_dump()
    malformed["user_id"] = ["scope-owner"]
    views, _ = cv.build_views([candidate, malformed], "scope-owner")
    ranked, trace = cv.rank_views([candidate, malformed], views, ("personalization",))
    assert ranked[0] is candidate and ranked[1] is malformed
    assert not trace["triggered"] and trace["scope_status"] == "unknown_or_mixed"


def test_local_consistency_is_not_a_source_gate_even_for_coherent_metadata():
    # Role/session/raw-version truth cannot be proved from these two arguments.
    candidate = span("Mira likes quiet rooms.", stamp=123)
    forged = candidate.model_copy(update={"role": "assistant", "session_id": "different-session"})
    views, trace = cv.build_views([forged], "scope-owner")
    assert views and trace["source_valid"] == "unknown"
    assert all(v["support"]["source_valid"] == "unknown" for v in views)
    original = views_for(candidate)
    _, ranked_trace = cv.rank_views([forged], original, ("personalization",))
    assert ranked_trace["rejected_views"] == len(original)
    assert not ranked_trace["triggered"]


def test_hard_fragment_cannot_advertise_occurrence():
    candidate = span("Mira visited the studio.").model_copy(update={"boundary_status": "hard_split_requires_context"})
    view = get_view(candidate, "event")
    assert view["navigation"]["event_status"] == "unknown"
    assert view["navigation"]["context_status"] == "requires_context"


def test_conflicting_preferences_rules_and_events_do_not_supersede_or_dedupe():
    candidates = [span(text, raw_id=f"raw_{i}", version=i + 1) for i, text in enumerate([
        "I prefer quiet rooms.", "I prefer bright rooms.", "I visited the studio.",
        "I visited the studio.", "Rule: never open the door.", "Rule: open the door if needed.",
    ])]
    views, _ = cv.build_views(candidates, "scope-owner")
    assert {v["source"]["raw_id"] for v in views} == {c.source_raw_id for c in candidates}
    ranked, _ = cv.rank_views(candidates, views, ("personalization", "procedure", "temporal"))
    assert sorted(map(id, ranked)) == sorted(map(id, candidates))
    assert all(v["navigation"]["update_policy"] == "coexist_no_supersede" for v in views)


@pytest.mark.parametrize("generic", [(), ("temporal",), ("event",)])
@pytest.mark.parametrize("intent,target_status", [("event_occurred", "occurrence_cue"), ("event_planned", "planned")])
def test_ranking_never_matches_plan_to_occurrence(intent, target_status, generic):
    candidates = [span(text, raw_id=f"raw_{i}") for i, text in enumerate([
        "I plan to visit tomorrow.", "I visited yesterday.", "If I visited, the door was open.",
        "I did not visit but plan to visit.", "I visited and will return.",
    ])]
    views, _ = cv.build_views(candidates, "scope-owner")
    ranked, trace = cv.rank_views(candidates, views, (intent,) + generic)
    status_by_index = {v["candidate_index"]: v["navigation"]["event_status"] for v in views if v["kind"] == "event"}
    for row in trace["rows"]:
        assert bool(row["type_boost"]) == (status_by_index[row["candidate_index"]] == target_status)
    assert sorted(map(id, ranked)) == sorted(map(id, candidates))
    assert all(hit.content == candidates[trace["output_indices"][i]].content for i, hit in enumerate(ranked))


def test_rank_boost_is_small_non_stacking_and_preserves_original_objects():
    candidates = [span("The sky is clear.", raw_id="raw_0"),
                  span("I prefer a quiet workflow; first avoid noise, then finish the task.", raw_id="raw_1")]
    before = [c.model_dump() for c in candidates]
    views, _ = cv.build_views(candidates, "scope-owner")
    ranked, trace = cv.rank_views(candidates, views + deepcopy(views), ("personalization", "procedure"))
    assert ranked[0] is candidates[1] and ranked[1] is candidates[0]
    assert 0 < trace["rows"][1]["type_boost"] <= 0.05 * trace["rows"][1]["base_rank_score"]
    assert trace["rows"][1]["type_boost"] == cv.TYPE_BOOST_FRACTION * trace["rows"][1]["base_rank_score"]
    assert trace["rows"][0]["type_boost"] == 0
    assert trace["duplicate_views"] == len(views)
    assert [c.model_dump() for c in candidates] == before
    assert trace["output_policy"] == "whole_original_span"


@pytest.mark.parametrize("tamper", ["raw", "version", "hash", "offset", "field", "status", "support", "index"])
def test_tampered_navigation_never_has_ranking_authority(tamper):
    candidates = [span("The sky is clear.", raw_id="r0"), span("I plan to visit tomorrow.", raw_id="r1")]
    views, _ = cv.build_views(candidates, "scope-owner")
    forged = deepcopy(next(v for v in views if v["kind"] == "event"))
    if tamper == "raw": forged["source"]["raw_id"] = "foreign"
    if tamper == "version": forged["source"]["version"] += 1
    if tamper == "hash": forged["source"]["span_sha256"] = "0" * 64
    if tamper == "offset": forged["source"]["absolute_start"] += 1
    if tamper == "field": forged["fields"]["statement"]["values"] = ["a generated assertion"]
    if tamper == "status": forged["navigation"]["event_status"] = "occurrence_cue"
    if tamper == "support": forged["support"]["field_supported"] = True
    if tamper == "index": forged["candidate_index"] = 0
    ranked, trace = cv.rank_views(candidates, [forged], ("temporal",))
    assert trace["rejected_views"] == 1 and not trace["triggered"]
    assert all(a is b for a, b in zip(ranked, candidates))


def test_foreign_deleted_or_stale_views_cannot_reorder_a_candidate_set():
    candidate = span("I prefer quiet rooms.")
    original = views_for(candidate)
    for updates in ({"deletion_state": "deleted"}, {"user_id": "foreign"}, {"source_version": 8}):
        changed = candidate.model_copy(update=updates)
        ranked, trace = cv.rank_views([changed], original, ("personalization",))
        assert ranked[0] is changed and not trace["triggered"]
        assert trace["accepted_views"] == 0
    foreign = span("I prefer quiet rooms.", user="foreign")
    _, trace = cv.rank_views([candidate, foreign], original, ("personalization",))
    assert trace["scope_status"] == "unknown_or_mixed" and not trace["triggered"]


def test_unknown_intents_missing_views_and_reordered_input_have_no_boost():
    candidates = [span("I prefer quiet rooms.", raw_id="r0"), span("I prefer bright rooms.", raw_id="r1")]
    views, _ = cv.build_views(candidates, "scope-owner")
    for supplied, intent in ((views, "unrecognized"), ([], "personalization")):
        ranked, trace = cv.rank_views(candidates, supplied, intent)
        assert not trace["triggered"] and all(a is b for a, b in zip(ranked, candidates))
    _, trace = cv.rank_views(candidates[::-1], views, "personalization")
    assert not trace["triggered"] and trace["rejected_views"] == len(views)


def test_same_candidate_multiplicity_and_json_mapping_support():
    candidate = span("I prefer quiet rooms.")
    views, trace = cv.build_views([candidate, candidate], "scope-owner")
    ranked, ranking = cv.rank_views([candidate, candidate], views, "personalization")
    assert len(ranked) == 2 and all(c is candidate for c in ranked)
    mapping = candidate.model_dump()
    mapped_views, _ = cv.build_views([mapping], "scope-owner")
    assert mapped_views == views_for(candidate)
    assert cv.rank_views([mapping], mapped_views, "personalization")[0][0] is mapping
    for data in (views, trace, ranking):
        assert json.loads(json.dumps(data, ensure_ascii=False, allow_nan=False)) == data


def test_limits_are_explicit_and_do_not_silently_drop_candidate_tail():
    candidate = span("I prefer quiet rooms.")
    views, trace = cv.build_views([candidate] * 200, "scope-owner")
    assert trace["candidate_count"] == 200 and len(views) <= cv.MAX_VIEWS
    assert len(cv.rank_views([candidate] * 200, views, "personalization")[0]) == 200
    _, ranking = cv.rank_views([candidate] * 200, views, "personalization")
    assert all(row["type_boost"] <= row["base_rank_score"] * 0.05 for row in ranking["rows"])
    for function, args in ((cv.build_views, ([candidate] * 201, "scope-owner")),
                           (cv.rank_views, ([candidate] * 201, [], ())),
                           (cv.rank_views, ([], [{}] * (cv.MAX_VIEWS + 1), ())),
                           (cv.rank_views, ([], [], ["claim"] * 17))):
        with pytest.raises(ValueError, match="exceeds"):
            function(*args)
    with pytest.raises(ValueError):
        cv.build_views([], "")
    assert views_for(span("a" * 1801)) == []
    repeated = span("I prefer rooms if open, not if shut. " * 40)
    view = get_view(repeated, "preference")
    assert view["fields"]["condition"]["truncated"]
    assert view["fields"]["negation"]["truncated"]
    assert_quotes(repeated, view)


def test_candidate_iterator_is_bounded_without_exhausting_unbounded_input():
    candidate = span("I prefer rooms.")
    visits = []
    def infinite():
        while True:
            visits.append(1)
            yield candidate
    with pytest.raises(ValueError, match="exceeds"):
        cv.build_views(infinite(), "scope-owner")
    assert len(visits) == 201


def test_module_has_no_external_runtime_or_clock_dependency():
    # A source audit supplements I/O tripwires: provider/store/tokenizer imports
    # could otherwise add hidden I/O even when this synthetic call does not.
    tree = ast.parse(Path(cv.__file__).read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom): imports.add(node.module)
    assert imports <= {"__future__", "collections.abc", "datetime", "hashlib", "itertools", "json", "re"}
    assert not any(isinstance(node, ast.Attribute) and node.attr in {"now", "today", "utcnow"} for node in ast.walk(tree))


def test_pure_calls_do_not_write_db_call_network_generate_prose_or_mutate(monkeypatch):
    candidates = [span("I prefer rooms if quiet, but never when locked.")]
    before = deepcopy(candidates[0].model_dump())
    def forbidden(*args, **kwargs):
        pytest.fail("pure capability views attempted external I/O")
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    monkeypatch.setattr("builtins.open", forbidden)
    views, trace = cv.build_views(candidates, "scope-owner")
    copied_views = deepcopy(views)
    ranked, ranking = cv.rank_views(candidates, views, ("personalization",))
    assert candidates[0].model_dump() == before and views == copied_views
    assert ranked[0] is candidates[0]
    def no_prose_keys(data):
        if isinstance(data, dict):
            assert not {"text", "generated_text", "summary", "answer", "object_text"}.intersection(data)
            for value in data.values(): no_prose_keys(value)
        elif isinstance(data, list):
            for value in data: no_prose_keys(value)
    for data in (views, trace, ranking):
        no_prose_keys(data)
    for view in views:
        assert_quotes(candidates[0], view)
