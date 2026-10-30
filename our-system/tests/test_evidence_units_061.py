"""061 engineering witnesses; no model calls or benchmark quality claims.

Run from our-system with its pinned Python and an exclusive --basetemp.
Every fixture source is published by formal Add; seeds are obtained through
formal Search plus the identical saved span snapshot (not hand-built spans).
"""
from contextlib import closing

import pytest
from fastapi.testclient import TestClient

from app import segments
from app.api import create_app
from app.config import Settings
from app.evidence_units import expand_units, RAW_CHAR_LIMIT
from app.support_groups import pack_groups


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(Settings(db_path=tmp_path / "units.sqlite", profile_id="cycle2_raw"))) as c:
        yield c


def add(c, texts, *, user="u", request="r"):
    response = c.post("/add", json={"user_id": user, "session_id": "s", "request_id": request,
        "messages": [{"role": "user", "content": text, "timestamp": 1704067200000} for text in texts]})
    assert response.status_code == 200, response.text[:300]


def recall(c, query, *, user="u", count=200):
    response = c.post("/search", json={"user_id": user, "query": query, "top_k": 100})
    assert response.status_code == 200, response.text[:300]
    snapshot = c.app.state.store.search_snapshot(user_id=user, match_query='"' + query + '"',
        query_vector=None, version="unconfigured", dimensions=0, mode="raw_fts",
        candidate_limit=count, lexical_quota=count, dense_quota=0, span=True)
    hits = snapshot.lexical
    # Search response rendering may change presentation, but not saved identity.
    if hits:
        assert {h.id for h in hits} & {h["id"] for h in response.json()["data"]}
    return hits


def run(c, seeds, **kwargs):
    with closing(c.app.state.store._connect()) as db:
        db.execute("BEGIN")
        return expand_units(db, "u", seeds, **kwargs)


def table(marker="tailneedle", *, heading="Product", unit="Units: kg", rows=105):
    return unit + "\n| " + heading + " | Quantity |\n| --- | --- |\n" + "".join(
        f"| item{i:03d} | {i:03d} |\n" for i in range(rows)) + f"| {marker} | 4321 |\n"


def rules(padding=27):
    return ("# Dispatch rules\nPrerequisite: the operator must be certified.\n"
            + "Routine preparation notes are recorded before dispatch.\n" * padding
            + "The operator must run rulemarker before departure.\n"
            + "Routine review notes are retained for inspection.\n" * padding
            + "Exception: unless the alarm is clear, departure is prohibited.\n")


def assert_final(c, hits):
    store, coordinator = c.app.state.store, c.app.state.coordinator
    policy = store.policy_snapshot("u", hits)
    coordinator.final_check("u", coordinator.require_ready("u"), hits, policy)
    with closing(store._connect()) as db:
        for hit in hits:
            src = db.execute("""SELECT m.*,p.source_version,p.content_sha256 FROM raw_messages m
                JOIN source_policies p ON p.raw_id=m.id AND p.user_id=m.user_id
                WHERE m.user_id=? AND m.id=?""", ("u", hit.source_raw_id)).fetchone()
            assert segments.valid_candidate(db, hit, src)
            assert hit.role == "user" and hit.session_id == "s"
            assert hit.created_at == hit.source_timestamp == 1704067200000


def test_tail_recalls_header_unit_and_final_coordinator_accepts(client):
    text = table()
    add(client, [text])
    seed = recall(client, "tailneedle")[0]
    assert "Product" not in seed.content and "Units:" not in seed.content
    hits, groups, trace = run(client, [seed])
    assert hits[0] == seed and trace["expanded_spans"] > 0
    assert any("Product" in h.content and "Units: kg" in h.content for h in hits)
    assert groups[0]["witness"] == "table_schema"
    assert groups[0]["required_ids"] and groups[0]["optional_ids"] == []
    assert set(groups[0]["member_ids"]) <= {h.id for h in hits}
    assert all(h.source_raw_id == seed.source_raw_id for h in hits)
    assert_final(client, hits)
    packed, _, _ = pack_groups(hits, groups, 100, 32000)
    assert packed == hits
    assert run(client, [seed]) == (hits, groups, trace)


def test_second_table_never_borrows_first_header(client):
    add(client, [table("firsttail", heading="WrongHeader", unit="Units: USD") + "\n# Separate stock\n"
                 + table("secondtail", heading="CorrectHeader", unit="Units: grams")])
    seed = recall(client, "secondtail")[0]
    hits, groups, _ = run(client, [seed])
    assert groups and any("CorrectHeader" in h.content for h in hits)
    assert not any("WrongHeader" in h.content or "Units: USD" in h.content for h in hits)


@pytest.mark.parametrize("where", ["before", "after", "header"])
def test_explicit_unit_locations(client, where):
    text = table(unit="单位：千元")
    if where == "after":
        text = text.replace("单位：千元\n", "") + "单位：千元\n"
    if where == "header":
        text = text.replace("单位：千元\n", "").replace("Quantity", "金额（千元）")
    add(client, [text])
    hits, groups, _ = run(client, recall(client, "tailneedle")[:1])
    assert groups and any("千元" in h.content for h in hits)
    assert_final(client, hits)


def test_rule_complete_short_section_includes_prerequisite_exception(client):
    text = rules() + "\n# Unrelated policy\nException: unrelatedexception is not permitted.\n"
    add(client, [text])
    seed = recall(client, "rulemarker")[0]
    hits, groups, _ = run(client, [seed])
    assert groups and groups[0]["witness"] == "rule_section"
    assert any("Prerequisite" in h.content for h in hits)
    assert any("unless the alarm" in h.content for h in hits)
    # Witness intervals stop at the next heading, regardless of span overlap.
    assert all(hi <= text.index("# Unrelated") for _, hi in groups[0]["support_intervals"])
    assert_final(client, hits)


def test_rule_does_not_borrow_exception_from_another_section(client):
    text = "# Boarding rules\nThe operator must run lonerule.\n" + "Notes about boarding.\n" * 100
    text += "# Other rules\nException: unrelated weather permits entry.\n"
    add(client, [text])
    seeds = recall(client, "lonerule")[:1]
    hits, groups, _ = run(client, seeds)
    assert hits == seeds and not groups


def test_no_cross_user_source_or_global_recent(client):
    add(client, ["ordinaryneedle is an independent short fact.", table("foreignsource")])
    add(client, [table("ordinaryneedle")], user="other", request="foreign")
    seed = recall(client, "ordinaryneedle")[0]
    foreign = recall(client, "ordinaryneedle", user="other")[0]
    hits, groups, trace = run(client, [seed, foreign])
    assert hits == [seed] and not groups and trace["rejected_ids"] == [foreign.id]


def test_short_facts_are_exactly_unchanged(client):
    add(client, ["factneedle tea is green.", "factneedle train leaves at nine."])
    seeds = recall(client, "factneedle")
    hits, groups, trace = run(client, seeds)
    assert hits == seeds and not groups and trace["expanded_spans"] == 0
    assert [h.model_dump() for h in hits] == [h.model_dump() for h in seeds]


@pytest.mark.parametrize("kwargs", [{"max_extra": 0}, {"limit": 1}, {"group_item_limit": 1},
    {"group_char_limit": 1}, {"group_token_limit": 1}, {"seed_limit": 0}])
def test_budget_insufficient_keeps_independent_seed(client, kwargs):
    add(client, [table()])
    seeds = recall(client, "tailneedle")[:1]
    hits, groups, _ = run(client, seeds, **kwargs)
    assert hits == seeds and not groups
    assert pack_groups(hits, groups, 1, 32000)[0] == seeds


def test_full_pool_gives_tail_slots_to_support_but_preserves_first100(client):
    add(client, [table()] + [f"basepool independent fact {i}." for i in range(199)])
    seed = recall(client, "tailneedle")[0]
    bases = recall(client, "basepool")
    assert len(bases) == 199
    seeds = [seed] + bases
    hits, groups, trace = run(client, seeds)
    assert len(hits) == 200 and groups and trace["expanded_spans"] > 0
    assert hits[:100] == seeds[:100]
    assert len(trace["evicted_ids"]) == trace["expanded_spans"]
    assert not {h.id for h in seeds[:100]} & set(trace["evicted_ids"])
    assert seed.id in {h.id for h in hits}
    assert_final(client, hits)


@pytest.mark.parametrize("limit", [0, 1, 2, 100, 200, 999])
def test_candidate_limit_and_small_protected_pool(client, limit):
    add(client, [table(), "filler independent fact."])
    seeds = recall(client, "tailneedle")[:1] + recall(client, "filler")
    hits, groups, _ = run(client, seeds, limit=limit)
    assert len(hits) <= min(200, limit)
    if limit:
        assert hits[0] == seeds[0]
    if limit <= 2:
        assert not groups


@pytest.mark.parametrize("mutation", ["retract", "delete"])
def test_formal_retraction_deletion_reject_stale_seed(client, mutation):
    text = "lifeneedle is my old statement."
    add(client, [text])
    seeds = recall(client, "lifeneedle")
    command = f'I retract my statement that "{text}"' if mutation == "retract" else f'请忘记“{text}”'
    add(client, [command], request="mutation")
    hits, groups, trace = run(client, seeds)
    assert not hits and not groups and trace["rejected_ids"]


@pytest.mark.parametrize("field,value", [("retraction_state", "retracted"), ("visibility", "suppressed"),
    ("disclosure_policy", "restricted")])
def test_structural_source_policy_denial(client, field, value):
    add(client, [table()])
    seeds = recall(client, "tailneedle")[:1]
    with closing(client.app.state.store._connect()) as db:
        db.execute(f"UPDATE source_policies SET {field}=? WHERE user_id=? AND raw_id=?",
                   (value, "u", seeds[0].source_raw_id))
        db.commit()
    assert run(client, seeds)[:2] == ([], [])


def test_missing_header_dependency_retains_seed(client):
    add(client, [table()])
    seed = recall(client, "tailneedle")[0]
    with closing(client.app.state.store._connect()) as db:
        db.execute("DELETE FROM source_spans WHERE user_id=? AND source_raw_id=? AND span_start=0",
                   ("u", seed.source_raw_id))
        db.commit()
    hits, groups, trace = run(client, [seed])
    assert hits == [seed] and not groups
    assert any(s["reason"] == "unavailable_dependency" for s in trace["skipped"])


def test_tampered_metadata_and_header_content_rejected(client):
    add(client, [table()])
    seed = recall(client, "tailneedle")[0]
    forged = seed.model_copy(update={"role": "assistant"})
    assert run(client, [forged])[:2] == ([], [])
    with closing(client.app.state.store._connect()) as db:
        db.execute("UPDATE source_spans SET content='forged header' WHERE user_id=? AND source_raw_id=? AND span_start=0",
                   ("u", seed.source_raw_id))
        db.commit()
    hits, groups, trace = run(client, [seed])
    assert hits == [seed] and not groups
    assert any(s["reason"] == "invalid_dependency" for s in trace["skipped"])


def test_raw_scan_limit_is_not_truncation_or_seed_loss(client):
    text = table() + "unstructured padding line\n" * (RAW_CHAR_LIMIT // 24 + 1)
    add(client, [text])
    seeds = recall(client, "tailneedle")[:1]
    hits, groups, trace = run(client, seeds)
    assert hits == seeds and not groups and trace["scanned_chars"] == 0
    assert trace["oversized_base_ids"] == [seeds[0].id]
    assert any(s["reason"] == "raw_scan_budget" for s in trace["skipped"])
    assert_final(client, hits)


def test_many_same_table_hits_have_only_directed_schema_dependencies(client):
    text = table(rows=650).replace("| item", "| rowneedle item")
    add(client, [text])
    seeds = recall(client, "rowneedle")
    assert len(seeds) > 8
    hits, groups, trace = run(client, seeds)
    assert len(groups) > 8
    edges = {g["seed_id"]: set(g["required_ids"]) for g in groups}
    for group in groups:
        closure, pending = {group["seed_id"]}, [group["seed_id"]]
        while pending:
            for dependency in edges.get(pending.pop(), ()):
                if dependency not in closure:
                    closure.add(dependency)
                    pending.append(dependency)
        assert len(closure) <= 8
        assert set(group["member_ids"]) == {group["seed_id"]} | set(group["required_ids"])
    assert trace["expanded_spans"] <= 50
    assert trace["source_ref_visits"] <= trace["logical_ref_limit"]


def test_renderer_budget_is_accounted_before_hard_binding(client):
    add(client, [table()])
    seeds = recall(client, "tailneedle")[:1]
    hits, groups, _ = run(client, seeds, renderer=lambda h: "metadata " * 2000 + h.content)
    assert hits == seeds and not groups


def test_fenced_table_and_unheaded_exception_do_not_bind(client):
    add(client, ["```text\n" + table() + "```\n", "A person must run unheadedneedle.\n"
                 + "Routine notes.\n" * 100 + "Exception: the weather may vary.\n"])
    for query in ("tailneedle", "unheadedneedle"):
        seeds = recall(client, query)[:1]
        hits, groups, _ = run(client, seeds)
        assert hits == seeds and not groups


def test_next_table_unit_is_not_required_by_previous_table(client):
    text = table("firsttail", unit="Units: kg") + table("secondtail", unit="Units: dollars")
    add(client, [text])
    seed = recall(client, "firsttail")[0]
    hits, groups, _ = run(client, [seed])
    # A boundary seed overlapping both tables is deliberately independent.
    for group in groups:
        assert all(hi <= text.index("Units: dollars") for _, hi in group["support_intervals"])
    assert hits[0] == seed


def test_code_exception_is_not_a_rule_dependency(client):
    text = "# Rules\nThe operator must run codeneedle.\n" + "Ordinary notes.\n" * 100
    text += "```text\nException: this is an example only.\n~~~\n```\n"
    add(client, [text])
    seeds = recall(client, "codeneedle")[:1]
    hits, groups, _ = run(client, seeds)
    assert hits == seeds and not groups


def test_max_extra_clamps_to_fifty_and_seed_limit_to_twenty(client):
    add(client, [rules(45) + f"Audit identifier {i}.\n" for i in range(23)])
    recalled = recall(client, "rulemarker")
    unique = {}
    for hit in recalled:
        unique.setdefault(hit.source_raw_id, hit)
    seeds = list(unique.values())
    assert len(seeds) == 23
    hits, groups, trace = run(client, seeds, seed_limit=999, max_extra=999)
    assert 0 < trace["expanded_spans"] <= 50
    with closing(client.app.state.store._connect()) as db:
        expected = sum(db.execute("SELECT length(content) FROM raw_messages WHERE user_id=? AND id=?",
                                  ("u", h.source_raw_id)).fetchone()[0] for h in seeds[:20])
    assert trace["scanned_chars"] == expected
    assert {g["seed_id"] for g in groups} <= {h.id for h in seeds[:20]}
    assert hits[:23] == seeds
    assert any(s["reason"] == "candidate_budget" for s in trace["skipped"])
    assert_final(client, hits)
