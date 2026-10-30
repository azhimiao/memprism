"""061 query-only policy checks against real SQLite FTS5/unicode61.

Synthetic coverage and explicit loss examples are engineering evidence only;
they do not measure benchmark recall, Answer quality, or official scores.
"""

from __future__ import annotations

import sqlite3

import pytest

from app.index import build_lexical_text, build_match_query


@pytest.fixture
def fts():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE VIRTUAL TABLE documents USING fts5(content, tokenize='unicode61')"
    )
    yield connection
    connection.close()


def insert(fts, documents):
    fts.executemany(
        "INSERT INTO documents(rowid, content) VALUES (?, ?)",
        [(i, build_lexical_text(text)) for i, text in enumerate(documents, 1)],
    )


def hits(fts, query, policy="balanced_v1"):
    expression = build_match_query(query, policy=policy)
    if expression is None:
        return []  # The retrieval caller must skip an empty MATCH expression.
    return [
        row[0]
        for row in fts.execute(
            "SELECT rowid FROM documents WHERE documents MATCH ? ORDER BY rowid",
            (expression,),
        )
    ]


def terms(query, policy="balanced_v1"):
    expression = build_match_query(query, policy=policy)
    return [] if expression is None else [part[1:-1] for part in expression.split(" OR ")]


@pytest.mark.parametrize(
    ("query", "legacy", "balanced"),
    [
        ("", [], []),
        ("a I x 中 😀 !", [], []),
        ("7", ["7"], ["7"]),
        ("北京", ["北京"], ["北京"]),
        ("北京 Ab 7 上海", ["Ab", "7", "北京", "上海"], ["北京", "Ab", "7", "上海"]),
        ("北京北京 7 7 ab ab AB", ["7", "ab", "AB", "北京", "京北"],
         ["北京", "京北", "7", "ab", "AB"]),
        ('"alpha" OR beta* NEAR(foo) -bar', ["alpha", "OR", "beta", "NEAR", "foo", "bar"],
         ["alpha", "OR", "beta", "NEAR", "foo", "bar"]),
        ("café naïve Straße ＡＢ １２ ١٢ e\u0301 😀 北京",
         ["caf", "na", "ve", "Stra", "北京"], ["caf", "na", "ve", "Stra", "北京"]),
    ],
)
def test_legacy_exact_and_short_equivalent_real_fts(fts, query, legacy, balanced):
    expected = " OR ".join(f'"{word}"' for word in legacy) or None
    assert build_match_query(query) == expected
    assert build_match_query(query, policy="legacy") == expected
    assert terms(query) == balanced
    assert set(balanced) == set(legacy)
    # One document per eligible term plus a disjoint negative control: all
    # terms must remain usable by the real parser, including reserved words.
    insert(fts, balanced + ["zznegativecontrolzz"])
    expected_hits = list(range(1, len(balanced) + 1))
    assert hits(fts, query) == expected_hits
    assert hits(fts, query, "legacy") == expected_hits


@pytest.mark.parametrize("policy", ["legacy", "balanced_v1"])
def test_quotes_operators_do_not_change_match_structure(fts, policy):
    insert(fts, ["alpha", "beta", "OR", "NEAR", "7", "北京", "unrelated"])
    query = '"alpha" OR (beta*) NEAR("北京", 7) "" - + : ^'
    assert hits(fts, query, policy) == [1, 2, 3, 4, 5, 6]


@pytest.mark.parametrize("query", ["", "a", "中", "😀 — \x00 \u200b", "１２ ١٢ 𠀀"])
def test_no_eligible_terms_does_not_broaden_search(fts, query):
    insert(fts, [query, "7 北京 alpha"])
    assert build_match_query(query, policy="balanced_v1") is None
    assert hits(fts, query) == []
    assert hits(fts, "7") == [2]  # The same FTS table remains queryable.


def test_document_lexical_representation_is_unchanged(fts):
    content = "北京上海 7 １２ ١٢ café"
    assert build_lexical_text(content) == content + "\n北京 京上 上海 7 １２ ١٢"
    insert(fts, [content, "unrelated"])
    stored = fts.execute("SELECT content FROM documents WHERE rowid = 1").fetchone()[0]
    for policy in ("legacy", "balanced_v1"):
        assert hits(fts, "上海 7", policy) == [1]
    assert fts.execute("SELECT content FROM documents WHERE rowid = 1").fetchone()[0] == stored


def test_single_character_numbers_and_no_single_ascii_letters(fts):
    query = "a 0 b 1 c 2 d 3 e 4 f 5 g 6 h 7 i 8 j 9"
    insert(fts, list("0123456789") + ["a", "z"])
    assert terms(query) == list("0123456789")
    assert hits(fts, query) == list(range(1, 11))


@pytest.mark.parametrize("count", [1, 2, 127, 128, 129, 255, 10000])
def test_cap_determinism_full_span_and_legacy_prefix(fts, count):
    words = [f"word{i}" for i in range(count)]
    query = " ".join(words)
    chosen = terms(query)
    assert terms(query, "legacy") == words[:128]
    assert build_match_query(query) == build_match_query(query, policy="legacy")
    assert len(chosen) == len(set(chosen)) == min(count, 128)
    assert terms(query) == chosen
    assert chosen[0] == words[0] and chosen[-1] == words[-1]
    if count <= 128:
        assert chosen == words
    else:
        # Require coverage in each quarter, not only prefix/tail preservation.
        indices = [int(word[4:]) for word in chosen]
        assert indices == sorted(indices)
        assert {min(3, i * 4 // count) for i in indices} == {0, 1, 2, 3}
    insert(fts, words + ["unrelated"])
    assert hits(fts, query) == [int(word[4:]) + 1 for word in chosen]


@pytest.mark.parametrize("chinese_first", [False, True])
def test_exact_128_mixed_terms_are_all_preserved(fts, chinese_first):
    words = [f"word{i}" for i in range(125)]
    chinese = ["北京", "上海", "主题"]
    ordered = chinese + words if chinese_first else words + chinese
    query = " ".join(ordered)
    assert terms(query) == ordered
    assert set(terms(query)) == set(terms(query, "legacy"))
    insert(fts, ordered + ["unrelated"])
    assert hits(fts, query) == hits(fts, query, "legacy") == list(range(1, 129))


@pytest.mark.parametrize("chinese_first", [False, True])
def test_minority_chinese_is_not_displaced_by_english(fts, chinese_first):
    english = " ".join(f"word{i}" for i in range(300))
    chinese = "北京 项目 上海"
    query = chinese + " " + english if chinese_first else english + " " + chinese
    chosen = terms(query)
    assert len(chosen) == 128
    assert {"北京", "项目", "上海", "word0", "word299"} <= set(chosen)
    insert(fts, ["北京", "项目", "上海", "word0", "word299", "unrelated"])
    assert hits(fts, query) == [1, 2, 3, 4, 5]
    assert hits(fts, query, "legacy") == [4]


def test_both_large_languages_cover_head_middle_tail(fts):
    # Distinct CJK pairs avoid coincidental vocabulary overlap between regions.
    chinese = [chr(0x4E00 + i) + "题" for i in range(200)]
    english = [f"word{i}" for i in range(200)]
    query = " ".join(token for pair in zip(chinese, english) for token in pair)
    chosen = terms(query)
    assert len(chosen) == 128
    for group in (chinese, english):
        kept = [word for word in chosen if word in group]
        assert len(kept) == 64
        assert kept[0] == group[0] and kept[-1] == group[-1]
        assert set(kept) & set(group[80:120])
    insert(fts, [" ".join(chinese[:20]), " ".join(chinese[80:120]),
                 " ".join(chinese[-20:]), " ".join(english[:20]),
                 " ".join(english[80:120]), " ".join(english[-20:]), "unrelated"])
    assert hits(fts, query) == [1, 2, 3, 4, 5, 6]
    # Original encounter order is restored even after language-specific quotas.
    positions = {word: i for i, word in enumerate(query.split())}
    assert [positions[word] for word in chosen] == sorted(positions[word] for word in chosen)


def test_minority_ascii_slots_transfer_from_chinese(fts):
    chinese = [chr(0x4E00 + i) + "题" for i in range(300)]
    query = "beginning " + " ".join(chinese[:150]) + " 7 " + " ".join(chinese[150:]) + " ending"
    assert len(terms(query)) == 128
    assert {"beginning", "7", "ending", chinese[0], chinese[-1]} <= set(terms(query))
    insert(fts, ["beginning", "7", "ending", chinese[0], chinese[-1], "unrelated"])
    assert hits(fts, query) == [1, 2, 3, 4, 5]


def test_repetition_does_not_consume_slots_or_shift_first_occurrence(fts):
    words = [f"word{i}" for i in range(200)]
    original = "北京 " + " ".join(words) + " 上海"
    repeated = "北京 " * 300 + " ".join(word + " " + word for word in words) + " 上海 " * 300
    assert build_match_query(original, policy="balanced_v1") == build_match_query(repeated, policy="balanced_v1")
    insert(fts, ["北京", "word0", "word199", "上海", "unrelated"])
    assert hits(fts, repeated) == [1, 2, 3, 4]


def test_natural_language_topics_survive_across_long_context(fts):
    # A realistic wrapper with a long list of distinct references between
    # questions, rather than a tail-only keyword appended to repeated padding.
    query = (
        "Passport renewal: which documents should be prepared? Reference records: "
        + " ".join(f"record{i}" for i in range(160))
        + ". Penicillin allergy: what precautions were recorded? More references: "
        + " ".join(f"record{i}" for i in range(160, 320))
        + ". 请核对上海项目。 Finally, explain the Kyoto rail itinerary."
    )
    insert(fts, ["Passport renewal", "Penicillin allergy precautions", "上海项目",
                 "Kyoto rail itinerary", "zznegativecontrolzz"])
    assert hits(fts, query) == [1, 2, 3, 4]
    assert len(terms(query)) == 128


def test_counterexample_balancing_loses_a_middle_critical_keyword(fts):
    # With 255 unique ASCII terms, 128 evenly spaced ranks are 0, 2, ..., 254.
    # The only relevant keyword at rank 127 is retained by legacy, lost here.
    words = [f"reference{i}" for i in range(255)]
    words[127] = "contraindication"
    query = " ".join(words)
    insert(fts, ["contraindication", "reference254", "unrelated"])
    assert hits(fts, query, "legacy") == [1]
    assert hits(fts, query) == [2]
    assert "contraindication" not in terms(query)


@pytest.mark.parametrize("policy", ["", "balanced", "BALANCED_V1", None, 1])
@pytest.mark.parametrize("query", ["", "北京 alpha"])
def test_unknown_policy_rejected_even_for_empty_input(query, policy):
    with pytest.raises(ValueError, match="Unknown lexical query policy"):
        build_match_query(query, policy=policy)


def test_policy_is_keyword_only():
    with pytest.raises(TypeError):
        build_match_query("alpha", "balanced_v1")
