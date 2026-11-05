> Historical submission notes. For the Full completion snapshot, see [SUBMISSION-current.md](SUBMISSION-current.md).

# MemPrism — evidence-architecture-062-03

The platform's existing version label is `memprism-hybrid-037`; the current
implementation revision is **evidence-architecture-062-03**. Use the fixed commit
containing this file and the 35-file candidate manifest to identify the submission.
The older `8af992301afab498be696932d1cf38a1c1bebd70` commit identifies 037 code,
not the currently verified 062 implementation. This publication does not rename
the platform version or change its frozen submission records.

## Interface and configuration

- Add: `https://test.memprism.com/add` (POST, synchronous acknowledgement).
- Search: `https://test.memprism.com/search` (POST, source evidence).
- Health: `https://test.memprism.com/health`.
- Add/Search authentication: Bearer service token, supplied privately to AML.
  No token or leaderboard key is included here.
- Profile: `cycle2_dense` / `raw_hybrid_rrf`; policy: `evidence_architecture_v1`.
- Configuration hash: `c1f5d8441ef03bd3ef2bf5075af4cce125c23bdac7b4b2bf3cfc8acca385009a`.
- Embedding: DashScope `text-embedding-v4`, 1024 dimensions. Add/Search use no
  generative LLM and no reranker. This is disclosed as a no-LLM method, not a
  claim that an alternative LLM complies with the academic model restriction.
- Shared service concurrency limit: 32; selected platform limits: Add16/Search16;
  returned records: at most 100, within 1,000,000 characters and 65,536 packed tokens.
- Current deployment container limits: 2 CPUs and 4 GiB RAM. SQLite/WAL is stored
  on persistent host storage; diagnostic logs are bounded and contain metadata.

## Original work and modifications

[README attribution](README.md#attribution) identifies the studied projects,
authors via their repositories, revisions, retrieval method paper and libraries.
The system preserves original evidence with deterministic source identities.
Changes since 037 include diagnostic events, bounded long-query embedding,
deterministic source-unit packing, trusted date labels, a source-block catalog,
history navigation, rule/table support groups, and bounded state/privacy handling.
No third-party memory-system source is vendored, and no benchmark conversations,
gold answers, task-specific answer tables or live credentials are distributed.

## Validation and remaining capacity work

The operator verified the deployed 35-file identity and reported a successful
Smoke on 2026-10-30. Local application engineering tests passed separately.
These observations do not certify Full performance, provider quota or storage.
Full-scale database/index/WAL growth, backup space, provider balance/rate limits
and target-host load still require an explicit operational capacity check.

No Full result or broad semantic privacy guarantee is claimed. Preserve live
evaluation data and the same compatible implementation during interruptions;
resume an existing task only where the platform offers a compatible checkpoint.
The operator must update any editable platform code reference/description to
this fixed release before relying on it as the current submission identity.
