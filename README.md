# MemPrism

MemPrism implements the Agent Memory Leaderboard (AML) Add/Search contract with
source-backed retrieval. This release publishes **evidence-architecture-062-03**.
It retains SQLite FTS5, DashScope `text-embedding-v4` (1024 dimensions), and
Reciprocal Rank Fusion. Its deployed configuration uses **no generative LLM
extraction, summarization, answer generation, or reranker**. Embedding API calls
still incur provider charges; this deployment is hybrid retrieval.

## API

| Endpoint | Behavior |
|---|---|
| `POST /add` | Synchronously stores user/assistant messages and required indexes for a user. A successful response acknowledges searchable data. Identical request replay is idempotent; conflicting reuse returns 409. |
| `POST /search` | Returns `{"data": [...]}` with at most `top_k` evidence records, capped at 100, restricted to the requested user. Evidence is grounded in saved source spans; deterministic date labels may accompany source text. It does not generate an answer. |
| `GET /health` | Reports control/data readiness and retrieval configuration hash. |

Add/Search use `Authorization: Bearer <service key>`. That key is distinct from
the leaderboard evaluation key and the embedding provider key. Never commit any
of these keys or benchmark conversations.

## Retrieval architecture

- Raw messages, source offsets/hashes, FTS postings, and dense vectors share a
  SQLite/WAL store. Add acknowledgement follows the required publication work.
- Lexical and exact dense candidates are fused with RRF (`k=60`). Per-user
  transaction coordination and snapshot checks protect concurrent Add/Search.
- Long query embedding uses bounded UTF-8 windows and an aggregate vector;
  requests beyond the total query budget are rejected instead of retried forever.
- The 062 policy adds a deterministic source-block catalog, user-scoped catalog
  FTS, history navigation, and groups that preserve rule references or table
  headers with supporting rows. All retrieval routes share a 200-candidate budget.
- Explicit state/conflict and disclosure markers guide evidence selection.
  These bounded rules do not constitute general semantic conflict resolution or
  a guarantee that every sensitive fact will be identified.
- Source provenance is checked before return. Source bodies are not instructions
  to the service. Retrieval does not inspect hidden labels or store query-generated
  facts. Engineering tests use synthetic inputs, not official evaluation answers.

The 062 policy is opt-in (`evidence_architecture_v1`). Legacy optional LLM and
reranking modules remain available as research code but are disabled in the
configuration below. Candidate manifests are immutable build-time records;
their historical `official_smoke_passed: false` is not a live evaluation status.

## Reproduce this configuration

Use Python 3.12.11 and the pinned dependencies. For a fresh database:

```bash
cd our-system
python -m venv .venv
. .venv/bin/activate
pip install --no-deps -r requirements-cycle2.txt
set -a
. deploy/server/env.062.example
set +a
export AML_DB_PATH="$PWD/data/aml.sqlite" AML_HOST=127.0.0.1
export AML_SERVICE_API_KEY='<choose-a-key>' DASHSCOPE_API_KEY='<your-DashScope-key>'
python -m deploy.start
```

For a fresh Docker deployment, follow
[the server guide](our-system/deploy/server/README.md), using
[`env.062.example`](our-system/deploy/server/env.062.example) as the environment
template. The older `env.server.example` is retained for historical configurations.
Do not apply these fresh-install commands over a live evaluation database. Existing
databases require backed-up, identity-checked index migration and certification;
changing an environment variable alone is not an upgrade procedure.

The 062 profile hash is
`c1f5d8441ef03bd3ef2bf5075af4cce125c23bdac7b4b2bf3cfc8acca385009a`.
Its 35 application files are identified by
[`candidate-03.json`](experiments/062-evidence-architecture/candidate-03.json).
The operator verified these files on the deployed service on 2026-10-30.
A health hash alone does not establish source identity. Publication does not
change the server or update a platform submission automatically.

## Verification and operational limits

```bash
cd our-system
python -m pytest tests -q -p no:cacheprovider
```

The 062 workspace application suite passed 1190 tests, with 35 pre-existing skips
for absent opt-in historical snapshots. These are offline engineering checks;
they do not prove Full capacity or official answer quality. Necessary historical
source manifests and minimal baseline code are included for reproducibility.

The operator reported a successful platform Smoke on 2026-10-30. No Full result
is claimed here. See [current submission notes](SUBMISSION-evidence-062.md) for
deployed limits and unresolved capacity verification. Existing 027 notes remain
historical and must not be used as the current deployment description.

Keep the same acknowledged database, indexes, receipts, and compatible code during
an interrupted evaluation. Logs help diagnose failure but cannot create an official
checkpoint. Recovery uses the original platform task when the platform permits it.
Provider processing before a local persistence failure can cause duplicate billing;
request idempotency does not guarantee exactly-once provider charges.

Benchmark data is not distributed. Operators are responsible for the official
retention deadline (currently within 30 days after the task), covering the live
database, WAL/SHM, backups, and any other data copies. Automatic cleanup is not
implemented. Preserve acknowledged data throughout permitted recovery.

## Attribution

This is a system implementation informed by the following prior work. These
references are not claims of having invented FTS, RRF, embeddings, or memory
governance. The repository contains our implementations and attributed runtime
dependencies, not vendored copies of these memory-system repositories.

| Prior work | Studied revision | Design influence |
|---|---|---|
| [InvMem](https://github.com/wenxiaof345-ctrl/vanilla-rag-memory) | `31ab7bf9` | Per-user lexical/dense fusion, retrieval views and context support. |
| [ActiveMemoryIndex](https://github.com/linxuhao/ActiveMemoryIndex) | `9bb0497c` | Preserve raw input, idempotent ingestion and separate derived views. |
| [Mem0](https://github.com/mem0ai/mem0) | `94c3fe9f` | User-scoped storage and explicit update/history concepts. |
| [MemMachine](https://github.com/MemMachine/MemMachine) | `d57f5cb3` | Separate searchable representation from returned source evidence. |
| [memfusion](https://github.com/huangpengtao00-dotcom/memfusion-memory) | `8578b646` | Separate fusion rank from relevance score. |
| [ChronoHybridMem](https://github.com/Tin11Mn/chrono-hybrid-mem) | `a8411527` | Query plans, supporting witnesses, temporal evidence. |
| [M-flow](https://github.com/FlowElement-xinliuyuansu/m_flow) | `0d585cda` | Typed/episode organization and procedural evidence. |
| [agent-memory-leaderboard-2026](https://github.com/cenjoy/agent-memory-leaderboard-2026) | `553f9c96` | Replacement versus coexistence and privacy filtering boundaries. |

Our 061/062 changes implement bounded deterministic evidence assembly, source
catalogs, dependency-preserving groups, and source-certified snapshots. These are
implementation differences, not demonstrated superiority over the referenced systems.

- RRF: G. V. Cormack, C. L. A. Clarke and S. Büttcher, *Reciprocal Rank Fusion
  outperforms Condorcet and individual Rank Learning Methods*, SIGIR 2009,
  [DOI:10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114).
- [SQLite FTS5](https://sqlite.org/fts5.html), Alibaba Cloud Model Studio/DashScope
  `text-embedding-v4`, and OpenAI `tiktoken` (`o200k_base`).
- FastAPI, Starlette, Uvicorn, Pydantic, HTTPX, NumPy and other pinned libraries
  are listed in `our-system/requirements-cycle2.txt`.
- Evaluation contract: [AML API Guide](https://agentmemories.ai/api-guide) and
  [official harness](https://github.com/AML-memory/agent-memory-leaderboard).

## License

GNU Affero General Public License v3.0; see [LICENSE](LICENSE).
