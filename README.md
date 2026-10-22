# MemPrism

MemPrism is a candidate memory service for the Agent Memory Leaderboard (AML) Cycle 2, open-source
method category. It implements the leaderboard's HTTP contract:

| Endpoint | Behaviour |
|---|---|
| `POST /add` | Synchronously persists one batch of `user`/`assistant` messages for a `user_id`. When it returns 200 the messages are stored and searchable. Idempotent per `request_id`: the same body replays the first result, a different body returns 409. Other roles are rejected with 422. |
| `POST /search` | Returns `{"data": [...]}`: at most `top_k` evidence items from that `user_id` only, ordered by relevance. `content` is text copied verbatim from stored messages; nothing is generated. |
| `GET /health` | Readiness plus the retrieval `config_hash`; no authentication. |

`/add` and `/search` require `Authorization: Bearer <service key>`.

## Method

- Messages are split into exact spans of the original text and stored in SQLite (WAL) with an FTS5
  index and 1024-dimensional dense vectors from Alibaba Cloud Model Studio `text-embedding-v4`.
  Every query is scoped to the requesting `user_id`.
- Search takes up to 100 lexical (FTS5) and 100 dense candidates and fuses them with Reciprocal Rank
  Fusion (k = 60), returning up to 100 items within the configured return budget.
- Add and Search call no LLM and no reranker; the only model is the embedding model.
- A lifecycle coordinator fences concurrent Adds: a Search that overlaps a commit for the same user
  retries against a stable snapshot and returns 503 if the snapshot keeps changing, so a result never
  mixes two commits.

The existing 027 deployment template is `our-system/deploy/server/env.server.example`
(`AML_PROFILE=cycle2_dense`, `AML_RETRIEVAL_MODE=raw_hybrid_rrf`); `/health` reports its
`config_hash` (`08e945a951ade441…`). Experimental mechanisms in the code base (sentence index,
neighbour/parent expansion, typed or option routing, temporal projection, reranking, atomic memories,
governance) are off in this configuration.

## Run

### Docker (the deployed path)

The server deployment (Ubuntu, Docker Compose, nginx, Let's Encrypt) is described in
[`our-system/deploy/server/README.md`](our-system/deploy/server/README.md) (Chinese). In short:

```bash
cp our-system/deploy/server/env.server.example /etc/aml/aml.env   # fill AML_SERVICE_API_KEY, DASHSCOPE_API_KEY
AML_ENV_FILE=/etc/aml/aml.env bash our-system/deploy/server/deploy.sh
curl -fsS http://127.0.0.1:28717/health
```

### Python

Python 3.12.11 with the pinned dependencies:

```bash
cd our-system
python -m venv .venv && . .venv/bin/activate
pip install --no-deps -r requirements-cycle2.txt
set -a; . deploy/server/env.server.example; set +a
export AML_DB_PATH="$PWD/data/aml.sqlite" AML_HOST=127.0.0.1 \
       AML_SERVICE_API_KEY=<choose a key> DASHSCOPE_API_KEY=<your DashScope key>
python -m deploy.start
```

## Tests

```bash
cd our-system
python -m pytest -q -p no:cacheprovider
```

The application tests run offline, with engineering embedding doubles. The 037 application was
verified locally with 1022 passed and 35 existing skipped tests; the public export is checked again
before publication. This is engineering validation, not an AML score or a Full capacity guarantee.
The skipped tests require opt-in historical build-v3/build-v5 snapshots absent from this repository.

`tests/test_optional_adversarial.py` checks the source identity chain. Public manifests include the
030/031/036 ancestors and `experiments/037-retrieval-performance/candidate-final.json`; the current
29 application files must match that manifest. Historical manifests retain their original identity.
The frozen 037 `baseline/app/segments.py` is included for the performance equivalence regression.

## Data handling

Benchmark data is not included in this repository. The service stores only what the platform sends
to `/add`, in a single SQLite database (`/var/lib/aml/aml-hybrid.sqlite` in the server deployment).
Operators must delete evaluation data within 30 days after the task by removing that database
(with its `-wal`/`-shm` files) and its backups under `/var/backups/aml/`. This is an operational
retention requirement; automatic cleanup is not implemented. Preserve acknowledged memory
and indexes throughout an interrupted evaluation and its recovery.

## Release and deployment status

This repository now includes the locally verified 037 concurrency/performance candidate,
`cycle2-exact-retrieval-performance-v2`. Its manifest SHA256 is
`bf78e687f3485f2403365034c57849226b6a32257c204ec9615e9f0d8cb2f66e`.
Publishing this code does not deploy it. The existing public service was originally released
from commit `c37de8791805227fa43181bdf0013e2e6f7a33c7` (027); its current source identity
must be verified by the operator before claiming an exact deployed commit. A health/config
hash alone does not prove application source identity.

For an access request concerning that old deployment, use the fixed old commit and the
[027 submission notes](SUBMISSION-hybrid-027.md), after verifying the server matches it.
Do not identify the new main branch as deployed until the operator has deployed and verified it.

The 037 changes serialize per-user preparation and reduce repeated readiness-count and exact
dense-scoring work. Six final local mixed-load rounds completed 384/384 request positions;
measured mixed-load Search p95 remained about 27-29 seconds for FTS and 98-103 seconds for
Hybrid. These were local handlers with cached real vectors, not public HTTP or live supplier
capacity tests. Production model dispatch can still be billed twice if a supplier processes a
request before local persistence fails; request-id idempotency is not exactly-once billing.

The research configuration `cycle2_raw` / `raw_fts` uses no embedding or LLM API. A small local
unseen-source comparison favored keeping FTS as a cost-first candidate and Hybrid optional;
this does not establish FTS as universally better. The supplied `deploy.start`/`deploy.runtime`
launcher still requires a formal dense configuration and cannot launch FTS merely by changing
the two profile/mode variables. A compatible production entry point must be provided and
verified before deploying FTS. LLM extraction remains optional and disabled by default.

No official Smoke/Full result is established by this release. The operator handles deployment,
capacity verification, and official evaluation separately. Existing confirmed raw data, indexes,
request receipts and accounting state must be backed up and retained during upgrades.

## Attribution

- Reciprocal Rank Fusion: G. V. Cormack, C. L. A. Clarke and S. Büttcher, "Reciprocal Rank Fusion
  outperforms Condorcet and individual Rank Learning Methods", SIGIR 2009.
- Lexical retrieval: SQLite FTS5.
- Embeddings: Alibaba Cloud Model Studio (DashScope) `text-embedding-v4`.
- Token counting: OpenAI `tiktoken` (`o200k_base`).
- Runtime libraries: FastAPI, Starlette, Uvicorn, Pydantic, HTTPX, NumPy (versions in
  `our-system/requirements-cycle2.txt`).

- HTTP contract and evaluation: the AML [API guide](https://agentmemories.ai/api-guide) and
  [official harness](https://github.com/AML-memory/agent-memory-leaderboard).

The public memory systems below were studied at the listed revisions. Their mechanisms were
reimplemented; no third-party source code is included.

| Project | Revision studied | Mechanisms that informed MemPrism |
|---|---|---|
| [InvMem](https://github.com/wenxiaof345-ctrl/vanilla-rag-memory) | `31ab7bf9` | Per-user lexical + dense retrieval fused with RRF (027 template; retained in code). Query/option views, context windows (experimental, off). |
| [ActiveMemoryIndex](https://github.com/linxuhao/ActiveMemoryIndex) | `9bb0497c` | Store raw messages first; per-user idempotent Add (027 template; retained in code). Derived facts, single rewrite (experimental, off). |
| [Mem0](https://github.com/mem0ai/mem0) | `94c3fe9f` | Explicit user scoping (027 template; retained in code). Change history, thresholds (experimental, off). |
| [MemMachine](https://github.com/MemMachine/MemMachine) | `d57f5cb3` | Index representation kept separate from returned source text (027 template; retained in code). Context expansion (experimental, off). |
| [memfusion](https://github.com/huangpengtao00-dotcom/memfusion-memory) | `8578b646` | Fusion rank kept separate from relevance score (027 template; retained in code). Query expansion (experimental, off). |
| [ChronoHybridMem](https://github.com/Tin11Mn/chrono-hybrid-mem) | `a8411527` | Query plans, support witnesses, temporal views (experimental, off). |
| [M-flow](https://github.com/FlowElement-xinliuyuansu/m_flow) | `0d585cda` | Typed routing, episode/facet structure, procedures (experimental, off). |
| [agent-memory-leaderboard-2026](https://github.com/cenjoy/agent-memory-leaderboard-2026) | `553f9c96` | Type-aware replacement vs. coexistence, privacy gate position (experimental, off). |

## License

GNU Affero General Public License v3.0; see [LICENSE](LICENSE).
