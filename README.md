# MemPrism

MemPrism is the memory service submitted to the Agent Memory Leaderboard (AML) Cycle 2, open-source
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

The deployed configuration is `our-system/deploy/server/env.server.example`
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

The suite runs offline (embedding providers are replaced by doubles). At the first published commit it
reports 892 passed and 35 skipped; the skipped tests are opt-in checks against frozen snapshots
(`experiments/015-final-frontier/build-v3` and `build-v5`) that are not part of this repository.

`tests/test_optional_adversarial.py` binds the application sources to recorded source identities.
The files under `experiments/` are that chain: earlier source snapshots and their hash manifests
(`experiments/027-coordinator-lock-wait/candidate-origin.json` is the current one). They contain code
and hashes only.

## Data handling

Benchmark data is not included in this repository. The service stores only what the platform sends
to `/add`, in a single SQLite database (`/var/lib/aml/aml-hybrid.sqlite` in the server deployment).
Evaluation data is deleted within 30 days after the task by removing that database (with its
`-wal`/`-shm` files) and its backups under `/var/backups/aml/`.

## Status

No official leaderboard result has been published for this configuration yet.

## Attribution

- Reciprocal Rank Fusion: G. V. Cormack, C. L. A. Clarke and S. Büttcher, "Reciprocal Rank Fusion
  outperforms Condorcet and individual Rank Learning Methods", SIGIR 2009.
- Lexical retrieval: SQLite FTS5.
- Embeddings: Alibaba Cloud Model Studio (DashScope) `text-embedding-v4`.
- Token counting: OpenAI `tiktoken` (`o200k_base`).
- Runtime libraries: FastAPI, Starlette, Uvicorn, Pydantic, HTTPX, NumPy (versions in
  `our-system/requirements-cycle2.txt`).

- HTTP contract and evaluation: the AML [API guide](https://agentmemoryleaderboard.ai/api-guide) and
  [official harness](https://github.com/AML-memory/agent-memory-leaderboard).

The public memory systems below were studied at the listed revisions. Their mechanisms were
reimplemented; no third-party source code is included.

| Project | Revision studied | Mechanisms that informed MemPrism |
|---|---|---|
| [InvMem](https://github.com/wenxiaof345-ctrl/vanilla-rag-memory) | `31ab7bf9` | Per-user lexical + dense retrieval fused with RRF (deployed). Query/option views, context windows (experimental, off). |
| [ActiveMemoryIndex](https://github.com/linxuhao/ActiveMemoryIndex) | `9bb0497c` | Store raw messages first; per-user idempotent Add (deployed). Derived facts, single rewrite (experimental, off). |
| [Mem0](https://github.com/mem0ai/mem0) | `94c3fe9f` | Explicit user scoping (deployed). Change history, thresholds (experimental, off). |
| [MemMachine](https://github.com/MemMachine/MemMachine) | `d57f5cb3` | Index representation kept separate from returned source text (deployed). Context expansion (experimental, off). |
| [memfusion](https://github.com/huangpengtao00-dotcom/memfusion-memory) | `8578b646` | Fusion rank kept separate from relevance score (deployed). Query expansion (experimental, off). |
| [ChronoHybridMem](https://github.com/Tin11Mn/chrono-hybrid-mem) | `a8411527` | Query plans, support witnesses, temporal views (experimental, off). |
| [M-flow](https://github.com/FlowElement-xinliuyuansu/m_flow) | `0d585cda` | Typed routing, episode/facet structure, procedures (experimental, off). |
| [agent-memory-leaderboard-2026](https://github.com/cenjoy/agent-memory-leaderboard-2026) | `553f9c96` | Type-aware replacement vs. coexistence, privacy gate position (experimental, off). |

## License

GNU Affero General Public License v3.0; see [LICENSE](LICENSE).
