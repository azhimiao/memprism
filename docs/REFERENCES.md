# 研究参考与技术归属


MemPrism 的实现贡献集中在**来源绑定的证据目录、依赖保持的证据组装、事务发布与快照核验，以及针对流式读写的工程实现**。FTS、Embedding、RRF 和既有记忆系统的基础思想均有明确来源；本仓库不声称发明这些通用方法，也不以本次结果声明优于所有参考系统。

| 参考工作 | 研究版本 | 借鉴方向与本项目的实现取舍 |
|---|---|---|
| [InvMem](https://github.com/wenxiaof345-ctrl/vanilla-rag-memory) | `31ab7bf9` | 用户内词面／语义融合；本项目加入来源核验及确定性证据组装 |
| [ActiveMemoryIndex](https://github.com/linxuhao/ActiveMemoryIndex) | `9bb0497c` | 原始输入与派生表示分离；本项目围绕同步发布和回执实现 |
| [Mem0](https://github.com/mem0ai/mem0) | `94c3fe9f` | 用户范围、更新与历史；当前配置采用确定性规则而非生成式抽取 |
| [MemMachine](https://github.com/MemMachine/MemMachine) | `d57f5cb3` | 区分检索表示与返回证据；返回内容绑定保存的来源 |
| [memfusion](https://github.com/huangpengtao00-dotcom/memfusion-memory) | `8578b646` | 分离融合排名和相关性含义 |
| [ChronoHybridMem](https://github.com/Tin11Mn/chrono-hybrid-mem) | `a8411527` | 查询计划、支持证据及时间来源 |
| [M-flow](https://github.com/FlowElement-xinliuyuansu/m_flow) | `0d585cda` | 过程性记忆组织；保留规则和引用的依赖关系 |
| [agent-memory-leaderboard-2026](https://github.com/cenjoy/agent-memory-leaderboard-2026) | `553f9c96` | 新旧状态共存／替换与披露边界 |

RRF 原始工作：G. V. Cormack, C. L. A. Clarke, S. Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*, SIGIR 2009，[DOI](https://doi.org/10.1145/1571941.1572114)。运行依赖包括 [SQLite FTS5](https://sqlite.org/fts5.html)、DashScope `text-embedding-v4`、OpenAI `tiktoken`、FastAPI、HTTPX 等；精确版本见 [requirements-cycle2.txt](../our-system/requirements-cycle2.txt)。
