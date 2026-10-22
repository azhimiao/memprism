# MemPrism：现有线上Hybrid版本申请说明

本文件对应旧版本`memprism-hybrid-027`。提交申请前，运营方应确认服务器代码与以下固定Commit一致；仅health的config_hash相同不能证明源码一致。仓库main的新代码尚未部署，不应将新main Commit填作当前线上版本。

以下正文可复制到“提交说明与运行指南”：

---

系统：MemPrism

申请类别：学术榜／开源方法榜；评测赛道：文本记忆。

版本：memprism-hybrid-027。

公开仓库：https://github.com/azhimiao/memprism

固定Commit：c37de8791805227fa43181bdf0013e2e6f7a33c7。

版本材料：https://github.com/azhimiao/memprism/tree/c37de8791805227fa43181bdf0013e2e6f7a33c7

接口：Add为https://test.memprism.com/add；Search为https://test.memprism.com/search；健康检查为https://test.memprism.com/health。

Add、Search采用Authorization: Bearer认证，专用服务密钥通过表单密钥字段提供；Health无需认证。使用同步Add，Add状态地址留空。

方法：SQLite持久化原始历史消息与原文片段，使用FTS5词法检索和text-embedding-v4语义检索，以RRF融合排序。向量1024维，区分document/query。该版本配置为cycle2_dense/raw_hybrid_rrf。Add和Search不调用LLM，不启用LLM抽取或重排序；Search只返回user_id范围内的历史原文证据，不生成最终答案。

协议：Add写入与索引完成后返回成功；request_id结合请求体校验用于幂等重放；Search返回按相关性排序的data数组，包含id、content等字段，最多top_k条，正式top_k=100；不同user_id隔离。

运行与限制：HTTPS经nginx转发至Docker应用；配置的总请求准入上限为32，拟使用Add并发16、Search并发16。完整Full规模尚未在目标主机验收；供应商异常或持续快照变化可能返回503。平台重试须复用同一request_id与请求体，服务持久保留已确认写入状态。应用成功回放幂等不等于外部向量服务严格只计费一次。

来源披露：RRF参考Cormack、Clarke、Büttcher的SIGIR 2009论文；词法检索使用SQLite FTS5，向量来自阿里云DashScope text-embedding-v4。README的Attribution章节列出InvMem、ActiveMemoryIndex、Mem0、MemMachine、memfusion、ChronoHybridMem、M-flow等参考项目的链接、修订和机制关系。复用机制的实现与来源说明见固定Commit的README；许可证为AGPL-3.0。未发布独立技术论文，技术说明以仓库README及部署文档为准。

运行说明见固定Commit下README和our-system/deploy/server/README.md。目前没有该版本的官方Full成绩。
