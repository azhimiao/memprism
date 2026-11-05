# 复现与启动

## 固定运行身份

本发布将完成 Full 时使用的 local-103 运行文件映射到 `our-system/`。`releases/runtime-103.json` 列出 50 份文件的 SHA256；其中 42 份为 app 模块，其余为运行入口和依赖清单。配置模板 `env.evaluated.example` 对应本次配置，不应以旧的 027 或 062 提交说明代替当前运行源码。

Python 固定为 3.12.11，依赖以 `our-system/requirements-cycle2.txt` 为准。运行时还要求 SQLite 支持 FTS5，并核对 tokenizer 资产。原始评测数据库不公开，以下过程复现引擎和配置，不重放或重新发起官方评测。

## 新环境

在 Bash 中从仓库根目录执行：

```bash
set -e
cd our-system
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --no-deps -r requirements-cycle2.txt
export TIKTOKEN_CACHE_DIR="$PWD/.token-cache"
python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"
```

先确认 `python --version` 确实为 3.12.11。仅 `python3.12` 的主次版本相同并不足以通过固定运行时检查。上述依赖与编码资产准备需要网络；后续离线检查本身不发起模型调用。

## 全新数据库

```bash
set -e
set -a
. deploy/server/env.evaluated.example
set +a
export AML_DB_PATH="$PWD/data/aml.sqlite"
export AML_HOST=127.0.0.1 AML_PORT=8787
export AML_SERVICE_API_KEY='<choose-a-service-key>'
export DASHSCOPE_API_KEY='<your-DashScope-key>'

test ! -e "$AML_DB_PATH"
python -m deploy.runtime --configuration
python -c "from deploy.runtime import build_app; build_app()"
python -m deploy.start
```

此处的 `build_app` 仅用于初始化**不存在的全新库**。配置核验会建立 provider 对象，但空库初始化不执行文档或查询 Embedding。启动入口随后使用评测时的服务工厂。不要在其他 Add/Search 进程正在写入同一库时运行初始化。

PowerShell 使用等价环境变量语法 `$env:AML_DB_PATH = ...`，虚拟环境激活脚本为 `.venv/Scripts/Activate.ps1`。不要把本页的 Bash 环境赋值直接粘贴到 PowerShell。

## 已有数据库

保留来源、发布元数据、回执以及配置身份；停写后备份主库和必要 WAL/其他关联文件，再按对应版本的迁移流程操作。不要覆盖正在运行的工作库，不要把较旧备份当成后续成功 Add 已保留的证明。

服务工厂会拒绝缺少当前配置身份的已有库。迁移或升级必须核对数据库与源码身份、原有成功回执和实际公网通路；修改环境变量不等于完成旧索引升级。

## 最小连通性检查

服务启动后，从另一终端执行：

```bash
curl --fail --silent --show-error http://127.0.0.1:8787/health
curl --silent --show-error --output /dev/null --write-out '%{http_code}\n' \
  --header 'Content-Type: application/json' --data '{}' \
  http://127.0.0.1:8787/search
```

正常情况下健康返回 200，匿名 Search 返回 401。它们只验证相应通路，不证明有效业务召回、供应商容量或正式质量。有效 Add/Search 需要服务鉴权和相应协议输入，并可能产生 Embedding 费用。

## 测试

在独立临时目录运行，每个进程使用自己的 `--basetemp`：

```bash
cd our-system
python -m pip install -r requirements-verification.txt
python -m pytest tests -q -p no:cacheprovider --basetemp="$(mktemp -d)/pytest"
```

合成测试涵盖协议、隔离、重复请求、来源有效性、预算、发布和读写协调等。历史可选快照缺失会按原测试逻辑跳过；跳过项不计为通过。专项测试和本次公开树的实际结果见 `releases/verification.json`。测试不会代表平台 Answer/Judge，也不是对官方成绩的复现。

## 外部部署

引擎只需能够提供受保护的 HTTP Add/Search。HTTPS 入口可由部署方的反向代理提供。本次云入口与本机反向隧道拓扑见 `ARCHITECTURE.md`；仓库不提供参赛方密钥、站点私有配置或原始数据。

旧 `our-system/deploy/server/README.md` 保留了早期容器部署方式，其固定发行假设属于历史版本。当前 103 的可复现入口以本页和 `env.evaluated.example` 为准，不能照搬旧发行身份直接升级现有服务。
