# AML 记忆服务 — Ubuntu 公网服务器（Docker + nginx）

域名：**test.memprism.com**（DNS 在 Cloudflare 由用户管理，A 记录指向本机公网 IP）。容器仅发布到宿主 **127.0.0.1:28717**（容器内 8787；宿主 8787 已被其他服务占用）；外网仅经 nginx 443 暴露 `/health`、`/add`、`/search`。

## 前置条件

| 项 | 建议 |
|----|------|
| OS | Ubuntu 22.04 / 24.04 LTS |
| vCPU / 内存 | 2C / 4GiB 起（Compose 上限 2 CPU、4GiB 内存） |
| 磁盘 | ≥40GiB（镜像、SQLite、WAL、日志、备份） |
| 网络 | 出站 HTTPS（DashScope embedding）；入站 22/80/443 |
| 密钥 | `AML_SERVICE_API_KEY`、`DASHSCOPE_API_KEY` 仅写入 `/etc/aml/aml.env`（600，root） |

**不要**在 UFW 或安全组中开放 **28717**。

## 一次性部署

### 0. 放置代码（首次，无 git 仓库时）

本机生成 tarball（只含镜像构建所需文件，并校验 app 源码与登记的候选身份一致）：

```powershell
cd F:\competitions\1AML\our-system
.\.venv312-tk14\Scripts\python.exe deploy\server\make_release.py --out <新文件>.tar.gz --origin ..\experiments\<候选>\candidate-origin.json
scp <新文件>.tar.gz <user>@<服务器IP>:/tmp/
```

服务器上：

```bash
sha256sum /tmp/<新文件>.tar.gz          # 与本机输出的 sha256 一致
sudo mkdir -p /opt/aml && sudo tar -xzf /tmp/<新文件>.tar.gz -C /opt/aml
```

### 1. 主机引导（幂等，需 sudo）

```bash
sudo bash /opt/aml/our-system/deploy/server/bootstrap-ubuntu.sh
```

创建目录：`/opt/aml`（代码）、`/var/lib/aml`（数据）、`/etc/aml`（环境）。防火墙只追加 22/80/443 规则，不重置已有规则；ufw 原本未启用时保持未启用，需要启用时用 `sudo AML_ENABLE_UFW=1 bash …/bootstrap-ubuntu.sh`。已装软件包不升级（`--no-upgrade`）；主机已有 nginx 时直接复用，不另装、不重启。

### 2. 环境文件

```bash
sudo cp /opt/aml/our-system/deploy/server/env.server.example /etc/aml/aml.env
sudo chmod 600 /etc/aml/aml.env
sudo editor /etc/aml/aml.env   # 填入 AML_SERVICE_API_KEY、DASHSCOPE_API_KEY
```

Hybrid 配置：`cycle2_dense` + `raw_hybrid_rrf`，库路径 `/var/lib/aml/aml-hybrid.sqlite`，DashScope 北京、`text-embedding-v4`、1024 维。

### 3. DNS（Cloudflare）

- 类型 **A**，名称 `test`（或完整 `test.memprism.com`），内容 = 服务器公网 IP。
- **签证书前**建议 **DNS only（灰云）**，避免代理干扰 HTTP-01。
- 若全程 **Proxied（橙云）**：Let's Encrypt 仍可用 nginx 插件；Cloudflare SSL/TLS 设为 **Full (strict)**，源站证书有效。

### 4. 代码与镜像

首次请自行 `git clone` 到 `/opt/aml`，或设置 `AML_GIT_REMOTE` 后：

```bash
export AML_ENV_FILE=/etc/aml/aml.env
sudo -E bash /opt/aml/our-system/deploy/server/deploy.sh --ref main
```

也可上传 tarball 后：

```bash
sudo bash deploy/server/deploy.sh --ref /tmp/1AML.tar.gz
```

成功后会等待 `http://127.0.0.1:28717/health` JSON 就绪；失败自动打印 `docker compose logs --tail 100` 并非零退出。上一版镜像保留为 `aml-cycle2-v4:server-previous`，回滚：

```bash
sudo bash deploy/server/deploy.sh --rollback
```

### 5. nginx 与 TLS

站点配置引用的证书要先存在。先装只含 80 端口验证路径的临时站点，签证书后换成正式站点（DNS 须已解析到本机）。以下按 Ubuntu 的 `sites-enabled` 布局；nginx 只 include `conf.d` 时，改放 `/etc/nginx/conf.d/test.memprism.com.conf`：

```bash
sudo mkdir -p /var/www/html
sudo cp /opt/aml/our-system/deploy/server/nginx-test.memprism.com.acme.conf \
  /etc/nginx/sites-available/test.memprism.com
sudo ln -sf /etc/nginx/sites-available/test.memprism.com /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot certonly --webroot -w /var/www/html -d test.memprism.com \
  --deploy-hook 'systemctl reload nginx'
sudo cp /opt/aml/our-system/deploy/server/nginx-test.memprism.com.conf \
  /etc/nginx/sites-available/test.memprism.com
sudo nginx -t && sudo systemctl reload nginx
sudo certbot renew --dry-run
```

证书默认路径：`/etc/letsencrypt/live/test.memprism.com/`（配置已对齐）。80 端口块保留 `/.well-known/acme-challenge/`，续期仍走 webroot。`/add`、`/search` 反代超时 1800s，对齐平台单次 30 分钟上限。

### 6. 验证

```bash
curl -fsS https://test.memprism.com/health
curl -sS -o /dev/null -w '%{http_code}\n' -X POST -H 'Content-Type: application/json' -d '{}' \
  https://test.memprism.com/search   # 无 Key 应 401
curl -sS -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer <服务Key>" \
  -H 'Content-Type: application/json' \
  -d '{"query":"probe","user_id":"u1","top_k":1}' \
  https://test.memprism.com/search   # 应 200（需库就绪且 DashScope 可用）
```

## 从本机 Windows 迁移数据库（可选）

本机 Hybrid 库默认：`F:\competitions\.aml-prod\db\aml-hybrid.sqlite`。

**方式 A — 拷贝文件**（需停写、含 WAL 一致性）：

```powershell
# 在本机：先停止 aml-service.ps1
scp F:\competitions\.aml-prod\db\aml-hybrid.sqlite user@SERVER:/tmp/
# 若有 WAL，一并 scp aml-hybrid.sqlite-wal / -shm，或 checkpoint 后单文件拷贝
```

在服务器：

```bash
sudo systemctl stop docker  # 或 docker compose stop aml
sudo install -m 0640 -o root -g root /tmp/aml-hybrid.sqlite /var/lib/aml/aml-hybrid.sqlite
sudo bash deploy/server/deploy.sh --rollback  # 或 deploy.sh 无 --ref 仅 up（若代码已在）
```

**方式 B — 在服务器重建**：空库启动后通过平台重新 Add，或使用 `deploy.rebuild`（需授权、可能产生 embedding 费用，见 `docs/20-v4-deploy.md`）。

## AML 平台填写

| 用途 | URL | 鉴权 |
|------|-----|------|
| 健康检查 | `https://test.memprism.com/health` | 无 |
| Add | `https://test.memprism.com/add` | `Authorization: Bearer <AML_SERVICE_API_KEY>` |
| Search | `https://test.memprism.com/search` | 同上 |

平台鉴权方式选 Bearer。`AML_AUTH_MODE=bearer` 时应用只接受 `Authorization: Bearer`；nginx 虽转发 `X-API-Key`，改用它需同时把 `AML_AUTH_MODE` 设为 `x-api-key`。

## 日常更新

```bash
cd /opt/aml && git pull   # 或 deploy.sh --ref <tag>
sudo AML_ENV_FILE=/etc/aml/aml.env bash our-system/deploy/server/deploy.sh
```

日志：

```bash
cd /opt/aml/our-system
sudo AML_ENV_FILE=/etc/aml/aml.env docker compose -f deploy/compose.yaml -f deploy/server/compose.server.yaml logs -f --tail 200
```

## 备份

```bash
sudo bash /opt/aml/our-system/deploy/server/backup.sh
```

Cron 示例（每日 03:15 UTC）：

```cron
15 3 * * * root /opt/aml/our-system/deploy/server/backup.sh >> /var/log/aml-backup.log 2>&1
```

保留最近 7 个时间戳目录于 `/var/backups/aml/`。

## 安全注意

- 密钥只在 `/etc/aml/aml.env`，不进镜像、不进 `deploy.sh` 参数、不进 git。
- 28717 仅 loopback；公网只走 TLS 反代。
- nginx `access_log off`，不记录请求体。

## 已知问题（现有 `deploy/Dockerfile` / Compose，未改仓库内文件）

1. **构建期需联网下载 tiktoken o200k**  
   当前 Dockerfile 在 `RUN python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"` 时拉取公开编码文件，并由 `RUN python -m deploy.runtime` 校验 SHA256 `446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d`。  
   **离线构建**可在 Dockerfile 的 pip install 之后、`get_encoding` 之前增加（示例补丁，勿含密钥）：

   ```dockerfile
   # 可选：从构建机复制已校验缓存（文件名 = sha1(ENCODING_URL)）
   COPY o200k_tiktoken_cache /opt/token-cache/fb374d419588a4632f3f557e76b4b70aebbca790
   RUN python -c "import hashlib, pathlib; p=pathlib.Path('/opt/token-cache/fb374d419588a4632f3f557e76b4b70aebbca790'); \
   assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()=='446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d'"
   RUN python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"
   ```

   Windows 源文件可参考：`%TEMP%\data-gym-cache\fb374d419588a4632f3f557e76b4b70aebbca790`（复制到构建上下文并重命名目录为 `o200k_tiktoken_cache`）。

2. **基础 `compose.yaml` 默认端口 8000 / 镜像 tag `local`**  
   服务器必须使用 `-f deploy/server/compose.server.yaml` 覆盖为宿主 **127.0.0.1:28717**（`ports: !override` 去掉基础文件的 8000 映射，需 Docker Compose ≥ 2.24.4）与 `aml-cycle2-v4:server`；`deploy.sh` 已固定双文件 compose。

3. **`deploy.sh` 默认 git 远程为占位符**  
   生产请先 `git clone <你的仓库> /opt/aml`，或导出 `AML_GIT_REMOTE` 后再 `--ref`。

4. **只读根文件系统 + SQLite**  
   数据目录挂载 `/var/lib/aml` 须对容器用户 uid/gid 10001 可写，`bootstrap-ubuntu.sh` 已按此建目录；若手工建过该目录，执行 `sudo chown 10001:10001 /var/lib/aml`。
