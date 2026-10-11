# AML 记忆服务 — 本机常驻与 Cloudflare 命名隧道

本目录提供 Windows PowerShell 5.1 脚本，在本机运行 AML Memory 服务（仅监听 `127.0.0.1:8787`），并通过 Cloudflare 命名隧道暴露公网 HTTPS 入口。**脚本不会把 API Key 打印到控制台**；DashScope Key 与平台服务 Key 分别保存在 `%LOCALAPPDATA%\AML-Memory\credentials\` 的 DPAPI 保险库中。

运行根目录：`F:\competitions\.aml-prod\`（数据库、日志、PID、隧道配置）。

## 一次性部署步骤

1. **安装 cloudflared**（默认路径 `C:\Program Files (x86)\cloudflared\cloudflared.exe`）。
2. **Cloudflare 登录**（需交互，生成账户证书）：
   ```powershell
   cloudflared tunnel login
   ```
3. **配置 DashScope Key**（若尚未配置）：使用 `experiments/dashscope-credentials.ps1` 中的 `Save-AmlDashScopeKey`，或确保 `%LOCALAPPDATA%\AML-Memory\credentials\dashscope.dpapi` 已存在。
4. **创建隧道与 DNS**：
   ```powershell
   & 'F:\competitions\1AML\our-system\deploy\local\aml-tunnel.ps1' -Action Setup -Hostname test.memprism.com -TunnelName aml-test
   ```
5. **启动 AML 服务**（Hybrid 模式，含 dense + DashScope v4）：
   ```powershell
   & 'F:\competitions\1AML\our-system\deploy\local\aml-service.ps1' -Action Start -Mode Hybrid
   ```
6. **启动隧道**：
   ```powershell
   & 'F:\competitions\1AML\our-system\deploy\local\aml-tunnel.ps1' -Action Start
   ```
7. **验证**（需隧道与服务均已运行）：
   - 公网：`https://test.memprism.com/health` 应返回 200
   - 本机：`http://127.0.0.1:8787/health`

## 日常启停

| 操作 | 命令 |
|------|------|
| 服务状态 | `aml-service.ps1 -Action Status` |
| 停止服务 | `aml-service.ps1 -Action Stop` |
| 隧道状态 | `aml-tunnel.ps1 -Action Status` |
| 停止隧道 | `aml-tunnel.ps1 -Action Stop` |

日志目录：`F:\competitions\.aml-prod\logs\`（服务 `service-*.log`，隧道 `tunnel-*.log`；各保留最近 10 个）。

## 将服务 Key 交给 AML 平台

平台表单需要 **Bearer 服务 Key**（与 `AML_SERVICE_API_KEY` 相同，脚本自动生成并 DPAPI 存储）：

```powershell
& 'F:\competitions\1AML\our-system\deploy\local\aml-service.ps1' -Action ShowKeyOnce
```

输出仅为 `copied_to_clipboard`，Key 已写入剪贴板，请粘贴到平台配置。**不要在聊天或日志中粘贴 Key。**

轮换 Key（需重启服务）：

```powershell
& 'F:\competitions\1AML\our-system\deploy\local\aml-service.ps1' -Action RotateKey
& 'F:\competitions\1AML\our-system\deploy\local\aml-service.ps1' -Action Stop
& 'F:\competitions\1AML\our-system\deploy\local\aml-service.ps1' -Action Start
```

## 公网 API 地址

在隧道 Hostname 为 `test.memprism.com` 时：

| 用途 | URL | 鉴权 |
|------|-----|------|
| 健康检查 | `https://test.memprism.com/health` | 无 |
| 写入记忆 | `https://test.memprism.com/add` | `Authorization: Bearer <服务 Key>` |
| 检索记忆 | `https://test.memprism.com/search` | `Authorization: Bearer <服务 Key>` |

## 本机不可睡眠 / 断网

隧道与服务依赖本机持续在线。建议：

- 电源：阻止睡眠（**以下命令需管理员，本 README 不自动执行**）：
  ```powershell
  powercfg /change standby-timeout-ac 0
  powercfg /change hibernate-timeout-ac 0
  powercfg /change monitor-timeout-ac 30
  ```
- 网络：保持有线/Wi‑Fi 稳定；Cloudflare 隧道断开后公网入口不可用，本机 `127.0.0.1:8787` 仍可访问。

## 回滚与切换 Mode

- **Hybrid**（默认）：`cycle2_dense` + `raw_hybrid_rrf`，数据库 `F:\competitions\.aml-prod\db\aml-hybrid.sqlite`。
- **Fts**：`deploy.runtime` 的 `--configuration` 要求 **formal dense 配置**（`configured_provider` 检查 `dense` 路由与 embedding），**当前 `-Mode Fts` 会输出 `not_supported` 并退出**；纯 FTS 需另立入口或后续改造，**请勿对 FTS 复用 Hybrid 库文件**（`config_hash` 不同）。若将来支持 FTS，将使用 `aml-fts.sqlite`。

回滚步骤：

1. `aml-tunnel.ps1 -Action Stop`
2. `aml-service.ps1 -Action Stop`
3. 如需恢复旧 Key，在轮换前自行备份 `service.dpapi`（DPAPI 绑定当前 Windows 用户）。

## 环境前提

- Python：`F:\competitions\1AML\our-system\.venv312-tk14\Scripts\python.exe`
- 工作目录：`F:\competitions\1AML\our-system`
- **Tiktoken 缓存**：启动前需已有校验通过的缓存目录，可通过用户/系统环境变量 `TIKTOKEN_CACHE_DIR` 或 `DATA_GYM_CACHE_DIR` 指定，否则使用 `%TEMP%\data-gym-cache`（与 `deploy.runtime` 一致）。

## 已知限制

- 未执行本 README 中的 `cloudflared` / 服务启动时，仅做脚本与离线检查。
- `Start` 会调用本机 `127.0.0.1` 健康检查；`aml-tunnel.ps1 -Action Status` 在隧道运行时会尝试 `https://<hostname>/health`。
- Fts 模式暂不支持（见上）。
- 服务 Key 与 DashScope Key 仅存在于子进程环境；脚本进程在启动后会清除进程级 Secret 变量。
