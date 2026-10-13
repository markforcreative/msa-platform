# MSA Platform

基于 FastAPI、PostgreSQL、ONNX Runtime、Whisper、BERT、MMSA-FET/OpenFace 和 Ollama 的多模态情感分析与对话后端。

> 本仓库中由维护者原创的源代码以 GPL-3.0-only 发布；第三方组件沿用各自许可证。运行所需的模型权重、OpenFace 资产、本地共享库、真实数据和 `.env` 不在本次发布范围内；详情见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 架构

```text
浏览器或客户端
      │ 127.0.0.1:8000
      ▼
FastAPI / AI 推理容器
  ├── ONNX 多模态情感模型
  ├── Whisper 语音识别
  ├── BERT 文本特征
  ├── MMSA-FET / OpenFace
  ├── Ollama（宿主机:11434）
  └── PostgreSQL（Compose 内部网络:5432）
```

## 当前安全边界

此版本只适合本机开发和受控演示，默认端口绑定到 `127.0.0.1`。不要直接暴露到公网，原因包括：

- 登录接口尚未签发会话令牌，其他业务接口没有完整的访问授权检查。
- `/uploads` 会直接提供上传视频；视频和对话可能包含敏感个人信息。
- 推理接口计算成本较高，公网部署前需要认证、限流、超时和并发控制。
- 本项目不是医疗器械、心理诊断工具或危机干预服务。

## 实测环境

以下是 2026-10-02 的实际运行环境，不代表最低要求：

- Windows + Docker Desktop WSL2 后端
- NVIDIA GeForce RTX 4070，12,282 MiB 显存
- NVIDIA 驱动 610.88
- Python 3.10.12
- PyTorch 2.4.0 + CUDA 12.4
- ONNX Runtime GPU 1.19.2
- OpenCV 4.9.0
- MediaPipe 0.10.35

最低 GPU、内存和磁盘要求尚未完成基准测试。在给出明确硬件下限前，需要分别测量启动峰值、单请求峰值和并发场景。

## 前置条件

- Docker Desktop（WSL2 后端）或兼容的 Linux Docker 环境
- NVIDIA GPU、兼容驱动和 NVIDIA Container Toolkit
- 宿主机已安装并运行 Ollama
- Ollama 已准备 `qwen2.5:7b`，或在 `.env` 中改为其他模型
- 首次启动能够访问 Whisper/BERT 的模型来源，或已经准备 Hugging Face 缓存

## 不进入 Git 的运行资源

GitHub 普通仓库不能承载当前这些大文件，且部分第三方资产存在再分发限制。构建前需要从已获授权的来源获得，并放到以下原路径：

| 路径 | 当前实测大小/校验 |
|---|---|
| `lnln_deploy.onnx` | 429,094,696 bytes；SHA-256 `695129e911ba2f116b8587415bd4c9249cdec76608e56d39c3cb82778dad4c58` |
| `MSA_FET/exts/` | 约 601 MiB；包含 OpenFace 和 MMSA-FET 预训练资产 |
| `libs/` | 约 78 MiB；包含 OpenFace 所需共享库 |

模型网盘链接、版本号、许可证和资产压缩包 SHA-256 尚待维护者补充。不要使用来源不明的镜像或模型文件。

放置完成后运行：

```bash
python3 scripts/check_assets.py
```

## 配置

复制环境变量示例：

```bash
cp .env.example .env
```

至少修改：

```env
POSTGRES_PASSWORD=使用一个新的本地密码
```

`.env` 是明文文件，已经被 `.gitignore` 和 `.dockerignore` 排除，不要上传。

## 启动

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f ai-inference
```

健康检查：

```text
http://127.0.0.1:8000/health
```

FastAPI 文档：

```text
http://127.0.0.1:8000/docs
```

## 会话摘要与删除接口

- 第一轮完整问答落库后，服务端只使用这一轮“用户消息 + 助手回复”向 Ollama 请求一次简短标题，并写入 `sessions.summary`。字段一旦非空，后续对话不会再次生成或覆盖。
- `GET /users/{user_id}/sessions` 会返回 nullable 的 `summary`，供客户端侧边栏显示。
- `DELETE /sessions/{session_id}` 会永久删除整段会话和消息，并尽力清理消息所引用的本地上传视频。
- `DELETE /users/{user_id}` 要求 JSON body 中提供当前密码；验证通过后永久删除账号、全部会话和消息，并尽力清理关联上传视频。

`sessions.summary` 会在服务启动时通过兼容性迁移自动补充。删除操作不可恢复；当前版本仍没有会话令牌和完整资源归属校验，只应在本机开发或受控 Demo 中使用。

停止服务并保留数据：

```bash
docker compose down
```

删除数据库、上传文件和 Hugging Face 缓存卷：

```bash
docker compose down -v
```

## 持久化数据

Compose 创建三个命名卷：

- `postgres_data`：数据库
- `uploads_data`：用户上传的视频
- `hf_cache`：Whisper/BERT 下载缓存

这些卷不会进入 Git。重要数据需要独立备份。

## 许可证与发布范围

除另有第三方声明外，本项目原创源代码 Copyright (C) 2026 markforcreative，并以 GNU GPL version 3 only（GPL-3.0-only）发布，详见 [LICENSE](LICENSE)。

`MSA_FET/` 是基于上游 `v_0.4.2` 的修改版 vendored 副本，沿用上游 GPL-3.0-or-later；修改内容见 [MSA_FET/MODIFICATIONS.md](MSA_FET/MODIFICATIONS.md)。

本源码仓库不包含 OpenFace 二进制与模型、项目 ONNX 权重、本地共享库、真实上传数据和本地配置。第三方依赖和运行资产仍受各自许可证约束。

## 当前限制与后续工作

- [x] 项目原创源码采用 GPL-3.0-only；第三方组件保留各自许可证
- [x] 确认 vendored MMSA-FET 基线为 `v_0.4.2`，并记录本地修改
- [x] 将 OpenFace、共享库、ONNX 模型、训练数据和真实上传文件排除在源码发布之外
- [ ] 在未来公开 Docker 镜像或资产包之前，单独确认所有运行资产的再分发权利
- [ ] 提供模型/资源的合法获取地址、备用地址和压缩包 SHA-256
- [x] 在 Windows、Docker Desktop、WSL2 和 RTX 4070 环境完成 Docker 构建与端到端推理测试（2026-10-10）
- [ ] 在第二台干净环境完成复现测试
- [ ] 为公开 Demo 增加令牌认证、资源所有权校验、限流和上传清理策略
- [ ] 测量最低显存、内存、磁盘与首次下载耗时
- [ ] 将固定 Git 版本、镜像 Tag 和镜像 Digest 写入发布说明
