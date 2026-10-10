# MSA Platform

基于 FastAPI、PostgreSQL、ONNX Runtime、Whisper、BERT、MMSA-FET/OpenFace 和 Ollama 的多模态情感分析与对话后端。

> 当前目录是发布草稿，还不能直接公开发布。第三方许可证、模型下载地址和项目自身许可证仍需确认；请先阅读 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

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

最低 GPU、内存和磁盘要求尚未完成基准测试。正式发布时应分别测试启动峰值、单请求峰值和并发场景。

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

## 发布前待办

- [ ] 选择与第三方依赖兼容的项目许可证
- [ ] 确认本地 MMSA-FET 副本的上游版本，并补齐版权声明和修改记录（GPL-3.0 原文已收录）
- [ ] 确认 OpenFace、共享库、ONNX 模型和训练数据的再分发权利
- [ ] 提供模型/资源下载地址、备用地址和压缩包 SHA-256
- [ ] 完成一次从空环境开始的 Docker 构建与端到端推理测试
- [ ] 为公开 Demo 增加令牌认证、资源所有权校验、限流和上传清理策略
- [ ] 测量最低显存、内存、磁盘与首次下载耗时
- [ ] 将固定 Git 版本、镜像 Tag 和镜像 Digest 写入发布说明
