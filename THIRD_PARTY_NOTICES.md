# 第三方组件与发布限制

本文件记录公开源码仓库中已识别的第三方组件和运行资产边界。它不是法律意见；分发模型、二进制文件或容器镜像前，维护者仍需确认相应许可证。

## MMSA-FET

- 上游项目：https://github.com/thuiar/MMSA-FET
- 上游基线：`v_0.4.2`（提交 `11c8f8c`，2022-10-25）
- 上游许可证：GPL-3.0-or-later
- 本仓库的 `MSA_FET/` 是基于该版本修改的 vendored 副本，不是上游原样版本或官方维护分支。
- 已确认的功能性修改位于 `MSA_FET/extractors/audio/librosa.py`，用于增加 `log_f0` 特征提取和帧数对齐。
- 详细修改记录见 [`MSA_FET/MODIFICATIONS.md`](MSA_FET/MODIFICATIONS.md)。
- 该目录继续遵守 GPL-3.0-or-later，并保留许可证原文及上游版权声明。

## openSMILE

- Python 依赖：`opensmile==2.6.0`
- 官方许可证：https://github.com/audeering/opensmile-python/blob/main/LICENSE
- 该依赖作为外部软件包安装，未以源码或二进制形式收录在本仓库。
- 其许可证允许非商业研究、教学、公开演示和个人实验；商业用途通常需要另行取得许可。
- 本仓库的 GPL-3.0-only 不会覆盖或替代 openSMILE 的独立许可证。

## OpenFace

- 上游项目：https://github.com/TadasBaltrusaitis/OpenFace
- 官方许可证：https://github.com/TadasBaltrusaitis/OpenFace/blob/master/OpenFace-license.txt
- OpenFace 的代码、二进制文件和训练模型具有独立且较严格的使用、复制和再分发条件。
- 本源码仓库不包含 `MSA_FET/exts/OpenFace/`；不要在没有确认授权的情况下把当前 OpenFace 二进制或模型打入公开镜像。

## Hugging Face 模型

默认运行时会获取：

- `openai/whisper-small`：https://huggingface.co/openai/whisper-small
- `bert-base-chinese`：https://huggingface.co/google-bert/bert-base-chinese

这些模型未进入 Git。使用者应查看对应模型卡、固定版本和许可证，并自行确认使用与再分发条件。

## 项目 ONNX 模型

`lnln_deploy.onnx` 未进入 Git。本源码仓库不提供该模型。维护者仍需补充：

- 模型来源和训练数据说明
- 允许使用和再分发的许可证
- 固定版本、文件大小和 SHA-256

## 本地共享库

原项目的 `libs/` 包含 OpenCV、GTK、ATK、TBB 等共享库。本源码仓库将其排除。未来如需分发这些共享库或完整容器镜像，应保留相应许可证与版权通知，或改为从系统包或官方构建流程获取。
