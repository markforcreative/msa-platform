# 第三方组件与发布限制

本文件记录发布草稿中已识别的第三方组件。它不是法律意见；公开发布前应由维护者确认每个组件和模型的许可证。

## MMSA-FET

- 上游项目：https://github.com/thuiar/MMSA-FET
- 上游声明许可证：GPL-3.0
- 当前草稿保留了一份 `MSA_FET/` 源码副本；上游许可证原文已保存为 `LICENSES/MMSA-FET-GPL-3.0.txt`。
- 正式发布前仍必须核对本地副本对应的上游版本、是否被修改，并补齐版权声明和修改说明。项目整体许可证也必须与 GPL-3.0 义务兼容。

## OpenFace

- 上游项目：https://github.com/TadasBaltrusaitis/OpenFace
- 官方许可证：https://github.com/TadasBaltrusaitis/OpenFace/blob/master/OpenFace-license.txt
- OpenFace 的代码、二进制文件和训练模型具有独立且较严格的使用、复制和再分发条件。
- 本发布草稿不包含 `MSA_FET/exts/OpenFace/`；不要在没有确认授权的情况下把当前 OpenFace 二进制或模型打入公开镜像。

## Hugging Face 模型

默认运行时会获取：

- `openai/whisper-small`
- `bert-base-chinese`

这些模型各自有模型卡、许可证和使用条件。正式发布时应在 README 中链接到固定版本，并确认再分发方式。

## 项目 ONNX 模型

`lnln_deploy.onnx` 未进入 Git 草稿。维护者需要补充：

- 模型来源和训练数据说明
- 允许使用和再分发的许可证
- 固定版本、文件大小和 SHA-256

## 本地共享库

原项目的 `libs/` 含 OpenCV、GTK、ATK、TBB 等共享库。发布草稿将它们排除，正式分发前需要保留相应许可证和版权通知，或改为从系统包/官方构建过程获得。
