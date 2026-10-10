# MMSA-FET 本地修改记录

## 上游基线

- 上游仓库：https://github.com/thuiar/MMSA-FET
- 上游标签：[`v_0.4.2`](https://github.com/thuiar/MMSA-FET/tree/v_0.4.2)
- 上游提交：`11c8f8c`
- 上游日期：2022-10-25
- 本地比对日期：2026-10-10

本目录是为 MSA Platform 保留并修改的 vendored 副本，不是 MMSA-FET 官方维护版本。

## 功能性修改

### `MSA_FET/extractors/audio/librosa.py`

修改目的：为本项目的多模态模型补充 `log_f0` 音频特征。

主要变化：

- 不再把 `log_f0` 交给通用的 `librosa.feature` 调用；
- 使用 `librosa.pyin` 提取基频 F0；
- 处理无声帧、零值和 `NaN`，然后计算自然对数；
- 将 `log_f0` 与其他音频特征按最短帧数对齐并拼接。

原始修改日期未保留。可以确认该修改在 2026-10-10 之前已经存在，并于 2026-10-10 补充本修改记录。

## 其他文件

除上述文件外，本次与上游 `v_0.4.2` 的字节级比对未发现其他已跟踪文件差异。

## 许可证

本目录继续遵守 GNU General Public License version 3 or later（GPL-3.0-or-later）。

许可证正文见：

- [`../LICENSE`](../LICENSE)
- [`../LICENSES/MMSA-FET-GPL-3.0.txt`](../LICENSES/MMSA-FET-GPL-3.0.txt)

上游项目及其贡献者保留原有版权。