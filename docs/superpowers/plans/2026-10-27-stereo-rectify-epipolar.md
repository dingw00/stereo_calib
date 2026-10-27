# Stereo Rectify + Epipolar Implementation Plan

> **For agentic workers:** Implement task-by-task. Steps use checkbox syntax.

**Goal:** 两个脚本完成双目拼接图极线校正与棋盘格水平线可视化。

**Architecture:** `stereo_utils.py` 共享 IO/路径；`stereo_rectify.py` 用 R1/R2/P1/P2 remapping；`draw_epipolar.py` 左角点画水平线。

**Tech Stack:** Python 3, OpenCV, NumPy, 本地 venv

## Global Constraints

- 仅处理 img_dir 顶层图像；输出左右拼接
- 输出目录冲突时加 `_1`、`_2`…
- 校正不依赖棋盘格；极线默认 `chessboard/basic.yml`
- 左角点失败跳过；右失败警告仍画线

---

### Task 1: venv + requirements + stereo_utils

- [ ] 写 `requirements.txt`、`.gitignore`
- [ ] 创建 `.venv` 并安装依赖
- [ ] 实现 `stereo_utils.py`：列图、拆拼、唯一输出目录、读 param/chessboard、拷贝文件

### Task 2: stereo_rectify.py

- [ ] 实现 remapping 入口与 CLI
- [ ] 用 `images/calib2601` + `params/HY-MD-2601_param.yml` 跑通

### Task 3: draw_epipolar.py

- [ ] 实现角点检测与水平线绘制
- [ ] 对校正结果目录跑通
