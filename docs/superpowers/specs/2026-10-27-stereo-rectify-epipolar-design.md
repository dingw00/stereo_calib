# 双目极线校正与可视化 — 设计规格

日期：2026-10-27

## 目标

提供两个可直接运行的 Python 脚本：

1. `stereo_rectify.py`：根据双目内参 yml 对左右拼接图做极线校正（remapping）
2. `draw_epipolar.py`：在已校正拼接图上检测棋盘格角点，按左目角点 y 画彩色水平线，便于目视检查极线对齐

## 架构

采用「两个入口脚本 + 一个共享模块」：

```
stereo_calib/
├── stereo_rectify.py
├── draw_epipolar.py
├── stereo_utils.py
├── requirements.txt
├── chessboard/basic.yml
├── params/*.yml
└── images/...
```

校正算法直接使用参数文件中已有的 `R1/R2/P1/P2`，调用 `cv2.initUndistortRectifyMap` + `cv2.remap`，不重新运行 `stereoRectify`。

## CLI

```bash
python stereo_rectify.py --param params/HY-MD-2601_param.yml --img_dir images/calib2601

python draw_epipolar.py --img_dir images/calib2601_rectified
python draw_epipolar.py --img_dir images/calib2601_rectified --chessboard chessboard/basic.yml
```

## 图像约定

- 只处理 `--img_dir` **顶层**图像文件，忽略子目录（如 `left/`、`right/`）
- 输入与输出均为左右水平拼接图；单目尺寸来自参数 `imgsz=[w,h]`，拼接图期望为 `(h, 2w)`
- 支持常见后缀：`.png`、`.jpg`、`.jpeg`、`.bmp`、`.tif`、`.tiff`（大小写不敏感）

## 输出目录

- 校正：`{img_dir}_rectified`，与 `img_dir` 同级
- 极线图：`{img_dir}_epipolar`，与 `img_dir` 同级
- 若目标目录已存在，则依次尝试 `{name}_1`、`{name}_2`、…
- `stereo_rectify` 将 `--param` 指向的 yml **复制**到输出目录

## stereo_rectify 流程

1. 用 OpenCV `FileStorage` 读取：`leftK`、`leftD`、`rightK`、`rightD`、`R1`、`R2`、`P1`、`P2`、`imgsz`
2. 按单目尺寸各构建一次左右 remap 表并全程复用
3. 对每张顶层图像：左右对半切开 → `remap` → 水平拼接 → 以原文件名写入输出目录
4. 拷贝 param yml；打印处理数量

不依赖棋盘格参数。

## draw_epipolar 流程

1. 读取棋盘格配置（默认 `chessboard/basic.yml`）：`rows`、`cols` 为内角点数；`size` 可选不用于本工具
2. 对每张拼接图：切开 → 左右分别 `findChessboardCorners`（成功则亚像素精化）
3. 左目成功：对每个左角点的 y，在整幅拼接图上画彩色水平线；右目成功则额外绘制角点标记
4. 左目失败：打印警告并跳过该图
5. 右目失败：打印警告，仍按左目 y 画线
6. 结果以同名文件写入输出目录

## 错误处理

- 路径不存在或参数缺关键字段 → 报错退出（非零）
- 图像宽非偶数或与 `2 * imgsz[0]` 不一致 → 警告并跳过该图
- 无顶层图像 → 警告后正常退出

## 依赖

- Python 3 + OpenCV + NumPy
- `requirements.txt` 列出 `opencv-python`、`numpy`
- 使用项目本地 `venv` 运行；不做成可 pip 安装包

## 非目标

- 不处理子目录中的分目图像
- 不重新计算 `stereoRectify`
- 不输出角点 y 差等自动质量报告
- 不改变输入图像布局（保持左右拼接）
