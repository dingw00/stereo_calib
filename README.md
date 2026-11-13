# 双目标定与极线检查

## 1. 激活虚拟环境

每次打开新终端，先进入项目目录并激活 `.venv`：

```bash
cd /home/dwd/HYKJ/stereo_calib
source .venv/bin/activate
```

终端提示符通常会出现 `(.venv)`。以下命令都在项目根目录执行。
首次安装或缺少依赖时运行：

```bash
python -m pip install -r requirements.txt
```

## 2. 使用已有参数校正图像

已有参数时用 `stereo_rectify.py`，不需要重新标定：

```bash
python stereo_rectify.py \
  --param params/HY-2504_20261103_param.yml \
  --img_dir images/20261110_171818
```

输出到 `images/20261110_171818_rectified/`，同时复制所用参数文件。
输入应为左目在左、右目在右的水平拼接图，尺寸必须与参数文件一致。

## 3. 绘制极线并检查误差

对校正后的目录运行：

```bash
python draw_epipolar.py --img_dir images/20261110_171818_rectified
```

输出到 `images/20261110_171818_rectified_epipolar/`，包含极线图、
`delta_y_stats.json`（误差统计）和 `delta_y_scatter.png`（散点图，有有效统计时生成）。

- `mean_abs`：平均绝对纵向偏差，单位为像素，越接近 0 越好。
- `max_abs`：最大绝对纵向偏差，单位为像素。
- `mean`：平均有符号偏差；正值表示右目角点比左目低。
- 左目角点检测失败会跳过图像；右目检测失败仍绘线，但不参与误差统计。

## 4. 需要重新标定时

`calib_stereo.py` 会重新计算左右内参和双目外参，不读取已有内参：

```bash
python calib_stereo.py \
  --img_dir images/20261110_171818 \
  --chessboard chessboard/basic.yml \
  --output params/HY-2504_20261110_recalib.yml
```

至少需要 3 对左右均成功检测棋盘格角点的图像。完成后，将第 2 步的
`--param` 改为新生成的参数路径，再校正并检查极线。
`--output` 请使用新文件名，避免覆盖已有参数。

## 注意事项

- 所有脚本只读取输入目录顶层图像，不递归读取 `raw/` 等子目录。
- 标定和极线检查默认使用 `chessboard/basic.yml`：`cols=11`、`rows=8`，均为内角点数；`size=20.0` 为方格边长，基线单位与它一致（若为毫米，基线也为毫米）。
- 使用其他棋盘格时，为标定和极线检查命令添加 `--chessboard 配置路径`。
- 校正、极线输出目录若已存在，会自动添加后缀创建新目录；后续命令请使用终端实际打印的路径。
- 查看完整参数：`python 脚本名.py --help`。
- 退出虚拟环境：`deactivate`。