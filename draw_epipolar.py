#!/usr/bin/env python3
"""Draw horizontal lines through left chessboard corners on rectified stereo pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from stereo_utils import (
    detect_chessboard_corners,
    list_top_images,
    load_chessboard,
    rainbow_color,
    select_sparse_ys,
    split_stereo,
    unique_output_dir,
)

# Skip another line if its y is closer than this (pixels) to the previous kept line.
DEFAULT_MIN_DY = 10


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="在校正后的拼接图上绘制极线水平线")
    p.add_argument("--img_dir", required=True, type=Path, help="已校正拼接图目录（仅顶层）")
    p.add_argument(
        "--chessboard",
        type=Path,
        default=Path("chessboard/basic.yml"),
        help="棋盘格配置 yml（默认 chessboard/basic.yml）",
    )
    p.add_argument(
        "--min_dy",
        type=float,
        default=DEFAULT_MIN_DY,
        help=f"相邻横线最小竖直间距（像素），默认 {DEFAULT_MIN_DY}",
    )
    return p.parse_args(argv)


def draw_epipolar_lines(
    img: np.ndarray,
    left_corners: np.ndarray,
    min_dy: float,
) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    ys = left_corners.reshape(-1, 2)[:, 1].tolist()
    line_ys = select_sparse_ys(ys, min_dy)
    for i, yi in enumerate(line_ys):
        if yi < 0 or yi >= h:
            continue
        color = rainbow_color(i)
        cv2.line(out, (0, yi), (w - 1, yi), color, 1, cv2.LINE_AA)
    return out


def compute_delta_y(
    left_corners: np.ndarray,
    right_corners: np.ndarray,
) -> np.ndarray:
    """delta_y = y_right - y_left (right lower => positive). Matched by corner index."""
    ly = left_corners.reshape(-1, 2)[:, 1].astype(np.float64)
    ry = right_corners.reshape(-1, 2)[:, 1].astype(np.float64)
    if len(ly) != len(ry):
        raise ValueError(f"左右角点数量不一致: {len(ly)} vs {len(ry)}")
    return ry - ly


def mean_var(values: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Population mean/variance (ddof=0). Empty -> (None, None)."""
    if values.size == 0:
        return None, None
    return float(np.mean(values)), float(np.var(values, ddof=0))


def mean_abs_max_abs(
    values: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    """mean(|dy|), max(|dy|). Empty -> (None, None)."""
    if values.size == 0:
        return None, None
    abs_v = np.abs(values)
    return float(np.mean(abs_v)), float(np.max(abs_v))


def summarize_delta(values: np.ndarray) -> Dict[str, Any]:
    m, v = mean_var(values)
    mean_abs, max_abs = mean_abs_max_abs(values)
    return {
        "n": int(values.size),
        "mean": m,
        "var": v,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
    }


def build_delta_y_stats(
    per_image: List[Dict[str, Any]],
    all_delta: np.ndarray,
) -> Dict[str, Any]:
    g = summarize_delta(all_delta)
    return {
        "definition": "delta_y = y_right - y_left (right lower => positive)",
        "variance": "population (ddof=0)",
        "unit": "pixel",
        "images": per_image,
        "global": {
            "n_images": len(per_image),
            "n_points": g["n"],
            "mean": g["mean"],
            "var": g["var"],
            "mean_abs": g["mean_abs"],
            "max_abs": g["max_abs"],
        },
    }


def _scatter_pair(
    ax,
    images: List[Dict[str, Any]],
    glob: Dict[str, Any],
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    vline0: bool = False,
) -> None:
    if images:
        xs = [im[x_key] for im in images]
        ys = [im[y_key] for im in images]
        ax.scatter(
            xs,
            ys,
            s=48,
            c="#1f77b4",
            alpha=0.85,
            edgecolors="white",
            linewidths=0.6,
            label=f"per image (n={len(images)})",
            zorder=2,
        )
    gx, gy = glob.get(x_key), glob.get(y_key)
    if gx is not None and gy is not None:
        ax.scatter(
            [gx],
            [gy],
            s=160,
            c="#d62728",
            marker="*",
            edgecolors="black",
            linewidths=0.5,
            label="global",
            zorder=3,
        )
    if vline0:
        ax.axvline(0.0, color="#888888", linestyle="--", linewidth=1, zorder=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def plot_delta_y_scatter(stats: Dict[str, Any], out_path: Path) -> None:
    """One figure, two scatters: (mean, var) and (mean_abs, max_abs)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images = stats.get("images") or []
    glob = stats.get("global") or {}
    if not images and glob.get("mean") is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _scatter_pair(
        axes[0],
        images,
        glob,
        "mean",
        "var",
        "delta_y mean (px)",
        "delta_y var (px²)",
        "mean vs var",
        vline0=True,
    )
    _scatter_pair(
        axes[1],
        images,
        glob,
        "mean_abs",
        "max_abs",
        "mean(|dy|) (px)",
        "max(|dy|) (px)",
        "mean(|dy|) vs max(|dy|)",
    )
    fig.suptitle("delta_y = y_right - y_left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    args = parse_args(argv)
    img_dir = args.img_dir.resolve()
    chessboard_path = args.chessboard
    if not chessboard_path.is_absolute():
        cand = Path.cwd() / chessboard_path
        if not cand.is_file():
            cand = Path(__file__).resolve().parent / chessboard_path
        chessboard_path = cand
    else:
        chessboard_path = chessboard_path.resolve()

    try:
        rows, cols = load_chessboard(chessboard_path)
    except (FileNotFoundError, KeyError, RuntimeError) as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1

    pattern_size = (cols, rows)  # OpenCV: (width, height) = (cols, rows)

    try:
        images = list_top_images(img_dir)
    except FileNotFoundError as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1

    if not images:
        print(f"[警告] 目录中无顶层图像: {img_dir}")
        return 0

    out_dir = unique_output_dir(Path(f"{img_dir}_epipolar"))
    out_dir.mkdir(parents=True, exist_ok=False)

    ok = 0
    skipped = 0
    per_image_stats: List[Dict[str, Any]] = []
    all_delta_list: List[np.ndarray] = []

    for path in images:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[警告] 无法读取，跳过: {path.name}")
            skipped += 1
            continue
        if img.shape[1] % 2 != 0:
            print(f"[警告] 宽度非偶数，跳过: {path.name}")
            skipped += 1
            continue

        left, right = split_stereo(img)
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        left_ok, left_corners = detect_chessboard_corners(left_gray, pattern_size)
        if not left_ok:
            print(f"[警告] 左目角点检测失败，跳过: {path.name}")
            skipped += 1
            continue

        right_ok, right_corners = detect_chessboard_corners(right_gray, pattern_size)
        if not right_ok:
            print(f"[警告] 右目角点检测失败，仍按左目画线（无 delta_y 统计）: {path.name}")
            right_corners = None
        else:
            try:
                delta = compute_delta_y(left_corners, right_corners)
            except ValueError as e:
                print(f"[警告] {e}，跳过该图统计: {path.name}")
                delta = None
            else:
                s = summarize_delta(delta)
                per_image_stats.append({"file": path.name, **s})
                all_delta_list.append(delta)

        vis = draw_epipolar_lines(img, left_corners, args.min_dy)
        out_path = out_dir / path.name
        if not cv2.imwrite(str(out_path), vis):
            print(f"[警告] 写入失败，跳过: {out_path}")
            skipped += 1
            continue
        ok += 1

    all_delta = (
        np.concatenate(all_delta_list) if all_delta_list else np.array([], dtype=np.float64)
    )
    stats = build_delta_y_stats(per_image_stats, all_delta)
    stats_path = out_dir / "delta_y_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    plot_path = out_dir / "delta_y_scatter.png"
    if per_image_stats:
        plot_delta_y_scatter(stats, plot_path)
    else:
        plot_path = None

    print(f"输出目录: {out_dir}")
    print(f"棋盘格内角点: cols={cols}, rows={rows}")
    print(f"成功 {ok} 张，跳过 {skipped} 张，共 {len(images)} 张")
    print(f"delta_y 统计: {stats_path}")
    if plot_path is not None:
        print(f"delta_y 散点图: {plot_path}")
    g = stats["global"]
    print(
        f"全局 n_points={g['n_points']}, mean={g['mean']}, var={g['var']}, "
        f"mean(|dy|)={g['mean_abs']}, max(|dy|)={g['max_abs']} "
        f"（参与统计 {g['n_images']} 张）"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
