#!/usr/bin/env python3
"""Stereo camera calibration from side-by-side chessboard images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from stereo_utils import (
    detect_chessboard_corners,
    list_top_images,
    load_chessboard_config,
    resolve_chessboard_path,
    save_stereo_params,
    split_stereo,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="双目拼接图棋盘格标定")
    p.add_argument("--img_dir", required=True, type=Path, help="输入拼接图目录（仅顶层）")
    p.add_argument(
        "--chessboard",
        type=Path,
        default=Path("chessboard/basic.yml"),
        help="棋盘格配置 yml（默认 chessboard/basic.yml）",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="输出参数 yml 路径（自动创建目录）",
    )
    return p.parse_args(argv)


def make_object_points(rows: int, cols: int, square_size: float) -> np.ndarray:
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)
    return objp


def collect_corners(
    images: List[Path],
    pattern_size: Tuple[int, int],
    objp: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], Tuple[int, int], List[str]]:
    objpoints: List[np.ndarray] = []
    imgpoints_l: List[np.ndarray] = []
    imgpoints_r: List[np.ndarray] = []
    used: List[str] = []
    mono_size: Tuple[int, int] | None = None

    for path in images:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[警告] 无法读取，跳过: {path.name}")
            continue
        if img.shape[1] % 2 != 0:
            print(f"[警告] 宽度非偶数，跳过: {path.name}")
            continue
        try:
            left, right = split_stereo(img)
        except ValueError as e:
            print(f"[警告] {e}，跳过: {path.name}")
            continue

        h, w = left.shape[:2]
        if mono_size is None:
            mono_size = (w, h)
        elif mono_size != (w, h):
            print(f"[警告] 尺寸与首张不一致 {mono_size} vs {(w, h)}，跳过: {path.name}")
            continue

        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        lok, lc = detect_chessboard_corners(left_gray, pattern_size)
        rok, rc = detect_chessboard_corners(right_gray, pattern_size)
        if not lok or not rok:
            which = []
            if not lok:
                which.append("左")
            if not rok:
                which.append("右")
            print(f"[警告] {'/'.join(which)}目角点失败，跳过: {path.name}")
            continue

        objpoints.append(objp.copy())
        imgpoints_l.append(lc)
        imgpoints_r.append(rc)
        used.append(path.name)

    if mono_size is None:
        raise RuntimeError("没有可用图像")
    return objpoints, imgpoints_l, imgpoints_r, mono_size, used


def calibrate_stereo(
    objpoints: List[np.ndarray],
    imgpoints_l: List[np.ndarray],
    imgpoints_r: List[np.ndarray],
    image_size: Tuple[int, int],
) -> dict:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    rms_l, leftK, leftD, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_l, image_size, None, None, criteria=criteria
    )
    rms_r, rightK, rightD, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_r, image_size, None, None, criteria=criteria
    )
    print(f"单目标定 RMS: left={rms_l:.4f}, right={rms_r:.4f}")

    flags = cv2.CALIB_FIX_INTRINSIC
    rms_s, leftK, leftD, rightK, rightD, R, t, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_l,
        imgpoints_r,
        leftK,
        leftD,
        rightK,
        rightD,
        image_size,
        criteria=criteria,
        flags=flags,
    )
    print(f"双目标定 RMS: {rms_s:.4f}")

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        leftK,
        leftD,
        rightK,
        rightD,
        image_size,
        R,
        t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    fx = float(P1[0, 0])
    fy = float(P1[1, 1])
    cx = float(P1[0, 2])
    cy = float(P1[1, 2])
    # baseline in same unit as t / square_size; prefer |Tx| from P2
    if abs(fx) > 1e-12:
        baseline = abs(float(P2[0, 3]) / fx)
    else:
        baseline = float(np.linalg.norm(t))

    # Normalize distortion to 1xN like params/*.yml
    leftD = np.asarray(leftD, dtype=np.float64).reshape(1, -1)
    rightD = np.asarray(rightD, dtype=np.float64).reshape(1, -1)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)

    return {
        "leftK": np.asarray(leftK, dtype=np.float64),
        "leftD": leftD,
        "rightK": np.asarray(rightK, dtype=np.float64),
        "rightD": rightD,
        "R": np.asarray(R, dtype=np.float64),
        "t": t,
        "E": np.asarray(E, dtype=np.float64),
        "F": np.asarray(F, dtype=np.float64),
        "R1": np.asarray(R1, dtype=np.float64),
        "R2": np.asarray(R2, dtype=np.float64),
        "P1": np.asarray(P1, dtype=np.float64),
        "P2": np.asarray(P2, dtype=np.float64),
        "Q": np.asarray(Q, dtype=np.float64),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "baseline": baseline,
        "imgsz": image_size,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    img_dir = args.img_dir.resolve()
    out_path = args.output.resolve()
    chessboard_path = resolve_chessboard_path(args.chessboard, Path(__file__))

    try:
        rows, cols, square_size = load_chessboard_config(chessboard_path)
    except (FileNotFoundError, KeyError, RuntimeError) as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1

    pattern_size = (cols, rows)
    objp = make_object_points(rows, cols, square_size)

    try:
        images = list_top_images(img_dir)
    except FileNotFoundError as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1

    if not images:
        print(f"[错误] 目录中无顶层图像: {img_dir}", file=sys.stderr)
        return 1

    try:
        objpoints, imgpoints_l, imgpoints_r, image_size, used = collect_corners(
            images, pattern_size, objp
        )
    except RuntimeError as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1

    if len(objpoints) < 3:
        print(
            f"[错误] 有效双目角点对过少: {len(objpoints)}（至少需要 3 对）",
            file=sys.stderr,
        )
        return 1

    print(
        f"棋盘格: cols={cols}, rows={rows}, size={square_size}; "
        f"有效 {len(used)}/{len(images)} 张; 单目尺寸={image_size}"
    )

    params = calibrate_stereo(objpoints, imgpoints_l, imgpoints_r, image_size)
    try:
        save_stereo_params(out_path, params)
    except RuntimeError as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1

    print(f"已写入: {out_path}")
    print(
        f"fx={params['fx']:.4f}, fy={params['fy']:.4f}, "
        f"cx={params['cx']:.4f}, cy={params['cy']:.4f}, "
        f"baseline={params['baseline']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
