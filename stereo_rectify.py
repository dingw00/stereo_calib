#!/usr/bin/env python3
"""Rectify side-by-side stereo images using precomputed R1/R2/P1/P2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from stereo_utils import (
    build_rectify_maps,
    copy_param_to_dir,
    list_top_images,
    load_stereo_params,
    merge_stereo,
    rectify_pair,
    split_stereo,
    unique_output_dir,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="双目拼接图极线校正")
    p.add_argument("--param", required=True, type=Path, help="双目参数 yml 路径")
    p.add_argument("--img_dir", required=True, type=Path, help="输入拼接图目录（仅顶层）")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    param_path = args.param.resolve()
    img_dir = args.img_dir.resolve()

    try:
        params = load_stereo_params(param_path)
    except (FileNotFoundError, KeyError, RuntimeError) as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1

    try:
        images = list_top_images(img_dir)
    except FileNotFoundError as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1

    if not images:
        print(f"[警告] 目录中无顶层图像: {img_dir}")
        return 0

    out_dir = unique_output_dir(Path(f"{img_dir}_rectified"))
    out_dir.mkdir(parents=True, exist_ok=False)

    w, h = params["imgsz"]
    expected_w = 2 * w
    maps_left, maps_right = build_rectify_maps(params)

    ok = 0
    skipped = 0
    for path in images:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[警告] 无法读取，跳过: {path.name}")
            skipped += 1
            continue
        ih, iw = img.shape[:2]
        if iw % 2 != 0 or iw != expected_w or ih != h:
            print(
                f"[警告] 尺寸不符期望 ({h},{expected_w})，实际 ({ih},{iw})，跳过: {path.name}"
            )
            skipped += 1
            continue
        try:
            left, right = split_stereo(img)
            left_r, right_r = rectify_pair(left, right, maps_left, maps_right)
            out = merge_stereo(left_r, right_r)
        except ValueError as e:
            print(f"[警告] {e}，跳过: {path.name}")
            skipped += 1
            continue
        out_path = out_dir / path.name
        if not cv2.imwrite(str(out_path), out):
            print(f"[警告] 写入失败，跳过: {out_path}")
            skipped += 1
            continue
        ok += 1

    dest = copy_param_to_dir(param_path, out_dir)
    print(f"输出目录: {out_dir}")
    print(f"已拷贝参数: {dest.name}")
    print(f"成功 {ok} 张，跳过 {skipped} 张，共 {len(images)} 张")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
