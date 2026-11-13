"""Shared helpers for stereo rectify / epipolar tools."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def unique_output_dir(base: Path) -> Path:
    """Return base if missing, else base_1, base_2, ..."""
    if not base.exists():
        return base
    i = 1
    while True:
        candidate = Path(f"{base}_{i}")
        if not candidate.exists():
            return candidate
        i += 1


def list_top_images(img_dir: Path) -> List[Path]:
    """List image files in img_dir top level only (no subdirs)."""
    if not img_dir.is_dir():
        raise FileNotFoundError(f"图像目录不存在: {img_dir}")
    files = [
        p
        for p in sorted(img_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return files


def split_stereo(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    if w % 2 != 0:
        raise ValueError(f"拼接图宽度必须为偶数，当前 w={w}")
    mid = w // 2
    return img[:, :mid].copy(), img[:, mid:].copy()


def merge_stereo(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape:
        raise ValueError(f"左右尺寸不一致: {left.shape} vs {right.shape}")
    return np.hstack([left, right])


def load_stereo_params(param_path: Path) -> dict:
    if not param_path.is_file():
        raise FileNotFoundError(f"参数文件不存在: {param_path}")
    fs = cv2.FileStorage(str(param_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"无法打开参数文件: {param_path}")

    def _mat(name: str) -> np.ndarray:
        node = fs.getNode(name)
        if node.empty():
            fs.release()
            raise KeyError(f"参数文件缺少字段: {name}")
        mat = node.mat()
        if mat is None:
            fs.release()
            raise KeyError(f"参数字段不是矩阵: {name}")
        return mat

    required = ("leftK", "leftD", "rightK", "rightD", "R1", "R2", "P1", "P2")
    params = {name: _mat(name) for name in required}

    imgsz_node = fs.getNode("imgsz")
    if imgsz_node.empty():
        fs.release()
        raise KeyError("参数文件缺少字段: imgsz")
    # OpenCV may store as sequence or mat
    if imgsz_node.isSeq():
        w = int(imgsz_node.at(0).real())
        h = int(imgsz_node.at(1).real())
        imgsz = (w, h)
    else:
        arr = imgsz_node.mat()
        if arr is None:
            fs.release()
            raise KeyError("无法解析 imgsz")
        flat = np.asarray(arr).reshape(-1)
        imgsz = (int(flat[0]), int(flat[1]))

    fs.release()
    params["imgsz"] = imgsz  # (width, height)
    return params


def load_chessboard(chessboard_path: Path) -> Tuple[int, int]:
    rows, cols, _ = load_chessboard_config(chessboard_path)
    return rows, cols


def load_chessboard_config(chessboard_path: Path) -> Tuple[int, int, float]:
    """Return (rows, cols, square_size). size defaults to 1.0 if missing."""
    if not chessboard_path.is_file():
        raise FileNotFoundError(f"棋盘格配置不存在: {chessboard_path}")
    fs = cv2.FileStorage(str(chessboard_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"无法打开棋盘格配置: {chessboard_path}")
    rows_node = fs.getNode("rows")
    cols_node = fs.getNode("cols")
    if rows_node.empty() or cols_node.empty():
        fs.release()
        raise KeyError("棋盘格配置缺少 rows/cols")
    rows = int(rows_node.real())
    cols = int(cols_node.real())
    size_node = fs.getNode("size")
    square_size = float(size_node.real()) if not size_node.empty() else 1.0
    fs.release()
    return rows, cols, square_size


def resolve_chessboard_path(chessboard: Path, script_file: Path) -> Path:
    if chessboard.is_absolute():
        return chessboard.resolve()
    cand = Path.cwd() / chessboard
    if cand.is_file():
        return cand.resolve()
    return (script_file.resolve().parent / chessboard).resolve()


def save_stereo_params(out_path: Path, params: dict) -> None:
    """Write stereo params in the same schema as params/*_param.yml."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(out_path), cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError(f"无法写入参数文件: {out_path}")
    for key in (
        "leftK",
        "leftD",
        "rightK",
        "rightD",
        "R",
        "t",
        "E",
        "F",
        "R1",
        "R2",
        "P1",
        "P2",
        "Q",
    ):
        fs.write(key, np.asarray(params[key], dtype=np.float64))
    fs.write("fx", float(params["fx"]))
    fs.write("fy", float(params["fy"]))
    fs.write("cx", float(params["cx"]))
    fs.write("cy", float(params["cy"]))
    fs.write("baseline", float(params["baseline"]))
    w, h = params["imgsz"]
    # Match existing style: imgsz: [ w, h ]
    fs.startWriteStruct("imgsz", cv2.FileNode_SEQ)
    fs.write("", int(w))
    fs.write("", int(h))
    fs.endWriteStruct()
    fs.release()


def copy_param_to_dir(param_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / param_path.name
    shutil.copy2(param_path, dest)
    return dest


def build_rectify_maps(params: dict):
    w, h = params["imgsz"]
    map1x, map1y = cv2.initUndistortRectifyMap(
        params["leftK"],
        params["leftD"],
        params["R1"],
        params["P1"],
        (w, h),
        cv2.CV_32FC1,
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        params["rightK"],
        params["rightD"],
        params["R2"],
        params["P2"],
        (w, h),
        cv2.CV_32FC1,
    )
    return (map1x, map1y), (map2x, map2y)


def rectify_pair(
    left: np.ndarray,
    right: np.ndarray,
    maps_left,
    maps_right,
) -> Tuple[np.ndarray, np.ndarray]:
    map1x, map1y = maps_left
    map2x, map2y = maps_right
    left_r = cv2.remap(left, map1x, map1y, interpolation=cv2.INTER_LINEAR)
    right_r = cv2.remap(right, map2x, map2y, interpolation=cv2.INTER_LINEAR)
    return left_r, right_r


def detect_chessboard_corners(
    gray: np.ndarray,
    pattern_size: Tuple[int, int],
) -> Tuple[bool, Optional[np.ndarray]]:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


# High-contrast rainbow for vertically adjacent epipolar lines (BGR).
RAINBOW_BGR: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 255),      # red
    (0, 165, 255),    # orange
    (0, 255, 255),    # yellow
    (0, 255, 0),      # green
    (255, 255, 0),    # cyan
    (255, 0, 0),      # blue
    (255, 0, 255),    # magenta
)


def rainbow_color(i: int) -> Tuple[int, int, int]:
    return RAINBOW_BGR[i % len(RAINBOW_BGR)]


def select_sparse_ys(ys: List[float], min_gap: float) -> List[int]:
    """Keep y positions sorted top→bottom, skipping ones closer than min_gap to last kept."""
    if not ys:
        return []
    sorted_ys = sorted(float(y) for y in ys)
    kept = [sorted_ys[0]]
    for y in sorted_ys[1:]:
        if y - kept[-1] >= min_gap:
            kept.append(y)
    return [int(round(y)) for y in kept]
