from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from PIL import Image, ImageOps
import piexif

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def is_valid_image(file_path: str) -> bool:
    """
    验证文件是否为支持的图片格式。

    Args:
        file_path (str): 图片文件路径

    Returns:
        bool: 是否为有效图片
    """
    if not file_path or not os.path.isfile(file_path):
        return False

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return False

    try:
        with Image.open(file_path) as image:
            image.verify()
        return True
    except Exception:
        return False


def _parse_exif_datetime(value: Optional[bytes]) -> Optional[str]:
    if not value:
        return None
    try:
        text = value.decode("utf-8")
        dt = datetime.strptime(text, "%Y:%m:%d %H:%M:%S")
        return dt.isoformat()
    except Exception:
        return None


def _rational_to_float(value: Tuple[int, int]) -> Optional[float]:
    if not value or len(value) != 2 or value[1] == 0:
        return None
    return float(value[0]) / float(value[1])


def _convert_gps_coordinate(values: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]) -> Optional[float]:
    if not values or len(values) != 3:
        return None
    degrees = _rational_to_float(values[0])
    minutes = _rational_to_float(values[1])
    seconds = _rational_to_float(values[2])
    if degrees is None or minutes is None or seconds is None:
        return None
    return degrees + (minutes / 60.0) + (seconds / 3600.0)


def extract_exif_metadata(file_path: str) -> Dict[str, Any]:
    """
    解析EXIF元数据（拍摄时间、GPS、相机型号）。

    Args:
        file_path (str): 图片文件路径

    Returns:
        Dict[str, Any]: EXIF信息
    """
    metadata: Dict[str, Any] = {
        "datetime": None,
        "camera": None,
        "gps": None,
        "orientation": None,
    }

    if not is_valid_image(file_path):
        return metadata

    try:
        with Image.open(file_path) as image:
            exif_bytes = image.info.get("exif")
        if not exif_bytes:
            return metadata
        exif_data = piexif.load(exif_bytes)
    except Exception:
        return metadata

    datetime_value = (
        exif_data.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal)
        or exif_data.get("0th", {}).get(piexif.ImageIFD.DateTime)
    )
    metadata["datetime"] = _parse_exif_datetime(datetime_value)

    make = exif_data.get("0th", {}).get(piexif.ImageIFD.Make)
    model = exif_data.get("0th", {}).get(piexif.ImageIFD.Model)
    make_text = make.decode("utf-8", errors="ignore") if isinstance(make, (bytes, bytearray)) else None
    model_text = model.decode("utf-8", errors="ignore") if isinstance(model, (bytes, bytearray)) else None
    if make_text and model_text:
        metadata["camera"] = f"{make_text} {model_text}".strip()
    else:
        metadata["camera"] = make_text or model_text

    orientation = exif_data.get("0th", {}).get(piexif.ImageIFD.Orientation)
    metadata["orientation"] = int(orientation) if orientation is not None else None

    gps_info = exif_data.get("GPS", {})
    if gps_info:
        lat = _convert_gps_coordinate(gps_info.get(piexif.GPSIFD.GPSLatitude))
        lat_ref = gps_info.get(piexif.GPSIFD.GPSLatitudeRef)
        lon = _convert_gps_coordinate(gps_info.get(piexif.GPSIFD.GPSLongitude))
        lon_ref = gps_info.get(piexif.GPSIFD.GPSLongitudeRef)
        if lat is not None and lat_ref in (b"S", "S"):
            lat = -lat
        if lon is not None and lon_ref in (b"W", "W"):
            lon = -lon
        if lat is not None and lon is not None:
            metadata["gps"] = {"lat": lat, "lon": lon}

    return metadata


def get_file_time(file_path: str) -> Optional[str]:
    """
    获取文件时间（ISO 8601格式）。

    Args:
        file_path (str): 文件路径

    Returns:
        Optional[str]: ISO 8601时间字符串
    """
    try:
        timestamp = os.path.getmtime(file_path)
        return datetime.fromtimestamp(timestamp).isoformat()
    except Exception:
        return None


def get_image_dimensions(file_path: str) -> Tuple[int, int]:
    """
    获取图片宽高并考虑方向。

    Args:
        file_path (str): 图片文件路径

    Returns:
        Tuple[int, int]: (宽, 高)
    """
    try:
        with Image.open(file_path) as image:
            corrected = ImageOps.exif_transpose(image)
            return corrected.size
    except Exception:
        return 0, 0


def generate_fallback_description(file_path: str) -> str:
    """
    基于文件名生成降级描述。

    Args:
        file_path (str): 图片文件路径

    Returns:
        str: 降级描述文本
    """
    name = os.path.splitext(os.path.basename(file_path))[0]
    tokens = [token for token in re.split(r"[\W_]+", name) if token and not token.isdigit()]
    if not tokens:
        return "一张照片"
    if len(tokens) == 1:
        return f"与{tokens[0]}相关的照片"
    return f"与{tokens[0]}和{tokens[1]}相关的照片"
