import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Tuple

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import get_config

from PIL import Image
import piexif

from utils.image_parser import (
    extract_exif_metadata,
    generate_fallback_description,
    get_file_time,
    get_image_dimensions,
    is_valid_image,
)


def _create_image(path: str, size: Tuple[int, int] = (64, 48), exif_bytes: bytes | None = None) -> None:
    image = Image.new("RGB", size, color=(10, 20, 30))
    if exif_bytes:
        image.save(path, exif=exif_bytes)
    else:
        image.save(path)


class ImageParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """加载配置"""
        cls.config = get_config()

    def test_is_valid_image_true(self) -> None:
        """测试有效图片验证"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sample.jpg")
            _create_image(path)
            self.assertTrue(is_valid_image(path))

    def test_is_valid_image_false(self) -> None:
        """测试非图片文件验证"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sample.txt")
            with open(path, "w", encoding="utf-8") as file:
                file.write("not an image")
            self.assertFalse(is_valid_image(path))

    def test_extract_exif_metadata_with_exif(self) -> None:
        """测试从有EXIF的图片提取元数据"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "photo.jpg")
            exif_dict = {
                "0th": {
                    piexif.ImageIFD.Make: b"Canon",
                    piexif.ImageIFD.Model: b"EOS",
                    piexif.ImageIFD.Orientation: 1,
                },
                "Exif": {piexif.ExifIFD.DateTimeOriginal: b"2023:07:15 10:20:30"},
            }
            exif_bytes = piexif.dump(exif_dict)
            _create_image(path, exif_bytes=exif_bytes)
            metadata = extract_exif_metadata(path)
            self.assertEqual(metadata["datetime"], "2023-07-15T10:20:30")
            self.assertEqual(metadata["camera"], "Canon EOS")
            self.assertEqual(metadata["orientation"], 1)

    def test_extract_exif_metadata_without_exif(self) -> None:
        """测试从无EXIF的图片提取元数据"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "photo.jpg")
            _create_image(path)
            metadata = extract_exif_metadata(path)
            self.assertIsNone(metadata["datetime"])

    def test_generate_fallback_description(self) -> None:
        """测试从文件名生成降级描述"""
        text = generate_fallback_description("beach_party_2024.jpg")
        self.assertTrue("beach" in text or "party" in text)

    def test_get_image_dimensions(self) -> None:
        """测试获取图片尺寸"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dim.png")
            _create_image(path, size=(120, 80))
            width, height = get_image_dimensions(path)
            self.assertEqual((width, height), (120, 80))

    def test_get_file_time(self) -> None:
        """测试获取文件时间"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "time.jpg")
            _create_image(path)
            self.assertIsNotNone(get_file_time(path))

    def test_supported_image_formats(self) -> None:
        """测试支持的图片格式"""
        supported_formats = [".jpg", ".jpeg", ".png", ".webp"]
        with tempfile.TemporaryDirectory() as tmp:
            for ext in supported_formats:
                path = os.path.join(tmp, f"test{ext}")
                _create_image(path)
                self.assertTrue(is_valid_image(path), f"Failed for {ext}")


if __name__ == "__main__":
    unittest.main()
