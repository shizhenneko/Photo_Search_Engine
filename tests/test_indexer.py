import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import get_config

from PIL import Image
from core.indexer import Indexer
from utils.embedding_service import T5EmbeddingService
from utils.vision_llm_service import LocalVisionLLMService
from utils.vector_store import VectorStore


def _create_image(path: str, size: tuple[int, int] = (64, 48)) -> None:
    image = Image.new("RGB", size, color=(255, 0, 0))
    image.save(path)


class IndexerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """加载配置"""
        cls.config = get_config()
        cls.has_api_key = bool(cls.config.get("OPENROUTER_API_KEY"))

    def test_indexer_init(self) -> None:
        """测试Indexer初始化"""
        with tempfile.TemporaryDirectory() as tmp:
            photo_dir = os.path.join(tmp, "photos")
            data_dir = os.path.join(tmp, "data")
            os.makedirs(photo_dir)
            os.makedirs(data_dir)

            index_path = os.path.join(data_dir, "photo_search.index")
            metadata_path = os.path.join(data_dir, "metadata.json")
            vector_store = VectorStore(dimension=768, index_path=index_path, metadata_path=metadata_path)

            indexer = Indexer(
                photo_dir=photo_dir,
                vision=LocalVisionLLMService(),
                embedding=T5EmbeddingService(model_name="sentence-t5-base"),
                vector_store=vector_store,
                data_dir=data_dir,
            )

            self.assertEqual(indexer.photo_dir, os.path.abspath(photo_dir))
            self.assertEqual(indexer.data_dir, data_dir)
            self.assertEqual(indexer.batch_size, 10)

    def test_scan_photos_filters_and_sorts(self) -> None:
        """测试扫描照片-过滤和排序"""
        with tempfile.TemporaryDirectory() as tmp:
            photo_dir = os.path.join(tmp, "photos")
            os.makedirs(photo_dir)

            img1 = os.path.join(photo_dir, "a.jpg")
            img2 = os.path.join(photo_dir, "b.jpg")
            txt = os.path.join(photo_dir, "readme.txt")
            sub_dir = os.path.join(photo_dir, "sub")
            os.makedirs(sub_dir)
            img3 = os.path.join(sub_dir, "c.jpg")

            _create_image(img1)
            _create_image(img2)
            _create_image(img3)
            with open(txt, "w", encoding="utf-8") as file:
                file.write("x")

            os.utime(img1, (1, 1))
            os.utime(img2, (2, 2))
            os.utime(img3, (3, 3))

            index_path = os.path.join(tmp, "index.bin")
            metadata_path = os.path.join(tmp, "metadata.json")
            vector_store = VectorStore(dimension=768, index_path=index_path, metadata_path=metadata_path)

            indexer = Indexer(
                photo_dir=photo_dir,
                vision=LocalVisionLLMService(),
                embedding=T5EmbeddingService(model_name="sentence-t5-base", device="cuda"),
                vector_store=vector_store,
                data_dir=tmp,
            )
            photos = indexer.scan_photos()

            self.assertEqual(len(photos), 3)
            self.assertEqual(photos[0], img1)
            self.assertEqual(photos[1], img2)
            self.assertEqual(photos[2], img3)

    def test_scan_photos_empty_directory(self) -> None:
        """测试扫描空目录"""
        with tempfile.TemporaryDirectory() as tmp:
            photo_dir = os.path.join(tmp, "photos")
            os.makedirs(photo_dir)

            index_path = os.path.join(tmp, "index.bin")
            metadata_path = os.path.join(tmp, "metadata.json")
            vector_store = VectorStore(dimension=768, index_path=index_path, metadata_path=metadata_path)

            indexer = Indexer(
                photo_dir=photo_dir,
                vision=LocalVisionLLMService(),
                embedding=T5EmbeddingService(model_name="sentence-t5-base", device="cuda"),
                vector_store=vector_store,
                data_dir=tmp,
            )
            photos = indexer.scan_photos()

            self.assertEqual(photos, [])

    def test_generate_description_with_local_service(self) -> None:
        """测试使用本地Vision服务生成描述"""
        with tempfile.TemporaryDirectory() as tmp:
            photo_dir = os.path.join(tmp, "photos")
            os.makedirs(photo_dir)

            img_path = os.path.join(photo_dir, "test.jpg")
            _create_image(img_path, size=(100, 80))

            index_path = os.path.join(tmp, "index.bin")
            metadata_path = os.path.join(tmp, "metadata.json")
            vector_store = VectorStore(dimension=768, index_path=index_path, metadata_path=metadata_path)

            indexer = Indexer(
                photo_dir=photo_dir,
                vision=LocalVisionLLMService(),
                embedding=T5EmbeddingService(model_name="sentence-t5-base", device="cuda"),
                vector_store=vector_store,
                data_dir=tmp,
            )

            description = indexer.generate_description(img_path)
            self.assertIsInstance(description, str)
            self.assertGreater(len(description), 0)

    def test_get_status_initial(self) -> None:
        """测试获取初始状态"""
        with tempfile.TemporaryDirectory() as tmp:
            photo_dir = os.path.join(tmp, "photos")
            os.makedirs(photo_dir)

            index_path = os.path.join(tmp, "index.bin")
            metadata_path = os.path.join(tmp, "metadata.json")
            vector_store = VectorStore(dimension=768, index_path=index_path, metadata_path=metadata_path)

            indexer = Indexer(
                photo_dir=photo_dir,
                vision=LocalVisionLLMService(),
                embedding=T5EmbeddingService(model_name="sentence-t5-base", device="cuda"),
                vector_store=vector_store,
                data_dir=tmp,
            )

            status = indexer.get_status()
            self.assertEqual(status["status"], "idle")
            self.assertEqual(status["indexed_count"], 0)

    def test_build_index_fails_when_no_photos(self) -> None:
        """测试无照片时构建索引失败"""
        with tempfile.TemporaryDirectory() as tmp:
            photo_dir = os.path.join(tmp, "photos")
            data_dir = os.path.join(tmp, "data")
            os.makedirs(photo_dir)
            os.makedirs(data_dir)

            index_path = os.path.join(data_dir, "photo_search.index")
            metadata_path = os.path.join(data_dir, "metadata.json")
            vector_store = VectorStore(768, index_path=index_path, metadata_path=metadata_path)

            indexer = Indexer(
                photo_dir=photo_dir,
                vision=LocalVisionLLMService(),
                embedding=T5EmbeddingService(model_name="sentence-t5-base", device="cuda"),
                vector_store=vector_store,
                data_dir=data_dir,
                batch_size=5,
            )

            result = indexer.build_index()
            self.assertEqual(result["status"], "failed")
            self.assertIn("未找到可索引的图片文件", result["message"])

    def test_compute_fallback_ratio(self) -> None:
        """测试降级描述占比计算"""
        with tempfile.TemporaryDirectory() as tmp:
            photo_dir = os.path.join(tmp, "photos")
            data_dir = os.path.join(tmp, "data")
            os.makedirs(photo_dir)
            os.makedirs(data_dir)

            # 创建10张测试图片
            for i in range(10):
                _create_image(os.path.join(photo_dir, f"photo_{i}.jpg"))

            index_path = os.path.join(data_dir, "index.bin")
            metadata_path = os.path.join(data_dir, "metadata.json")
            vector_store = VectorStore(768, index_path=index_path, metadata_path=metadata_path)

            indexer = Indexer(
                photo_dir=photo_dir,
                vision=LocalVisionLLMService(),
                embedding=T5EmbeddingService(model_name="sentence-t5-base", device="cuda"),
                vector_store=vector_store,
                data_dir=data_dir,
                batch_size=5,
            )

            result = indexer.build_index()

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["indexed_count"], 10)
            self.assertLess(result["fallback_ratio"], 0.1)
            self.assertTrue(os.path.exists(os.path.join(data_dir, "index_ready.marker")))


class TestBuildSearchText(unittest.TestCase):
    """测试 _build_search_text() 方法 - 只返回纯 description。"""

    def test_build_search_text_returns_only_description(self) -> None:
        """测试 _build_search_text 只返回纯 description，不包含时间/文件名。"""
        with tempfile.TemporaryDirectory() as tmp:
            photo_dir = os.path.join(tmp, "photos")
            os.makedirs(photo_dir)

            index_path = os.path.join(tmp, "index.bin")
            metadata_path = os.path.join(tmp, "metadata.json")
            vector_store = VectorStore(dimension=768, index_path=index_path, metadata_path=metadata_path)

            indexer = Indexer(
                photo_dir=photo_dir,
                vision=LocalVisionLLMService(),
                embedding=T5EmbeddingService(model_name="sentence-t5-base"),
                vector_store=vector_store,
                data_dir=tmp,
            )

            description = "这是一张美丽的海边日落照片，天空呈现橙红色，波浪轻轻拍打沙滩。"
            exif_data = {"datetime": "2024-06-15T18:30:00", "camera": "iPhone 15"}
            file_time = "2024-06-15T18:30:00"

            search_text = indexer._build_search_text(description, "/photos/sunset.jpg", exif_data, file_time)

            # 应该只返回纯 description，不包含时间/文件名信息
            self.assertEqual(search_text, description)
            self.assertNotIn("年", search_text)
            self.assertNotIn("月", search_text)
            self.assertNotIn("季节", search_text)
            self.assertNotIn("时段", search_text)
            self.assertNotIn("文件名", search_text)

    def test_build_search_text_short_description(self) -> None:
        """测试短 description 返回空字符串。"""
        with tempfile.TemporaryDirectory() as tmp:
            photo_dir = os.path.join(tmp, "photos")
            os.makedirs(photo_dir)

            index_path = os.path.join(tmp, "index.bin")
            metadata_path = os.path.join(tmp, "metadata.json")
            vector_store = VectorStore(dimension=768, index_path=index_path, metadata_path=metadata_path)

            indexer = Indexer(
                photo_dir=photo_dir,
                vision=LocalVisionLLMService(),
                embedding=T5EmbeddingService(model_name="sentence-t5-base"),
                vector_store=vector_store,
                data_dir=tmp,
            )

            short_desc = "短描述"
            search_text = indexer._build_search_text(short_desc, "/photos/test.jpg", None, None)

            self.assertEqual(search_text, "")


class TestExtractTimeInfo(unittest.TestCase):
    """测试 _extract_time_info() 方法 - 7档时段细分。"""

    def _create_indexer(self, tmp_dir: str) -> Indexer:
        """创建 Indexer 实例用于测试。"""
        photo_dir = os.path.join(tmp_dir, "photos")
        os.makedirs(photo_dir, exist_ok=True)

        index_path = os.path.join(tmp_dir, "index.bin")
        metadata_path = os.path.join(tmp_dir, "metadata.json")
        vector_store = VectorStore(dimension=768, index_path=index_path, metadata_path=metadata_path)

        return Indexer(
            photo_dir=photo_dir,
            vision=LocalVisionLLMService(),
            embedding=T5EmbeddingService(model_name="sentence-t5-base"),
            vector_store=vector_store,
            data_dir=tmp_dir,
        )

    def test_extract_time_info_morning(self) -> None:
        """测试早晨时段提取 (5:00-8:00)。"""
        with tempfile.TemporaryDirectory() as tmp:
            indexer = self._create_indexer(tmp)

            exif_data = {"datetime": "2024-06-15T06:30:00"}
            time_info = indexer._extract_time_info(exif_data, None)

            self.assertEqual(time_info["year"], 2024)
            self.assertEqual(time_info["month"], 6)
            self.assertEqual(time_info["day"], 15)
            self.assertEqual(time_info["hour"], 6)
            self.assertEqual(time_info["season"], "夏天")
            self.assertEqual(time_info["time_period"], "早晨")
            self.assertEqual(time_info["weekday"], "星期六")

    def test_extract_time_info_noon(self) -> None:
        """测试中午时段提取 (12:00-14:00)。"""
        with tempfile.TemporaryDirectory() as tmp:
            indexer = self._create_indexer(tmp)

            exif_data = {"datetime": "2024-01-15T12:30:00"}
            time_info = indexer._extract_time_info(exif_data, None)

            self.assertEqual(time_info["time_period"], "中午")
            self.assertEqual(time_info["season"], "冬天")

    def test_extract_time_info_evening(self) -> None:
        """测试傍晚时段提取 (17:00-19:00)。"""
        with tempfile.TemporaryDirectory() as tmp:
            indexer = self._create_indexer(tmp)

            exif_data = {"datetime": "2024-09-20T18:00:00"}
            time_info = indexer._extract_time_info(exif_data, None)

            self.assertEqual(time_info["time_period"], "傍晚")
            self.assertEqual(time_info["season"], "秋天")

    def test_extract_time_info_night(self) -> None:
        """测试夜晚时段提取 (19:00-24:00)。"""
        with tempfile.TemporaryDirectory() as tmp:
            indexer = self._create_indexer(tmp)

            exif_data = {"datetime": "2024-03-10T21:00:00"}
            time_info = indexer._extract_time_info(exif_data, None)

            self.assertEqual(time_info["time_period"], "夜晚")
            self.assertEqual(time_info["season"], "春天")

    def test_extract_time_info_dawn(self) -> None:
        """测试凌晨时段提取 (0:00-5:00)。"""
        with tempfile.TemporaryDirectory() as tmp:
            indexer = self._create_indexer(tmp)

            exif_data = {"datetime": "2024-07-04T03:00:00"}
            time_info = indexer._extract_time_info(exif_data, None)

            self.assertEqual(time_info["time_period"], "凌晨")

    def test_extract_time_info_afternoon(self) -> None:
        """测试下午时段提取 (14:00-17:00)。"""
        with tempfile.TemporaryDirectory() as tmp:
            indexer = self._create_indexer(tmp)

            exif_data = {"datetime": "2024-05-01T15:30:00"}
            time_info = indexer._extract_time_info(exif_data, None)

            self.assertEqual(time_info["time_period"], "下午")

    def test_extract_time_info_forenoon(self) -> None:
        """测试上午时段提取 (8:00-12:00)。"""
        with tempfile.TemporaryDirectory() as tmp:
            indexer = self._create_indexer(tmp)

            exif_data = {"datetime": "2024-11-11T10:00:00"}
            time_info = indexer._extract_time_info(exif_data, None)

            self.assertEqual(time_info["time_period"], "上午")

    def test_extract_time_info_no_datetime(self) -> None:
        """测试无时间数据时返回空值。"""
        with tempfile.TemporaryDirectory() as tmp:
            indexer = self._create_indexer(tmp)

            time_info = indexer._extract_time_info(None, None)

            self.assertIsNone(time_info["year"])
            self.assertIsNone(time_info["time_period"])
            self.assertIsNone(time_info["season"])

    def test_extract_time_info_fallback_to_file_time(self) -> None:
        """测试 EXIF 无时间时回退到文件时间。"""
        with tempfile.TemporaryDirectory() as tmp:
            indexer = self._create_indexer(tmp)

            file_time = "2024-08-20T09:00:00"
            time_info = indexer._extract_time_info({}, file_time)

            self.assertEqual(time_info["year"], 2024)
            self.assertEqual(time_info["month"], 8)
            self.assertEqual(time_info["time_period"], "上午")
            self.assertEqual(time_info["season"], "夏天")


if __name__ == "__main__":
    unittest.main()
