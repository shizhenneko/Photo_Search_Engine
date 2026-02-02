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

            index_path = os.path.join(data_dir, "index.bin")
            metadata_path = os.path.join(data_dir, "metadata.json")
            vector_store = VectorStore(768, index_path=index_path, metadata_path=metadata_path)

            indexer = Indexer(
                photo_dir=photo_dir,
                vision=LocalVisionLLMService(),
                embedding=T5EmbeddingService(model_name="sentence-t5-base", device="cuda"),
                vector_store=vector_store,
                data_dir=data_dir,
            )

            result = indexer.build_index()

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["indexed_count"], 10)
            self.assertLess(result["fallback_ratio"], 0.1)
            self.assertTrue(os.path.exists(os.path.join(data_dir, "index_ready.marker")))


if __name__ == "__main__":
    unittest.main()
