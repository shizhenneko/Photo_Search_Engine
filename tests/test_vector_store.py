import os
import sys
import tempfile
import unittest
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import get_config

from utils.vector_store import VectorStore


class VectorStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """加载配置"""
        cls.config = get_config()
        cls.dimension = cls.config.get("EMBEDDING_DIMENSION", 768)

    def test_vector_store_init(self) -> None:
        """测试向量存储初始化"""
        with tempfile.TemporaryDirectory() as tmp:
            index_path = os.path.join(tmp, "index.bin")
            metadata_path = os.path.join(tmp, "metadata.json")
            store = VectorStore(dimension=self.dimension, index_path=index_path, metadata_path=metadata_path)

            self.assertEqual(store.dimension, self.dimension)
            self.assertEqual(store.index_path, index_path)
            self.assertEqual(store.metadata_path, metadata_path)
            self.assertEqual(store.get_total_items(), 0)

    def test_add_and_search_item(self) -> None:
        """测试添加和搜索向量"""
        with tempfile.TemporaryDirectory() as tmp:
            index_path = os.path.join(tmp, "index.bin")
            metadata_path = os.path.join(tmp, "metadata.json")
            store = VectorStore(dimension=self.dimension, index_path=index_path, metadata_path=metadata_path)

            vec1 = [0.1] * self.dimension
            vec2 = [0.5] * self.dimension
            store.add_item(vec1, {"id": 1, "photo": "test1.jpg"})
            store.add_item(vec2, {"id": 2, "photo": "test2.jpg"})

            results = store.search(vec1, top_k=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["metadata"]["id"], 1)
            self.assertGreater(results[0]["metadata"], 0.0)

    def test_save_and_load_index(self) -> None:
        """测试保存和加载索引"""
        with tempfile.TemporaryDirectory() as tmp:
            index_path = os.path.join(tmp, "index.bin")
            metadata_path = os.path.join(tmp, "metadata.json")
            store = VectorStore(dimension=self.dimension, index_path=index_path, metadata_path=metadata_path)

            store.add_item([0.1] * self.dimension, {"id": 1})
            store.add_item([0.2] * self.dimension, {"id": 2})
            store.save()

            self.assertTrue(os.path.exists(index_path))
            self.assertTrue(os.path.exists(metadata_path))

            new_store = VectorStore(dimension=self.dimension, index_path=index_path, metadata_path=metadata_path)
            self.assertTrue(new_store.load())
            self.assertEqual(new_store.get_total_items(), 2)

    def test_load_nonexistent_index(self) -> None:
        """测试加载不存在的索引"""
        with tempfile.TemporaryDirectory() as tmp:
            index_path = os.path.join(tmp, "nonexistent.bin")
            metadata_path = os.path.join(tmp, "nonexistent.json")
            store = VectorStore(dimension=self.dimension, index_path=index_path, metadata_path=metadata_path)
            self.assertFalse(store.load())

    def test_load_metadata_mismatch(self) -> None:
        """测试索引与元数据数量不匹配"""
        with tempfile.TemporaryDirectory() as tmp:
            index_path = os.path.join(tmp, "index.bin")
            metadata_path = os.path.join(tmp, "metadata.json")
            store = VectorStore(dimension=self.dimension, index_path=index_path, metadata_path=metadata_path)
            store.add_item([0.1] * self.dimension, {"id": 1})
            store.save()

            with open(metadata_path, "w", encoding="utf-8") as file:
                file.write("[]")

            new_store = VectorStore(dimension=self.dimension, index_path=index_path, metadata_path=metadata_path)
            with self.assertRaises(ValueError):
                new_store.load()

    def test_add_item_dimension_mismatch(self) -> None:
        """测试添加维度不匹配的向量"""
        with tempfile.TemporaryDirectory() as tmp:
            index_path = os.path.join(tmp, "index.bin")
            metadata_path = os.path.join(tmp, "metadata.json")
            store = VectorStore(dimension=self.dimension, index_path=index_path, metadata_path=metadata_path)

            store.add_item([0.1] * self.dimension, {"id": 1})

            with self.assertRaises(ValueError):
                store.add_item([0.1] * (self.dimension + 1), {"id": 2})

    def test_search_empty_index(self) -> None:
        """测试搜索空索引"""
        with tempfile.TemporaryDirectory() as tmp:
            index_path = os.path.join(tmp, "index.bin")
            metadata_path = os.path.join(tmp, "metadata.json")
            store = VectorStore(dimension=self.dimension, index_path=index_path, metadata_path=metadata_path)

            results = store.search([0.1] * self.dimension, top_k=10)
            self.assertEqual(len(results), 0)

    def test_search_with_top_k_limit(self) -> None:
        """测试搜索结果数量限制"""
        with tempfile.TemporaryDirectory() as tmp:
            index_path = os.path.join(tmp, "index.bin")
            metadata_path = os.path.join(tmp, "metadata.json")
            store = VectorStore(dimension=self.dimension, index_path=index_path, metadata_path=metadata_path)

            for i in range(10):
                store.add_item([i * 0.1] * self.dimension, {"id": i})

            results = store.search([0.1] * self.dimension, top_k=5)
            self.assertEqual(len(results), 5)


if __name__ == "__main__":
    unittest.main()
