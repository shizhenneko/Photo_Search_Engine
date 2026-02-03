import os
import sys
import unittest
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import get_config

from core.searcher import Searcher
from utils.embedding_service import T5EmbeddingService
from utils.time_parser import TimeParser
from utils.vector_store import VectorStore


class SearcherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """加载配置"""
        cls.config = get_config()
        cls.has_api_key = bool(cls.config.get("OPENROUTER_API_KEY"))

    def test_validate_query_short(self) -> None:
        """测试查询验证-过短"""
        vector_store = VectorStore(dimension=768, index_path="test.index", metadata_path="test.json")
        searcher = Searcher(
            embedding=T5EmbeddingService(model_name="sentence-t5-base", device="cuda"),
            time_parser=TimeParser(api_key="test-key"),
            vector_store=vector_store,
        )
        self.assertFalse(searcher.validate_query("a"))
        self.assertFalse(searcher.validate_query("1234"))

    def test_validate_query_invalid_chars(self) -> None:
        """测试查询验证-非法字符"""
        vector_store = VectorStore(dimension=768, index_path="test.index", metadata_path="test.json")
        searcher = Searcher(
            embedding=T5EmbeddingService(model_name="sentence-t5-base", device="cuda"),
            time_parser=TimeParser(api_key="test-key"),
            vector_store=vector_store,
        )
        self.assertFalse(searcher.validate_query("!!!@@@###"))
        self.assertFalse(searcher.validate_query("   "))

    def test_validate_query_valid(self) -> None:
        """测试查询验证-有效查询"""
        vector_store = VectorStore(dimension=768, index_path="test.index", metadata_path="test.json")
        searcher = Searcher(
            embedding=T5EmbeddingService(model_name="sentence-t5-base", device="cuda"),
            time_parser=TimeParser(api_key="test-key"),
            vector_store=vector_store,
        )
        self.assertTrue(searcher.validate_query("valid query text"))
        self.assertTrue(searcher.validate_query("海边度假的照片"))

    def test_distance_to_score(self) -> None:
        """测试L2距离转换为相似度分数"""
        vector_store = VectorStore(dimension=768, index_path="test.index", metadata_path="test.json")
        searcher = Searcher(
            embedding=T5EmbeddingService(model_name="sentence-t5-base", device="cuda"),
            time_parser=TimeParser(api_key="test-key"),
            vector_store=vector_store,
        )

        score_zero = searcher._distance_to_score(0.0)
        self.assertEqual(score_zero, 1.0)

        score_positive = searcher._distance_to_score(0.5)
        self.assertLess(score_positive, 1.0)
        self.assertGreater(score_positive, 0.0)

        score_large = searcher._distance_to_score(100.0)
        self.assertLess(score_large, 0.01)

    def test_parse_date_iso_format(self) -> None:
        """测试ISO格式日期解析"""
        vector_store = VectorStore(dimension=768, index_path="test.index", metadata_path="test.json")
        searcher = Searcher(
            embedding=T5EmbeddingService(model_name="sentence-t5-base", device="cuda"),
            time_parser=TimeParser(api_key="test-key"),
            vector_store=vector_store,
        )

        result = searcher._parse_date("2024-06-15")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2024)
        self.assertEqual(result.month, 6)
        self.assertEqual(result.day, 15)

    def test_parse_date_datetime_format(self) -> None:
        """测试datetime格式日期解析"""
        vector_store = VectorStore(dimension=768, index_path="test.index", metadata_path="test.json")
        searcher = Searcher(
            embedding=T5EmbeddingService(model_name="sentence-t5-base", device="cuda"),
            time_parser=TimeParser(api_key="test-key"),
            vector_store=vector_store,
        )

        result = searcher._parse_date("2024-06-15T10:30:00")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2024)
        self.assertEqual(result.month, 6)
        self.assertEqual(result.day, 15)

    def test_parse_date_invalid(self) -> None:
        """测试无效日期解析"""
        vector_store = VectorStore(dimension=768, index_path="test.index", metadata_path="test.json")
        searcher = Searcher(
            embedding=T5EmbeddingService(model_name="sentence-t5-base", device="cuda"),
            time_parser=TimeParser(api_key="test-key"),
            vector_store=vector_store,
        )

        result = searcher._parse_date("invalid-date")
        self.assertIsNone(result)

    def test_get_index_stats_not_loaded(self) -> None:
        """测试获取索引统计-未加载"""
        vector_store = VectorStore(dimension=768, index_path="test.index", metadata_path="test.json")
        searcher = Searcher(
            embedding=T5EmbeddingService(model_name="sentence-t5-base", device="cuda"),
            time_parser=TimeParser(api_key="test-key"),
            vector_store=vector_store,
        )

        stats = searcher.get_index_stats()
        self.assertEqual(stats["total_items"], 0)
        self.assertFalse(stats["index_loaded"])
        self.assertIsNone(stats["vector_dimension"])

    def test_search_without_loaded_index(self) -> None:
        """测试未加载索引时搜索"""
        vector_store = VectorStore(dimension=768, index_path="test.index", metadata_path="test.json")
        searcher = Searcher(
            embedding=T5EmbeddingService(model_name="sentence-t5-base", device="cuda"),
            time_parser=TimeParser(api_key="test-key"),
            vector_store=vector_store,
        )

        with self.assertRaises(ValueError):
            searcher.search("test query")

    def test_calculate_dynamic_threshold_high_quality(self) -> None:
        """测试动态阈值计算-高质量查询场景"""
        vector_store = VectorStore(dimension=768, index_path="test.index", metadata_path="test.json")
        searcher = Searcher(
            embedding=T5EmbeddingService(model_name="sentence-t5-base", device="cuda"),
            time_parser=TimeParser(api_key="test-key"),
            vector_store=vector_store,
        )

        scores = [0.92, 0.88, 0.85, 0.45, 0.42, 0.38, 0.35, 0.30, 0.28, 0.25, 0.22, 0.20, 0.18, 0.15, 0.12, 0.10]
        threshold = searcher._calculate_dynamic_threshold(scores, top_k=10)

        self.assertGreater(threshold, 0.15)
        self.assertLess(threshold, 0.5)

    def test_calculate_dynamic_threshold_uniform_distribution(self) -> None:
        """测试动态阈值计算-均匀分布场景"""
        vector_store = VectorStore(dimension=768, index_path="test.index", metadata_path="test.json")
        searcher = Searcher(
            embedding=T5EmbeddingService(model_name="sentence-t5-base", device="cuda"),
            time_parser=TimeParser(api_key="test-key"),
            vector_store=vector_store,
        )

        scores = [0.65, 0.62, 0.58, 0.55, 0.52, 0.50, 0.48, 0.45, 0.42, 0.40, 0.38, 0.35, 0.32, 0.30, 0.28]
        threshold = searcher._calculate_dynamic_threshold(scores, top_k=10)

        self.assertGreater(threshold, 0.15)
        self.assertLess(threshold, 0.5)

    def test_calculate_dynamic_threshold_low_scores(self) -> None:
        """测试动态阈值计算-低分场景"""
        vector_store = VectorStore(dimension=768, index_path="test.index", metadata_path="test.json")
        searcher = Searcher(
            embedding=T5EmbeddingService(model_name="sentence-t5-base", device="cuda"),
            time_parser=TimeParser(api_key="test-key"),
            vector_store=vector_store,
        )

        scores = [0.32, 0.30, 0.28, 0.25, 0.22, 0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03]
        threshold = searcher._calculate_dynamic_threshold(scores, top_k=10)

        self.assertGreaterEqual(threshold, 0.1)
        self.assertLess(threshold, 0.3)

    def test_calculate_dynamic_threshold_few_candidates(self) -> None:
        """测试动态阈值计算-候选数不足场景"""
        vector_store = VectorStore(dimension=768, index_path="test.index", metadata_path="test.json")
        searcher = Searcher(
            embedding=T5EmbeddingService(model_name="sentence-t5-base", device="cuda"),
            time_parser=TimeParser(api_key="test-key"),
            vector_store=vector_store,
        )

        scores = [0.85, 0.80, 0.75]
        threshold = searcher._calculate_dynamic_threshold(scores, top_k=10)

        self.assertEqual(threshold, 0.05)

    def test_calculate_dynamic_threshold_empty_scores(self) -> None:
        """测试动态阈值计算-空分数列表"""
        vector_store = VectorStore(dimension=768, index_path="test.index", metadata_path="test.json")
        searcher = Searcher(
            embedding=T5EmbeddingService(model_name="sentence-t5-base", device="cuda"),
            time_parser=TimeParser(api_key="test-key"),
            vector_store=vector_store,
        )

        threshold = searcher._calculate_dynamic_threshold([], top_k=10)
        self.assertEqual(threshold, 0.1)


if __name__ == "__main__":
    unittest.main()
