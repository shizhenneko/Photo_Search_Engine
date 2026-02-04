import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

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


class TestSearcherHybridSearch(unittest.TestCase):
    """混合检索测试。"""
    
    def test_hybrid_search_combines_scores(self) -> None:
        """测试混合检索正确融合向量和关键字分数。"""
        # Mock 依赖
        mock_embedding = Mock()
        mock_embedding.generate_embedding.return_value = [0.1] * 4096
        
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = [
            {"metadata": {"photo_path": "/a.jpg", "description": "测试A"}, "distance": 0.9},
            {"metadata": {"photo_path": "/b.jpg", "description": "测试B"}, "distance": 0.7},
        ]
        mock_vector_store.get_total_items.return_value = 100
        mock_vector_store.load.return_value = True
        mock_vector_store.dimension = 4096
        mock_vector_store.metric = "cosine"
        
        mock_keyword_store = Mock()
        mock_keyword_store.search.return_value = [
            {"photo_path": "/a.jpg", "score": 0.8},
            {"photo_path": "/c.jpg", "score": 0.6},
        ]
        
        mock_time_parser = Mock()
        mock_time_parser.extract_time_constraints.return_value = {
            "start_date": None, "end_date": None, "precision": "none"
        }
        
        searcher = Searcher(
            embedding=mock_embedding,
            time_parser=mock_time_parser,
            vector_store=mock_vector_store,
            keyword_store=mock_keyword_store,
            vector_weight=0.8,
            keyword_weight=0.2,
        )
        searcher.index_loaded = True
        
        results = searcher.search("测试查询", top_k=10)
        
        # 验证结果包含向量和关键字两边的照片
        paths = [r["photo_path"] for r in results]
        self.assertIn("/a.jpg", paths)  # 两边都有
    
    def test_hybrid_search_degrades_without_keyword_store(self) -> None:
        """测试无 KeywordStore 时降级为纯向量检索。"""
        mock_embedding = Mock()
        mock_embedding.generate_embedding.return_value = [0.1] * 512
        
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = [
            {"metadata": {"photo_path": "/a.jpg", "description": "测试"}, "distance": 0.9},
        ]
        mock_vector_store.get_total_items.return_value = 10
        mock_vector_store.load.return_value = True
        mock_vector_store.dimension = 512
        mock_vector_store.metric = "cosine"
        
        mock_time_parser = Mock()
        
        searcher = Searcher(
            embedding=mock_embedding,
            time_parser=mock_time_parser,
            vector_store=mock_vector_store,
            keyword_store=None,  # 不传入 KeywordStore
        )
        searcher.index_loaded = True
        
        results = searcher.search("测试查询", top_k=10)
        
        # 应该正常返回结果
        self.assertEqual(len(results), 1)
    
    def test_weight_validation(self) -> None:
        """测试权重必须和为 1。"""
        mock_embedding = Mock()
        mock_vector_store = Mock()
        mock_time_parser = Mock()
        
        with self.assertRaises(ValueError) as context:
            Searcher(
                embedding=mock_embedding,
                time_parser=mock_time_parser,
                vector_store=mock_vector_store,
                vector_weight=0.5,
                keyword_weight=0.3,  # 0.5 + 0.3 != 1
            )
        
        self.assertIn("必须等于 1.0", str(context.exception))


class TestSearcherQueryFormatting(unittest.TestCase):
    """查询格式化集成测试。"""
    
    def test_search_uses_formatter(self) -> None:
        """测试检索时调用查询格式化。"""
        mock_embedding = Mock()
        mock_embedding.generate_embedding.return_value = [0.1] * 4096
        
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = []
        mock_vector_store.load.return_value = True
        mock_vector_store.dimension = 4096
        
        mock_formatter = Mock()
        mock_formatter.is_enabled.return_value = True
        mock_formatter.format_query.return_value = {
            "search_text": "格式化后的查询",
            "time_hint": None,
            "season": None,
        }
        
        searcher = Searcher(
            embedding=mock_embedding,
            time_parser=Mock(),
            vector_store=mock_vector_store,
            query_formatter=mock_formatter,
        )
        searcher.index_loaded = True
        
        searcher.search("原始查询")
        
        mock_formatter.format_query.assert_called_with("原始查询")
        mock_embedding.generate_embedding.assert_called_with("格式化后的查询")


if __name__ == "__main__":
    unittest.main()
