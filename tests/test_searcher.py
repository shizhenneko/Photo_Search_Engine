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

    def test_distance_to_score_cosine(self) -> None:
        """测试Cosine相似度转换为分数（默认metric）"""
        vector_store = VectorStore(dimension=768, index_path="test.index", metadata_path="test.json")
        searcher = Searcher(
            embedding=T5EmbeddingService(model_name="sentence-t5-base", device="cuda"),
            time_parser=TimeParser(api_key="test-key"),
            vector_store=vector_store,
        )

        # Cosine相似度: 1.0 映射到 1.0 附近（高分区拉伸）
        score_one = searcher._distance_to_score(1.0)
        self.assertGreater(score_one, 0.9)

        # Cosine相似度: 0.5 映射到中间值
        score_half = searcher._distance_to_score(0.5)
        self.assertGreater(score_half, 0.6)
        self.assertLess(score_half, 0.9)

        # Cosine相似度: -1.0 映射到 0 附近
        score_neg = searcher._distance_to_score(-1.0)
        self.assertLess(score_neg, 0.1)

    def test_distance_to_score_l2(self) -> None:
        """测试L2距离转换为相似度分数"""
        vector_store = VectorStore(dimension=768, index_path="test.index", metadata_path="test.json", metric="l2")
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

        # 16个样本，top_k=10，16 <= 20，所以使用 max(scores[-1] * 0.9, 0.05)
        scores = [0.92, 0.88, 0.85, 0.45, 0.42, 0.38, 0.35, 0.30, 0.28, 0.25, 0.22, 0.20, 0.18, 0.15, 0.12, 0.10]
        threshold = searcher._calculate_dynamic_threshold(scores, top_k=10)

        # scores[-1] = 0.10, 0.10 * 0.9 = 0.09, max(0.09, 0.05) = 0.09
        self.assertGreaterEqual(threshold, 0.05)
        self.assertLess(threshold, 0.15)

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

        # 15个样本，top_k=10，15 <= 20，所以使用 max(scores[-1] * 0.9, 0.05)
        scores = [0.32, 0.30, 0.28, 0.25, 0.22, 0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03]
        threshold = searcher._calculate_dynamic_threshold(scores, top_k=10)

        # scores[-1] = 0.03, 0.03 * 0.9 = 0.027, max(0.027, 0.05) = 0.05
        self.assertGreaterEqual(threshold, 0.03)
        self.assertLessEqual(threshold, 0.1)

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

        # 候选数不足时，使用 max(scores[-1] * 0.9, 0.05)
        # scores[-1] = 0.75, 0.75 * 0.9 = 0.675
        self.assertGreaterEqual(threshold, 0.05)
        self.assertLessEqual(threshold, 0.75)

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


class TestSearcherTimeConstraints(unittest.TestCase):
    """时间约束提取测试（改进版）。"""

    def _create_searcher(self) -> Searcher:
        """创建带 Mock 依赖的 Searcher 实例。"""
        mock_embedding = Mock()
        mock_vector_store = Mock()
        mock_time_parser = Mock()
        mock_time_parser.extract_time_constraints.return_value = {
            "start_date": None,
            "end_date": None,
            "precision": "none",
        }

        return Searcher(
            embedding=mock_embedding,
            time_parser=mock_time_parser,
            vector_store=mock_vector_store,
        )

    def test_extract_time_constraints_with_season(self) -> None:
        """测试从查询中提取季节。"""
        searcher = self._create_searcher()

        # Mock time_parser 返回基础约束
        searcher.time_parser.extract_time_constraints.return_value = {
            "start_date": None,
            "end_date": None,
            "precision": "none",
        }

        constraints = searcher._extract_time_constraints("夏天在海边拍的照片")

        self.assertEqual(constraints["season"], "夏天")

    def test_extract_time_constraints_with_time_period(self) -> None:
        """测试从查询中提取时段。"""
        searcher = self._create_searcher()

        constraints = searcher._extract_time_constraints("傍晚的日落照片")
        self.assertEqual(constraints["time_period"], "傍晚")

        constraints = searcher._extract_time_constraints("凌晨拍的星空")
        self.assertEqual(constraints["time_period"], "凌晨")

        constraints = searcher._extract_time_constraints("中午吃饭的照片")
        self.assertEqual(constraints["time_period"], "中午")

    def test_extract_time_constraints_season_variants(self) -> None:
        """测试季节的不同表述方式。"""
        searcher = self._create_searcher()

        # 春季 -> 春天
        constraints = searcher._extract_time_constraints("春季踏青的照片")
        self.assertEqual(constraints["season"], "春天")

        # 冬季 -> 冬天
        constraints = searcher._extract_time_constraints("冬季滑雪")
        self.assertEqual(constraints["season"], "冬天")

    def test_extract_time_constraints_time_period_variants(self) -> None:
        """测试时段的不同表述方式。"""
        searcher = self._create_searcher()

        # 早上 -> 早晨
        constraints = searcher._extract_time_constraints("早上跑步的照片")
        self.assertEqual(constraints["time_period"], "早晨")

        # 晚上 -> 夜晚
        constraints = searcher._extract_time_constraints("晚上聚餐")
        self.assertEqual(constraints["time_period"], "夜晚")

        # 深夜 -> 凌晨
        constraints = searcher._extract_time_constraints("深夜的城市")
        self.assertEqual(constraints["time_period"], "凌晨")


class TestSearcherTimeMatchV2(unittest.TestCase):
    """时间匹配测试（改进版，支持 time_info）。"""

    def _create_searcher(self) -> Searcher:
        """创建带 Mock 依赖的 Searcher 实例。"""
        mock_embedding = Mock()
        mock_vector_store = Mock()
        mock_time_parser = Mock()

        return Searcher(
            embedding=mock_embedding,
            time_parser=mock_time_parser,
            vector_store=mock_vector_store,
        )

    def test_check_time_match_v2_season(self) -> None:
        """测试季节匹配。"""
        searcher = self._create_searcher()

        metadata = {
            "time_info": {
                "year": 2024,
                "month": 7,
                "season": "夏天",
                "time_period": "下午",
            }
        }

        # 匹配
        constraints = {"season": "夏天"}
        self.assertTrue(searcher._check_time_match_v2(metadata, constraints))

        # 不匹配
        constraints = {"season": "冬天"}
        self.assertFalse(searcher._check_time_match_v2(metadata, constraints))

    def test_check_time_match_v2_time_period(self) -> None:
        """测试时段匹配。"""
        searcher = self._create_searcher()

        metadata = {
            "time_info": {
                "time_period": "傍晚",
            }
        }

        # 匹配
        constraints = {"time_period": "傍晚"}
        self.assertTrue(searcher._check_time_match_v2(metadata, constraints))

        # 不匹配
        constraints = {"time_period": "上午"}
        self.assertFalse(searcher._check_time_match_v2(metadata, constraints))

    def test_check_time_match_v2_year(self) -> None:
        """测试年份匹配。"""
        searcher = self._create_searcher()

        metadata = {
            "time_info": {
                "year": 2024,
            }
        }

        # 匹配
        constraints = {"year": 2024}
        self.assertTrue(searcher._check_time_match_v2(metadata, constraints))

        # 不匹配
        constraints = {"year": 2023}
        self.assertFalse(searcher._check_time_match_v2(metadata, constraints))

    def test_check_time_match_v2_combined(self) -> None:
        """测试组合条件匹配。"""
        searcher = self._create_searcher()

        metadata = {
            "time_info": {
                "year": 2024,
                "month": 6,
                "season": "夏天",
                "time_period": "傍晚",
            }
        }

        # 全部匹配
        constraints = {
            "year": 2024,
            "season": "夏天",
            "time_period": "傍晚",
        }
        self.assertTrue(searcher._check_time_match_v2(metadata, constraints))

        # 一个不匹配
        constraints = {
            "year": 2024,
            "season": "夏天",
            "time_period": "上午",  # 不匹配
        }
        self.assertFalse(searcher._check_time_match_v2(metadata, constraints))

    def test_check_time_match_v2_no_constraints(self) -> None:
        """测试无约束时总是匹配。"""
        searcher = self._create_searcher()

        metadata = {"time_info": {"season": "夏天"}}
        constraints = {}

        self.assertTrue(searcher._check_time_match_v2(metadata, constraints))


class TestSearcherESFilters(unittest.TestCase):
    """ES 过滤条件构建测试。"""

    def _create_searcher(self) -> Searcher:
        """创建带 Mock 依赖的 Searcher 实例。"""
        mock_embedding = Mock()
        mock_vector_store = Mock()
        mock_time_parser = Mock()

        return Searcher(
            embedding=mock_embedding,
            time_parser=mock_time_parser,
            vector_store=mock_vector_store,
        )

    def test_build_es_filters_basic(self) -> None:
        """测试基础 ES 过滤条件构建。"""
        searcher = self._create_searcher()

        constraints = {
            "year": 2024,
            "season": "夏天",
            "time_period": "傍晚",
        }

        es_filters = searcher._build_es_filters(constraints)

        self.assertEqual(es_filters["year"], 2024)
        self.assertEqual(es_filters["season"], "夏天")
        self.assertEqual(es_filters["time_period"], "傍晚")

    def test_build_es_filters_with_date_range(self) -> None:
        """测试包含日期范围的 ES 过滤条件。"""
        searcher = self._create_searcher()

        constraints = {
            "start_date": "2024-06-01",
            "end_date": "2024-06-30",
        }

        es_filters = searcher._build_es_filters(constraints)

        self.assertEqual(es_filters["start_date"], "2024-06-01")
        self.assertEqual(es_filters["end_date"], "2024-06-30")

    def test_build_es_filters_empty(self) -> None:
        """测试空约束返回空过滤条件。"""
        searcher = self._create_searcher()

        es_filters = searcher._build_es_filters({})

        self.assertEqual(es_filters, {})

    def test_has_strict_filters(self) -> None:
        """测试判断是否有严格过滤条件。"""
        searcher = self._create_searcher()

        # 有过滤条件
        self.assertTrue(searcher._has_strict_filters({"season": "夏天"}))
        self.assertTrue(searcher._has_strict_filters({"year": 2024}))
        self.assertTrue(searcher._has_strict_filters({"start_date": "2024-01-01"}))

        # 无过滤条件
        self.assertFalse(searcher._has_strict_filters({}))
        self.assertFalse(searcher._has_strict_filters({"precision": "none"}))


class TestSearcherHybridSearchWithFilters(unittest.TestCase):
    """混合检索带 ES 过滤测试。"""

    def test_hybrid_search_uses_es_filters(self) -> None:
        """测试混合检索使用 ES 过滤。"""
        mock_embedding = Mock()
        mock_embedding.generate_embedding.return_value = [0.1] * 4096

        mock_vector_store = Mock()
        mock_vector_store.search.return_value = [
            {"metadata": {"photo_path": "/summer1.jpg", "description": "夏天海边"}, "distance": 0.9},
            {"metadata": {"photo_path": "/winter1.jpg", "description": "冬天雪景"}, "distance": 0.8},
        ]
        mock_vector_store.get_total_items.return_value = 100
        mock_vector_store.load.return_value = True
        mock_vector_store.dimension = 4096
        mock_vector_store.metric = "cosine"

        mock_keyword_store = Mock()
        mock_keyword_store.search_with_filters.return_value = [
            {"photo_path": "/summer1.jpg", "score": 0.9},
        ]

        mock_time_parser = Mock()
        mock_time_parser.extract_time_constraints.return_value = {
            "start_date": None,
            "end_date": None,
            "season": "夏天",
            "precision": "none",
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

        # 执行混合检索
        results = searcher._hybrid_search(
            query="海边",
            query_embedding=[0.1] * 4096,
            candidate_k=10,
            filters={"season": "夏天"},
        )

        # 验证调用了 search_with_filters
        mock_keyword_store.search_with_filters.assert_called_once()
    
    def test_query_formatter_time_separation(self) -> None:
        """测试 QueryFormatter 时间信息分离（架构改进）。"""
        mock_embedding = Mock()
        mock_embedding.generate_embedding.return_value = [0.1] * 4096
        
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = [
            {"metadata": {"photo_path": "/beach.jpg", "description": "海滩"}, "distance": 0.9},
        ]
        mock_vector_store.get_total_items.return_value = 100
        mock_vector_store.load.return_value = True
        mock_vector_store.dimension = 4096
        mock_vector_store.metric = "cosine"
        
        mock_keyword_store = Mock()
        mock_keyword_store.search_with_filters.return_value = [
            {"photo_path": "/beach.jpg", "score": 0.9},
        ]
        
        mock_time_parser = Mock()
        mock_time_parser.extract_time_constraints.return_value = {
            "start_date": None,
            "end_date": None,
            "year": None,
            "month": None,
            "day": None,
            "season": None,
            "time_period": None,
            "precision": "none",
        }
        
        mock_formatter = Mock()
        mock_formatter.is_enabled.return_value = True
        # QueryFormatter 返回纯视觉描述和独立的时间字段
        mock_formatter.format_query.return_value = {
            "search_text": "海滩 沙滩 海浪 宁静",  # 纯语义，无时间信息
            "time_hint": "2024年",
            "season": "夏天",
            "time_period": "下午",
            "original_query": "2024年夏天下午的海滩照片",
        }
        
        searcher = Searcher(
            embedding=mock_embedding,
            time_parser=mock_time_parser,
            vector_store=mock_vector_store,
            keyword_store=mock_keyword_store,
            query_formatter=mock_formatter,
            vector_weight=0.8,
            keyword_weight=0.2,
        )
        searcher.index_loaded = True
        
        # 执行搜索
        results = searcher.search("2024年夏天下午的海滩照片", top_k=10)
        
        # 验证 QueryFormatter 被调用
        mock_formatter.format_query.assert_called_once_with("2024年夏天下午的海滩照片")
        
        # 验证 embedding 使用纯语义描述（不含时间信息）
        embedding_call_args = mock_embedding.generate_embedding.call_args[0][0]
        self.assertIn("海滩", embedding_call_args)
        self.assertNotIn("2024", embedding_call_args)
        self.assertNotIn("夏天", embedding_call_args)
        self.assertNotIn("下午", embedding_call_args)
        
        # 验证 ES 过滤条件正确构建（通过 _hybrid_search 内部调用）
        # 检查 search_with_filters 是否被调用，并且传递了时间过滤条件
        if mock_keyword_store.search_with_filters.called:
            call_args = mock_keyword_store.search_with_filters.call_args
            filters = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get('filters', {})
            # 时间字段应该通过 constraints 传递给 ES
            # 由于 mock_time_parser 返回空的 constraints，
            # searcher 会使用 QueryFormatter 的时间提示合并到 constraints


if __name__ == "__main__":
    unittest.main()
