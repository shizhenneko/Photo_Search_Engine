import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.searcher import Searcher
from tests.helpers import FakeEmbeddingService, FakeQueryFormatter, FakeTimeParser
from utils.vector_store import VectorStore


class SearcherTests(unittest.TestCase):
    def _build_searcher(self) -> Searcher:
        vector_store = VectorStore(dimension=8, index_path="test.index", metadata_path="test.json")
        return Searcher(
            embedding=FakeEmbeddingService(dimension=8),
            time_parser=FakeTimeParser(),
            vector_store=vector_store,
        )

    def test_validate_query(self) -> None:
        searcher = self._build_searcher()
        self.assertFalse(searcher.validate_query("!"))
        self.assertFalse(searcher.validate_query("!!!@@@###"))
        self.assertTrue(searcher.validate_query("专辑"))
        self.assertTrue(searcher.validate_query("海边度假的照片"))

    def test_distance_to_score_cosine(self) -> None:
        searcher = self._build_searcher()
        self.assertGreater(searcher._distance_to_score(1.0), 0.9)
        self.assertLess(searcher._distance_to_score(-1.0), 0.1)

    def test_distance_to_score_l2(self) -> None:
        vector_store = VectorStore(dimension=8, index_path="test.index", metadata_path="test.json", metric="l2")
        searcher = Searcher(
            embedding=FakeEmbeddingService(dimension=8),
            time_parser=FakeTimeParser(),
            vector_store=vector_store,
        )
        self.assertEqual(searcher._distance_to_score(0.0), 1.0)
        self.assertLess(searcher._distance_to_score(2.0), 1.0)

    def test_get_index_stats_not_loaded(self) -> None:
        searcher = self._build_searcher()
        stats = searcher.get_index_stats()
        self.assertEqual(stats["total_items"], 0)
        self.assertFalse(stats["index_loaded"])

    def test_search_without_loaded_index(self) -> None:
        searcher = self._build_searcher()
        with self.assertRaises(ValueError):
            searcher.search("有效的查询内容")

    def test_hybrid_search_combines_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a_path = os.path.join(tmp, "a.jpg")
            b_path = os.path.join(tmp, "b.jpg")
            c_path = os.path.join(tmp, "c.jpg")
            for path in (a_path, b_path):
                with open(path, "wb") as file:
                    file.write(b"test")

            mock_embedding = FakeEmbeddingService(dimension=8)
            mock_vector_store = Mock()
            mock_vector_store.metric = "cosine"
            mock_vector_store.metadata = [
                {"photo_path": a_path, "description": "测试A"},
                {"photo_path": b_path, "description": "测试B"},
            ]
            mock_vector_store.search.return_value = [
                {"metadata": {"photo_path": a_path, "description": "测试A"}, "distance": 0.9},
                {"metadata": {"photo_path": b_path, "description": "测试B"}, "distance": 0.7},
            ]
            mock_keyword_store = Mock()
            mock_keyword_store.search.return_value = [
                {"photo_path": b_path, "score": 1.0},
                {"photo_path": c_path, "score": 0.5},
            ]

            searcher = Searcher(
                embedding=mock_embedding,
                time_parser=FakeTimeParser(),
                vector_store=mock_vector_store,
                keyword_store=mock_keyword_store,
                vector_weight=0.8,
                keyword_weight=0.2,
            )

            results = searcher._hybrid_search("测试查询", [0.1] * 8, 10, filters=None)
            by_path = {item["photo_path"]: item for item in results}
            self.assertIn(a_path, by_path)
            self.assertIn(b_path, by_path)
            self.assertNotIn(c_path, by_path)
            self.assertGreater(by_path[b_path]["score"], by_path[a_path]["score"] * 0.8)
            self.assertGreater(by_path[b_path]["keyword_score"], 0.0)

    def test_hybrid_search_does_not_penalize_missing_keyword_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vector_only = os.path.join(tmp, "vector-only.jpg")
            both_hit = os.path.join(tmp, "both-hit.jpg")
            for path in (vector_only, both_hit):
                with open(path, "wb") as file:
                    file.write(b"test")

            mock_embedding = FakeEmbeddingService(dimension=8)
            mock_vector_store = Mock()
            mock_vector_store.metric = "cosine"
            mock_vector_store.metadata = [
                {"photo_path": vector_only, "description": "测试A"},
                {"photo_path": both_hit, "description": "测试B"},
            ]
            mock_vector_store.search.return_value = [
                {"metadata": {"photo_path": vector_only, "description": "测试A"}, "distance": 0.9},
                {"metadata": {"photo_path": both_hit, "description": "测试B"}, "distance": 0.7},
            ]
            mock_keyword_store = Mock()
            mock_keyword_store.search.return_value = [
                {"photo_path": both_hit, "score": 1.0},
            ]

            searcher = Searcher(
                embedding=mock_embedding,
                time_parser=FakeTimeParser(),
                vector_store=mock_vector_store,
                keyword_store=mock_keyword_store,
                vector_weight=0.8,
                keyword_weight=0.2,
            )

            results = searcher._hybrid_search("测试查询", [0.1] * 8, 10, filters=None)
            by_path = {item["photo_path"]: item for item in results}
            self.assertGreater(by_path[vector_only]["score"], 0.9)
            self.assertAlmostEqual(by_path[both_hit]["keyword_score"], 1.0)

    def test_hybrid_search_skips_keyword_only_items_missing_from_vector_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            indexed_path = os.path.join(tmp, "indexed.jpg")
            stale_path = os.path.join(tmp, "stale-only.jpg")
            with open(indexed_path, "wb") as file:
                file.write(b"test")

            mock_embedding = FakeEmbeddingService(dimension=8)
            mock_vector_store = Mock()
            mock_vector_store.metric = "cosine"
            mock_vector_store.metadata = [
                {"photo_path": indexed_path, "description": "已索引图片"},
            ]
            mock_vector_store.search.return_value = [
                {"metadata": {"photo_path": indexed_path, "description": "已索引图片"}, "distance": 0.9},
            ]
            mock_keyword_store = Mock()
            mock_keyword_store.search.return_value = [
                {"photo_path": stale_path, "score": 1.0},
                {"photo_path": indexed_path, "score": 0.3},
            ]

            searcher = Searcher(
                embedding=mock_embedding,
                time_parser=FakeTimeParser(),
                vector_store=mock_vector_store,
                keyword_store=mock_keyword_store,
                vector_weight=0.85,
                keyword_weight=0.15,
            )

            results = searcher._hybrid_search("测试查询", [0.1] * 8, 10, filters=None)
            paths = [item["photo_path"] for item in results]
            self.assertIn(indexed_path, paths)
            self.assertNotIn(stale_path, paths)

    def test_hybrid_search_drops_weak_keyword_only_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vector_hit = os.path.join(tmp, "vector.jpg")
            keyword_only = os.path.join(tmp, "keyword-only.jpg")
            for path in (vector_hit, keyword_only):
                with open(path, "wb") as file:
                    file.write(b"test")

            mock_vector_store = Mock()
            mock_vector_store.metric = "cosine"
            mock_vector_store.metadata = [
                {"photo_path": vector_hit, "description": "向量图片"},
                {"photo_path": keyword_only, "description": "弱关键词图片"},
            ]
            mock_vector_store.search.return_value = [
                {"metadata": {"photo_path": vector_hit, "description": "向量图片"}, "distance": 0.92},
            ]

            mock_keyword_store = Mock()
            mock_keyword_store.search.return_value = [
                {"photo_path": keyword_only, "score": 0.3},
            ]

            searcher = Searcher(
                embedding=FakeEmbeddingService(dimension=8),
                time_parser=FakeTimeParser(),
                vector_store=mock_vector_store,
                keyword_store=mock_keyword_store,
                vector_weight=0.85,
                keyword_weight=0.15,
            )

            results = searcher._hybrid_search("测试查询", [0.1] * 8, 20, filters=None)
            paths = [item["photo_path"] for item in results]
            self.assertIn(vector_hit, paths)
            self.assertNotIn(keyword_only, paths)

    def test_check_time_match_v2_rejects_missing_exif_for_time_filters(self) -> None:
        searcher = self._build_searcher()
        metadata = {
            "file_time": "2025-08-17T12:34:56",
            "time_info": {},
            "exif_data": {},
        }

        self.assertFalse(searcher._check_time_match_v2(metadata, {"season": "夏天"}))
        self.assertFalse(searcher._check_time_match_v2(metadata, {"year": 2025}))

    def test_search_by_image_path_excludes_self(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index_path = os.path.join(tmp, "index.bin")
            metadata_path = os.path.join(tmp, "metadata.json")
            vector_store = VectorStore(dimension=8, index_path=index_path, metadata_path=metadata_path)
            embedding = FakeEmbeddingService(dimension=8)
            searcher = Searcher(
                embedding=embedding,
                time_parser=FakeTimeParser(),
                vector_store=vector_store,
            )
            paths = [os.path.join(tmp, f"photo_{index}.jpg") for index in range(3)]
            for index, path in enumerate(paths):
                with open(path, "wb") as file:
                    file.write(b"test")
                vector_store.add_item(
                    [float(index + offset) for offset in range(8)],
                    {
                        "photo_path": path,
                        "description": f"图片 {index}",
                        "retrieval_text": f"图片 {index}",
                    },
                )
            vector_store.save()
            searcher.load_index()

            results = searcher.search_by_image_path(paths[0], top_k=2)
            self.assertEqual(len(results), 2)
            self.assertTrue(all(item["photo_path"] != paths[0] for item in results))

    def test_search_by_image_path_deduplicates_same_photo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index_path = os.path.join(tmp, "index.bin")
            metadata_path = os.path.join(tmp, "metadata.json")
            vector_store = VectorStore(dimension=8, index_path=index_path, metadata_path=metadata_path)
            embedding = FakeEmbeddingService(dimension=8)
            searcher = Searcher(
                embedding=embedding,
                time_parser=FakeTimeParser(),
                vector_store=vector_store,
            )
            query_path = os.path.join(tmp, "query.jpg")
            dup_path = os.path.join(tmp, "dup.jpg")
            other_path = os.path.join(tmp, "other.jpg")
            for path in (query_path, dup_path, other_path):
                with open(path, "wb") as file:
                    file.write(b"test")

            vector_store.add_item([1.0] * 8, {"photo_path": query_path, "description": "query"})
            vector_store.add_item([0.9] * 8, {"photo_path": dup_path, "description": "dup-a"})
            vector_store.add_item([0.9] * 8, {"photo_path": dup_path, "description": "dup-b"})
            vector_store.add_item([0.8] * 8, {"photo_path": other_path, "description": "other"})
            vector_store.save()
            searcher.load_index()

            results = searcher.search_by_image_path(query_path, top_k=3)
            paths = [item["photo_path"] for item in results]
            self.assertEqual(paths.count(dup_path), 1)

    def test_search_by_uploaded_image_uses_temporary_analysis_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index_path = os.path.join(tmp, "index.bin")
            metadata_path = os.path.join(tmp, "metadata.json")
            vector_store = VectorStore(dimension=8, index_path=index_path, metadata_path=metadata_path)
            embedding = FakeEmbeddingService(dimension=8)
            searcher = Searcher(
                embedding=embedding,
                time_parser=FakeTimeParser(),
                vector_store=vector_store,
            )
            indexed_paths = [os.path.join(tmp, f"photo_{index}.jpg") for index in range(3)]
            for index, path in enumerate(indexed_paths):
                with open(path, "wb") as file:
                    file.write(b"test")
                vector_store.add_item(
                    [float(index + offset) for offset in range(8)],
                    {
                        "photo_path": path,
                        "description": f"图片 {index}",
                        "retrieval_text": f"photo 图片 {index}",
                    },
                )

            uploaded_path = os.path.join(tmp, "uploaded.jpg")
            with open(uploaded_path, "wb") as file:
                file.write(b"upload")

            vector_store.save()
            searcher.load_index()

            analysis = {
                "description": "上传图片",
                "outer_scene_summary": "上传图片外层场景",
                "inner_content_summary": "",
                "media_types": ["photo"],
                "tags": ["上传图片", "图片 1"],
                "ocr_text": "",
                "person_roles": [],
                "identity_candidates": [],
                "identity_names": [],
                "identity_evidence": [],
                "analysis_flags": {},
                "retrieval_text": "photo 图片 1 上传图片",
            }

            results = searcher.search_by_uploaded_image(
                uploaded_path,
                analysis=analysis,
                top_k=2,
            )

            self.assertEqual(len(results), 2)
            self.assertTrue(all(item["photo_path"] != uploaded_path for item in results))
            self.assertTrue(all(item["photo_path"] in indexed_paths for item in results))

    def test_identity_query_filters_out_candidates_without_matching_identity_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exact_path = os.path.join(tmp, "exact.jpg")
            generic_path = os.path.join(tmp, "generic.jpg")
            for path in (exact_path, generic_path):
                with open(path, "wb") as file:
                    file.write(b"test")

            vector_store = Mock()
            vector_store.load.return_value = True
            vector_store.dimension = 8
            vector_store.metric = "cosine"
            vector_store.metadata = [
                {
                    "photo_path": exact_path,
                    "description": "舞台歌手",
                    "identity_names": ["陶喆"],
                    "identity_candidates": [{"name": "陶喆", "confidence": 0.95, "evidence_sources": ["face_similarity"]}],
                },
                {
                    "photo_path": generic_path,
                    "description": "男歌手现场",
                    "identity_names": [],
                    "identity_candidates": [],
                },
            ]
            vector_store.get_total_items.return_value = 2
            vector_store.search.return_value = [
                {"metadata": vector_store.metadata[1], "distance": 0.99},
                {"metadata": vector_store.metadata[0], "distance": 0.8},
            ]

            searcher = Searcher(
                embedding=FakeEmbeddingService(dimension=8),
                time_parser=FakeTimeParser(),
                vector_store=vector_store,
                query_formatter=FakeQueryFormatter(
                    {
                        "请帮我找陶喆的照片": {
                            "search_text": "",
                            "media_terms": [],
                            "identity_terms": ["陶喆"],
                            "strict_identity_filter": True,
                            "time_hint": None,
                            "season": None,
                            "time_period": None,
                            "original_query": "请帮我找陶喆的照片",
                        }
                    }
                ),
            )

            results = searcher.search("请帮我找陶喆的照片", top_k=5)
            self.assertEqual([item["photo_path"] for item in results], [exact_path])

    def test_identity_style_query_does_not_force_hard_identity_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exact_path = os.path.join(tmp, "exact.jpg")
            generic_path = os.path.join(tmp, "generic.jpg")
            for path in (exact_path, generic_path):
                with open(path, "wb") as file:
                    file.write(b"test")

            vector_store = Mock()
            vector_store.load.return_value = True
            vector_store.dimension = 8
            vector_store.metric = "cosine"
            vector_store.metadata = [
                {
                    "photo_path": exact_path,
                    "description": "带有陶喆风格的舞台照片",
                    "identity_names": ["陶喆"],
                    "identity_candidates": [{"name": "陶喆", "confidence": 0.95, "evidence_sources": ["face_similarity"]}],
                },
                {
                    "photo_path": generic_path,
                    "description": "类似华语男歌手风格的舞台照片",
                    "identity_names": [],
                    "identity_candidates": [],
                },
            ]
            vector_store.get_total_items.return_value = 2
            vector_store.search.return_value = [
                {"metadata": vector_store.metadata[1], "distance": 0.99},
                {"metadata": vector_store.metadata[0], "distance": 0.8},
            ]

            searcher = Searcher(
                embedding=FakeEmbeddingService(dimension=8),
                time_parser=FakeTimeParser(),
                vector_store=vector_store,
                query_formatter=FakeQueryFormatter(
                    {
                        "像陶喆风格的舞台照": {
                            "search_text": "华语男歌手风格 舞台照",
                            "media_terms": ["stage_performance"],
                            "identity_terms": ["陶喆"],
                            "strict_identity_filter": False,
                            "time_hint": None,
                            "season": None,
                            "time_period": None,
                            "original_query": "像陶喆风格的舞台照",
                        }
                    }
                ),
            )

            results = searcher.search("像陶喆风格的舞台照", top_k=5)
            self.assertEqual([item["photo_path"] for item in results], [generic_path, exact_path])

    def test_search_uses_filter_only_branch_when_llm_returns_no_visual_semantics(self) -> None:
        vector_store = Mock()
        vector_store.load.return_value = True
        vector_store.dimension = 8
        vector_store.metric = "cosine"
        vector_store.metadata = []
        vector_store.get_total_items.return_value = 10

        keyword_store = Mock()
        keyword_store.search_with_filters.return_value = [
            {"photo_path": "/tmp/a.jpg", "score": 1.0},
        ]

        searcher = Searcher(
            embedding=FakeEmbeddingService(dimension=8),
            time_parser=FakeTimeParser(),
            vector_store=vector_store,
            keyword_store=keyword_store,
            query_formatter=FakeQueryFormatter(
                {
                    "去年的照片": {
                        "search_text": "",
                        "media_terms": [],
                        "identity_terms": [],
                        "time_hint": "去年",
                        "season": None,
                        "time_period": None,
                    }
                }
            ),
        )

        with patch.object(searcher, "_filter_only_search", return_value=[{"photo_path": "/tmp/a.jpg"}]) as filter_only, \
             patch.object(searcher.embedding_service, "generate_embedding") as generate_embedding:
            results = searcher.search("去年的照片", top_k=5)

        filter_only.assert_called_once_with(
            None,
            {"start_date": "2025-01-01", "end_date": "2025-12-31", "year": None, "month": None, "day": None, "season": None, "time_period": None, "precision": "year"},
            5,
        )
        generate_embedding.assert_not_called()
        self.assertEqual(results, [{"photo_path": "/tmp/a.jpg"}])

    def test_search_uses_hybrid_branch_for_non_filter_query_even_without_llm_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            photo_path = os.path.join(tmp, "IMG_20250425_204121.jpg")
            with open(photo_path, "wb") as file:
                file.write(b"test")

            vector_store = Mock()
            vector_store.load.return_value = True
            vector_store.dimension = 8
            vector_store.metric = "cosine"
            vector_store.metadata = [
                {"photo_path": photo_path, "description": "测试图片"},
            ]
            vector_store.get_total_items.return_value = 10

            searcher = Searcher(
                embedding=FakeEmbeddingService(dimension=8),
                time_parser=FakeTimeParser(),
                vector_store=vector_store,
                keyword_store=Mock(),
                query_formatter=FakeQueryFormatter(
                    {
                        "IMG_20250425_204121.jpg": {
                        "search_text": "",
                        "media_terms": [],
                        "identity_terms": [],
                        "time_hint": None,
                        "season": None,
                        "time_period": None,
                        }
                    }
                ),
            )

            expected_results = [
                {
                    "photo_path": photo_path,
                    "description": "测试图片",
                    "score": 0.91,
                    "vector_score": 0.91,
                    "keyword_score": 0.0,
                    "rank": 1,
                }
            ]

            with patch.object(searcher, "_hybrid_search", return_value=expected_results) as hybrid_search, \
                 patch.object(searcher.embedding_service, "generate_embedding", return_value=[0.1] * 8) as generate_embedding:
                results = searcher.search("IMG_20250425_204121.jpg", top_k=5)

            hybrid_search.assert_called_once()
            generate_embedding.assert_called_once_with("IMG_20250425_204121.jpg")
            self.assertEqual(results, expected_results)

    def test_search_uses_hybrid_branch_with_llm_extracted_visual_text(self) -> None:
        vector_store = Mock()
        vector_store.load.return_value = True
        vector_store.dimension = 8
        vector_store.metric = "cosine"
        vector_store.metadata = []
        vector_store.get_total_items.return_value = 10

        searcher = Searcher(
            embedding=FakeEmbeddingService(dimension=8),
            time_parser=FakeTimeParser(),
            vector_store=vector_store,
            keyword_store=Mock(),
            query_formatter=FakeQueryFormatter(
                {
                    "夏天海边的照片": {
                        "search_text": "海边",
                        "media_terms": [],
                        "identity_terms": [],
                        "time_hint": None,
                        "season": "夏天",
                        "time_period": None,
                    }
                }
            ),
        )

        expected_results = [
            {
                "photo_path": "/tmp/a.jpg",
                "description": "海边",
                "score": 0.9,
                "vector_score": 0.9,
                "keyword_score": 0.8,
                "rank": 1,
            }
        ]

        with patch.object(searcher, "_hybrid_search", return_value=expected_results) as hybrid_search, \
             patch.object(searcher.embedding_service, "generate_embedding", return_value=[0.1] * 8) as generate_embedding:
            results = searcher.search("夏天海边的照片", top_k=5)

        hybrid_search.assert_called_once()
        generate_embedding.assert_called_once_with("海边")
        self.assertEqual(results, expected_results)

    def test_search_supports_media_and_identity_terms_in_embedding_query(self) -> None:
        vector_store = Mock()
        vector_store.load.return_value = True
        vector_store.dimension = 8
        vector_store.metric = "cosine"
        vector_store.metadata = []
        vector_store.get_total_items.return_value = 10

        searcher = Searcher(
            embedding=FakeEmbeddingService(dimension=8),
            time_parser=FakeTimeParser(),
            vector_store=vector_store,
            keyword_store=Mock(),
            query_formatter=FakeQueryFormatter(
                {
                    "周杰伦演唱会": {
                        "search_text": "",
                        "media_terms": ["stage_performance"],
                        "identity_terms": ["周杰伦"],
                        "time_hint": None,
                        "season": None,
                        "time_period": None,
                    }
                }
            ),
        )

        with patch.object(searcher, "_hybrid_search", return_value=[]) as hybrid_search, patch.object(
            searcher.embedding_service,
            "generate_embedding",
            return_value=[0.1] * 8,
        ) as generate_embedding:
            searcher.search("周杰伦演唱会", top_k=5)

        generate_embedding.assert_called_once_with("stage_performance 周杰伦")
        _, kwargs = hybrid_search.call_args
        self.assertEqual(kwargs["media_terms"], ["stage_performance"])
        self.assertEqual(kwargs["identity_terms"], ["周杰伦"])

    def test_search_does_not_expand_when_first_round_is_strong(self) -> None:
        vector_store = Mock()
        vector_store.load.return_value = True
        vector_store.dimension = 8
        vector_store.metric = "cosine"
        vector_store.metadata = []
        vector_store.get_total_items.return_value = 10

        query_formatter = FakeQueryFormatter(
            {
                "海边日落": {
                    "search_text": "海边日落",
                    "media_terms": [],
                    "identity_terms": [],
                    "strict_identity_filter": False,
                    "time_hint": None,
                    "season": None,
                    "time_period": None,
                    "original_query": "海边日落",
                }
            }
        )

        searcher = Searcher(
            embedding=FakeEmbeddingService(dimension=8),
            time_parser=FakeTimeParser(),
            vector_store=vector_store,
            keyword_store=Mock(),
            query_formatter=query_formatter,
        )

        strong_results = [
            {"photo_path": "/tmp/a.jpg", "description": "海边日落", "score": 0.92, "vector_score": 0.92, "keyword_score": 0.0, "rank": 1},
        ]

        with patch.object(searcher, "_hybrid_search", return_value=strong_results) as hybrid_search, \
             patch.object(searcher.embedding_service, "generate_embedding", return_value=[0.1] * 8), \
             patch.object(searcher, "_maybe_expand_query_results", wraps=searcher._maybe_expand_query_results) as maybe_expand:
            results = searcher.search("海边日落", top_k=5)

        hybrid_search.assert_called_once()
        maybe_expand.assert_called_once()
        self.assertEqual(results, strong_results)

    def test_search_expands_when_first_round_is_weak(self) -> None:
        vector_store = Mock()
        vector_store.load.return_value = True
        vector_store.dimension = 8
        vector_store.metric = "cosine"
        vector_store.metadata = []
        vector_store.get_total_items.return_value = 10

        query_formatter = FakeQueryFormatter(
            {
                "请帮我找陶喆的照片": {
                    "search_text": "",
                    "media_terms": [],
                    "identity_terms": ["陶喆"],
                    "strict_identity_filter": True,
                    "time_hint": None,
                    "season": None,
                    "time_period": None,
                    "original_query": "请帮我找陶喆的照片",
                }
            }
        )
        query_formatter.expand_query_intents = Mock(return_value=[
            {
                "search_text": "舞台男歌手 现场演出",
                "media_terms": ["stage_performance"],
                "identity_terms": ["陶喆"],
                "strict_identity_filter": False,
                "time_hint": None,
                "season": None,
                "time_period": None,
                "original_query": "请帮我找陶喆的照片",
            }
        ])

        searcher = Searcher(
            embedding=FakeEmbeddingService(dimension=8),
            time_parser=FakeTimeParser(),
            vector_store=vector_store,
            keyword_store=Mock(),
            query_formatter=query_formatter,
        )

        weak_results = [
            {"photo_path": "/tmp/weak.jpg", "description": "模糊男歌手", "score": 0.41, "vector_score": 0.41, "keyword_score": 0.0, "rank": 1},
        ]
        expanded_results = [
            {"photo_path": "/tmp/better.jpg", "description": "舞台男歌手", "score": 0.78, "vector_score": 0.78, "keyword_score": 0.0, "rank": 1},
        ]

        with patch.object(searcher, "_hybrid_search", side_effect=[weak_results, expanded_results]) as hybrid_search, \
             patch.object(searcher.embedding_service, "generate_embedding", return_value=[0.1] * 8):
            results = searcher.search("请帮我找陶喆的照片", top_k=5)

        self.assertEqual(hybrid_search.call_count, 2)
        query_formatter.expand_query_intents.assert_called_once()
        self.assertEqual(results, expanded_results)

    def test_search_records_search_debug_for_expansion_round(self) -> None:
        vector_store = Mock()
        vector_store.load.return_value = True
        vector_store.dimension = 8
        vector_store.metric = "cosine"
        vector_store.metadata = []
        vector_store.get_total_items.return_value = 10

        query_formatter = FakeQueryFormatter(
            {
                "请帮我找陶喆的照片": {
                    "search_text": "",
                    "media_terms": [],
                    "identity_terms": ["陶喆"],
                    "strict_identity_filter": True,
                    "time_hint": None,
                    "season": None,
                    "time_period": None,
                    "original_query": "请帮我找陶喆的照片",
                }
            }
        )
        query_formatter.expansion_mapping["请帮我找陶喆的照片"] = [
            {
                "search_text": "舞台男歌手 现场演出",
                "media_terms": ["stage_performance"],
                "identity_terms": ["陶喆"],
                "strict_identity_filter": True,
                "time_hint": None,
                "season": None,
                "time_period": None,
                "original_query": "请帮我找陶喆的照片",
                "reason": "补充现场演出语义",
            }
        ]

        searcher = Searcher(
            embedding=FakeEmbeddingService(dimension=8),
            time_parser=FakeTimeParser(),
            vector_store=vector_store,
            keyword_store=Mock(),
            query_formatter=query_formatter,
        )

        weak_results = [
            {
                "photo_path": "/tmp/weak.jpg",
                "description": "模糊男歌手",
                "score": 0.41,
                "vector_score": 0.41,
                "keyword_score": 0.0,
                "rank": 1,
                "metadata": {"photo_path": "/tmp/weak.jpg", "identity_names": []},
            },
        ]
        expanded_results = [
            {
                "photo_path": "/tmp/better.jpg",
                "description": "陶喆舞台现场",
                "score": 0.78,
                "vector_score": 0.78,
                "keyword_score": 0.0,
                "rank": 1,
                "metadata": {"photo_path": "/tmp/better.jpg", "identity_names": ["陶喆"]},
            },
        ]

        with patch.object(searcher, "_hybrid_search", side_effect=[weak_results, expanded_results]), \
             patch.object(searcher.embedding_service, "generate_embedding", return_value=[0.1] * 8):
            results = searcher.search("请帮我找陶喆的照片", top_k=5)

        self.assertEqual(results, expanded_results)
        debug = searcher.get_last_search_debug()
        self.assertEqual(debug["base_intent"]["identity_terms"], ["陶喆"])
        self.assertTrue(debug["expansion_triggered"])
        self.assertFalse(debug["reflection_triggered"])
        self.assertEqual(len(debug["alternatives"]), 1)
        self.assertEqual(debug["alternatives"][0]["reason"], "补充现场演出语义")
        self.assertEqual(len(debug["rounds"]), 2)
        self.assertEqual(debug["rounds"][0]["round"], "base")
        self.assertEqual(debug["rounds"][1]["round"], "expansion")
        self.assertEqual(debug["rounds"][1]["result_count"], 1)

    def test_search_uses_reflection_when_expansion_is_still_weak(self) -> None:
        vector_store = Mock()
        vector_store.load.return_value = True
        vector_store.dimension = 8
        vector_store.metric = "cosine"
        vector_store.metadata = []
        vector_store.get_total_items.return_value = 10

        query_formatter = FakeQueryFormatter(
            {
                "请帮我找陶喆的照片": {
                    "search_text": "",
                    "media_terms": [],
                    "identity_terms": ["陶喆"],
                    "strict_identity_filter": True,
                    "time_hint": None,
                    "season": None,
                    "time_period": None,
                    "original_query": "请帮我找陶喆的照片",
                }
            }
        )
        query_formatter.expansion_mapping["请帮我找陶喆的照片"] = [
            {
                "search_text": "华语男歌手 舞台照",
                "media_terms": ["stage_performance"],
                "identity_terms": ["陶喆"],
                "strict_identity_filter": False,
                "time_hint": None,
                "season": None,
                "time_period": None,
                "original_query": "请帮我找陶喆的照片",
                "reason": "先扩大到舞台场景",
            }
        ]
        query_formatter.reflection_mapping["请帮我找陶喆的照片"] = {
            "search_text": "舞台近景 男歌手 特写",
            "media_terms": ["stage_performance"],
            "identity_terms": ["陶喆"],
            "strict_identity_filter": True,
            "time_hint": None,
            "season": None,
            "time_period": None,
            "original_query": "请帮我找陶喆的照片",
            "reason": "前两轮结果仍偏泛化，需要收紧到近景舞台人像",
        }

        searcher = Searcher(
            embedding=FakeEmbeddingService(dimension=8),
            time_parser=FakeTimeParser(),
            vector_store=vector_store,
            keyword_store=Mock(),
            query_formatter=query_formatter,
        )

        weak_results = [
            {
                "photo_path": "/tmp/weak.jpg",
                "description": "模糊男歌手",
                "score": 0.41,
                "vector_score": 0.41,
                "keyword_score": 0.0,
                "rank": 1,
                "metadata": {"photo_path": "/tmp/weak.jpg", "identity_names": []},
            },
        ]
        still_weak_results = [
            {
                "photo_path": "/tmp/still-weak.jpg",
                "description": "男歌手舞台远景",
                "score": 0.46,
                "vector_score": 0.46,
                "keyword_score": 0.0,
                "rank": 1,
                "metadata": {"photo_path": "/tmp/still-weak.jpg", "identity_names": []},
            },
        ]
        reflected_results = [
            {
                "photo_path": "/tmp/final.jpg",
                "description": "陶喆舞台近景",
                "score": 0.82,
                "vector_score": 0.82,
                "keyword_score": 0.0,
                "rank": 1,
                "metadata": {"photo_path": "/tmp/final.jpg", "identity_names": ["陶喆"]},
            },
        ]

        with patch.object(searcher, "_hybrid_search", side_effect=[weak_results, still_weak_results, reflected_results]), \
             patch.object(searcher.embedding_service, "generate_embedding", return_value=[0.1] * 8):
            results = searcher.search("请帮我找陶喆的照片", top_k=5)

        self.assertEqual(results, reflected_results)
        debug = searcher.get_last_search_debug()
        self.assertTrue(debug["expansion_triggered"])
        self.assertTrue(debug["reflection_triggered"])
        self.assertEqual(debug["reflection"]["reason"], "前两轮结果仍偏泛化，需要收紧到近景舞台人像")
        self.assertEqual(len(debug["rounds"]), 3)
        self.assertEqual(debug["rounds"][2]["round"], "reflection")
        self.assertEqual(debug["rounds"][2]["top_score"], 0.82)


if __name__ == "__main__":
    unittest.main()
