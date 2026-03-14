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

    def test_identity_query_prefers_matching_identity_evidence_but_can_backfill(self) -> None:
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
            self.assertEqual([item["photo_path"] for item in results], [exact_path, generic_path])

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
                        "search_text": "真人出镜 舞台现场 演唱会表演",
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

        generate_embedding.assert_called_once_with("真人出镜 舞台现场 演唱会表演 stage_performance")
        args, kwargs = hybrid_search.call_args
        self.assertEqual(args[0], "周杰伦演唱会")
        self.assertEqual(kwargs["media_terms"], ["stage_performance"])
        self.assertEqual(kwargs["identity_terms"], ["周杰伦"])

    def test_finalize_results_skips_label_prefilter_when_query_has_visual_grounding(self) -> None:
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
        )

        combined_results = [
            {
                "photo_path": "/tmp/live.jpg",
                "description": "现场照片",
                "score": 0.86,
                "vector_score": 0.86,
                "keyword_score": 0.0,
                "rank": 0,
                "metadata": {
                    "photo_path": "/tmp/live.jpg",
                    "identity_names": [],
                    "media_types": ["stage_performance"],
                },
            },
            {
                "photo_path": "/tmp/screen.jpg",
                "description": "带名字的截图",
                "score": 0.78,
                "vector_score": 0.78,
                "keyword_score": 0.0,
                "rank": 0,
                "metadata": {
                    "photo_path": "/tmp/screen.jpg",
                    "identity_names": ["河南说唱之神"],
                    "media_types": ["screen"],
                },
            },
        ]

        with patch.object(searcher, "_calculate_dynamic_threshold", return_value=0.1):
            results = searcher._finalize_results(
                combined_results=combined_results,
                normalized_top_k=2,
                has_filter=False,
                constraints={},
                search_text="真人出镜 舞台现场 说唱演出",
                media_terms=["stage_performance"],
                identity_terms=["河南说唱之神"],
                strict_identity_filter=True,
            )

        self.assertEqual(
            [item["photo_path"] for item in results],
            ["/tmp/live.jpg", "/tmp/screen.jpg"],
        )

    def test_finalize_results_prioritizes_media_matches_for_album_queries(self) -> None:
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
        )

        combined_results = [
            {
                "photo_path": "/tmp/screen.jpg",
                "description": "带歌名的截图",
                "score": 0.86,
                "vector_score": 0.86,
                "keyword_score": 0.0,
                "rank": 0,
                "metadata": {
                    "photo_path": "/tmp/screen.jpg",
                    "media_types": ["screen"],
                },
            },
            {
                "photo_path": "/tmp/cover.jpg",
                "description": "专辑封面",
                "score": 0.79,
                "vector_score": 0.79,
                "keyword_score": 0.0,
                "rank": 0,
                "metadata": {
                    "photo_path": "/tmp/cover.jpg",
                    "media_types": ["album_cover"],
                },
            },
        ]

        with patch.object(searcher, "_calculate_dynamic_threshold", return_value=0.4):
            results = searcher._finalize_results(
                combined_results=combined_results,
                normalized_top_k=2,
                has_filter=False,
                constraints={},
                search_text="专辑封面 艺人海报",
                media_terms=["album_cover"],
                identity_terms=[],
                strict_identity_filter=False,
            )

        self.assertEqual(
            [item["photo_path"] for item in results],
            ["/tmp/cover.jpg", "/tmp/screen.jpg"],
        )

    def test_search_uses_filter_only_mode_from_query_formatter(self) -> None:
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
                        "retrieval_mode": "filter_only",
                        "media_terms": [],
                        "identity_terms": [],
                        "strict_identity_filter": False,
                        "time_hint": "去年",
                        "season": None,
                        "time_period": None,
                        "original_query": "去年的照片",
                    }
                }
            ),
        )

        with patch.object(searcher, "_filter_only_search", return_value=[]) as filter_only_search, \
             patch.object(searcher.embedding_service, "generate_embedding") as generate_embedding:
            searcher.search("去年的照片", top_k=5)

        filter_only_search.assert_called_once()
        generate_embedding.assert_not_called()

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

    def test_search_continues_expansion_when_results_do_not_fill_top_k(self) -> None:
        vector_store = Mock()
        vector_store.load.return_value = True
        vector_store.dimension = 8
        vector_store.metric = "cosine"
        vector_store.metadata = []
        vector_store.get_total_items.return_value = 10

        query_formatter = FakeQueryFormatter(
            {
                "帮我找河南说唱之神的照片": {
                    "search_text": "",
                    "media_terms": [],
                    "identity_terms": ["河南说唱之神"],
                    "strict_identity_filter": True,
                    "intent_mode": "strict",
                    "intent_contract": {
                        "core_target": "河南说唱之神的照片",
                        "must_keep": ["河南说唱之神"],
                        "avoid_drift": "不要漂移到只出现名字的截图",
                    },
                    "time_hint": None,
                    "season": None,
                    "time_period": None,
                    "original_query": "帮我找河南说唱之神的照片",
                }
            }
        )
        query_formatter.expand_query_intents = Mock(return_value=[
            {
                "search_text": "演出现场 舞台 说唱歌手",
                "media_terms": ["stage_performance"],
                "identity_terms": ["河南说唱之神"],
                "strict_identity_filter": True,
                "intent_mode": "strict",
                "intent_contract": {
                    "core_target": "河南说唱之神的照片",
                    "must_keep": ["河南说唱之神"],
                    "avoid_drift": "不要漂移到只出现名字的截图",
                },
                "contract_satisfied": True,
                "reason": "补充演出现场视觉表达",
            }
        ])

        searcher = Searcher(
            embedding=FakeEmbeddingService(dimension=8),
            time_parser=FakeTimeParser(),
            vector_store=vector_store,
            keyword_store=Mock(),
            query_formatter=query_formatter,
        )

        first_round_results = [
            {
                "photo_path": "/tmp/screenshot.jpg",
                "description": "网易云截图",
                "score": 0.91,
                "vector_score": 0.91,
                "keyword_score": 0.0,
                "rank": 1,
                "metadata": {"photo_path": "/tmp/screenshot.jpg", "identity_names": ["河南说唱之神"]},
            }
        ]
        expanded_results = [
            {
                "photo_path": "/tmp/photo.jpg",
                "description": "河南说唱之神演出现场",
                "score": 0.86,
                "vector_score": 0.86,
                "keyword_score": 0.0,
                "rank": 1,
                "metadata": {"photo_path": "/tmp/photo.jpg", "identity_names": ["河南说唱之神"]},
            }
        ]

        with patch.object(searcher, "_hybrid_search", side_effect=[first_round_results, expanded_results]) as hybrid_search, \
             patch.object(searcher.embedding_service, "generate_embedding", return_value=[0.1] * 8):
            results = searcher.search("帮我找河南说唱之神的照片", top_k=5)

        self.assertEqual(hybrid_search.call_count, 2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["photo_path"], "/tmp/screenshot.jpg")
        self.assertEqual(results[1]["photo_path"], "/tmp/photo.jpg")

    def test_search_expands_when_top_k_is_filled_by_weak_backfill(self) -> None:
        vector_store = Mock()
        vector_store.load.return_value = True
        vector_store.dimension = 8
        vector_store.metric = "cosine"
        vector_store.metadata = []
        vector_store.get_total_items.return_value = 30

        query_formatter = FakeQueryFormatter(
            {
                "河南说唱之神专辑封面": {
                    "search_text": "说唱歌手 专辑封面",
                    "media_terms": ["album_cover"],
                    "identity_terms": ["河南说唱之神"],
                    "strict_identity_filter": False,
                    "intent_mode": "open",
                    "intent_contract": {
                        "core_target": "河南说唱之神专辑封面",
                        "must_keep": ["专辑封面"],
                        "avoid_drift": "不要扩成普通自拍或截图",
                    },
                    "time_hint": None,
                    "season": None,
                    "time_period": None,
                    "original_query": "河南说唱之神专辑封面",
                }
            }
        )
        query_formatter.expand_query_intents = Mock(return_value=[
            {
                "search_text": "唱片封面 艺人海报 方形封套",
                "media_terms": ["album_cover"],
                "identity_terms": ["河南说唱之神"],
                "strict_identity_filter": False,
                "intent_mode": "open",
                "intent_contract": {
                    "core_target": "河南说唱之神专辑封面",
                    "must_keep": ["专辑封面"],
                    "avoid_drift": "不要扩成普通自拍或截图",
                },
                "contract_satisfied": True,
                "reason": "第一轮后半段结果偏弱，补充更直接的封面载体表达",
            }
        ])

        searcher = Searcher(
            embedding=FakeEmbeddingService(dimension=8),
            time_parser=FakeTimeParser(),
            vector_store=vector_store,
            keyword_store=Mock(),
            query_formatter=query_formatter,
        )

        first_round_results = [
            {
                "photo_path": "/tmp/cover-a.jpg",
                "description": "专辑封面 A",
                "score": 0.93,
                "vector_score": 0.93,
                "keyword_score": 0.0,
                "rank": 0,
                "metadata": {"photo_path": "/tmp/cover-a.jpg", "media_types": ["album_cover"]},
            },
            {
                "photo_path": "/tmp/cover-b.jpg",
                "description": "专辑封面 B",
                "score": 0.81,
                "vector_score": 0.81,
                "keyword_score": 0.0,
                "rank": 0,
                "metadata": {"photo_path": "/tmp/cover-b.jpg", "media_types": ["album_cover"]},
            },
            {
                "photo_path": "/tmp/screen-a.jpg",
                "description": "截图 A",
                "score": 0.38,
                "vector_score": 0.38,
                "keyword_score": 0.0,
                "rank": 0,
                "metadata": {"photo_path": "/tmp/screen-a.jpg", "media_types": ["screen"]},
            },
            {
                "photo_path": "/tmp/screen-b.jpg",
                "description": "截图 B",
                "score": 0.34,
                "vector_score": 0.34,
                "keyword_score": 0.0,
                "rank": 0,
                "metadata": {"photo_path": "/tmp/screen-b.jpg", "media_types": ["screen"]},
            },
        ]
        expanded_results = [
            {
                "photo_path": "/tmp/cover-c.jpg",
                "description": "专辑封面 C",
                "score": 0.77,
                "vector_score": 0.77,
                "keyword_score": 0.0,
                "rank": 0,
                "metadata": {"photo_path": "/tmp/cover-c.jpg", "media_types": ["album_cover"]},
            },
            {
                "photo_path": "/tmp/cover-d.jpg",
                "description": "专辑封面 D",
                "score": 0.73,
                "vector_score": 0.73,
                "keyword_score": 0.0,
                "rank": 0,
                "metadata": {"photo_path": "/tmp/cover-d.jpg", "media_types": ["album_cover"]},
            },
        ]

        with patch.object(searcher, "_hybrid_search", side_effect=[first_round_results, expanded_results]), \
             patch.object(searcher.embedding_service, "generate_embedding", return_value=[0.1] * 8):
            results = searcher.search("河南说唱之神专辑封面", top_k=4)

        query_formatter.expand_query_intents.assert_called_once()
        self.assertEqual(
            [item["photo_path"] for item in results],
            ["/tmp/cover-a.jpg", "/tmp/cover-b.jpg", "/tmp/cover-c.jpg", "/tmp/cover-d.jpg"],
        )
        debug = searcher.get_last_search_debug()
        self.assertTrue(debug["expansion_triggered"])
        self.assertFalse(debug["reflection_triggered"])

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
                    "intent_mode": "strict",
                    "intent_contract": {
                        "core_target": "陶喆的照片",
                        "must_keep": ["陶喆"],
                        "avoid_drift": "不要扩成其他男歌手",
                    },
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
                "strict_identity_filter": True,
                "intent_mode": "strict",
                "intent_contract": {
                    "core_target": "陶喆的照片",
                    "must_keep": ["陶喆"],
                    "avoid_drift": "不要扩成其他男歌手",
                },
                "contract_satisfied": True,
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
        expanded_round_results = [
            {
                "photo_path": "/tmp/better.jpg",
                "description": "舞台男歌手",
                "score": 0.78,
                "vector_score": 0.78,
                "keyword_score": 0.0,
                "rank": 1,
                "metadata": {"photo_path": "/tmp/better.jpg", "identity_names": ["陶喆"]},
            },
        ]
        with patch.object(searcher, "_hybrid_search", side_effect=[weak_results, expanded_round_results]) as hybrid_search, \
             patch.object(searcher.embedding_service, "generate_embedding", return_value=[0.1] * 8):
            results = searcher.search("请帮我找陶喆的照片", top_k=5)

        self.assertEqual(hybrid_search.call_count, 2)
        query_formatter.expand_query_intents.assert_called_once()
        self.assertEqual(results[0]["photo_path"], "/tmp/better.jpg")
        self.assertEqual(results[1]["photo_path"], "/tmp/weak.jpg")

    def test_search_skips_expansion_that_breaks_strict_contract_for_non_person_query(self) -> None:
        vector_store = Mock()
        vector_store.load.return_value = True
        vector_store.dimension = 8
        vector_store.metric = "cosine"
        vector_store.metadata = []
        vector_store.get_total_items.return_value = 10

        query_formatter = FakeQueryFormatter(
            {
                "频谱分析仪 屏幕": {
                    "search_text": "频谱分析仪 屏幕",
                    "media_terms": ["screen"],
                    "identity_terms": [],
                    "strict_identity_filter": False,
                    "intent_mode": "strict",
                    "intent_contract": {
                        "core_target": "频谱分析仪屏幕",
                        "must_keep": ["频谱分析仪", "屏幕"],
                        "avoid_drift": "不要扩成一般设备屏幕",
                    },
                    "time_hint": None,
                    "season": None,
                    "time_period": None,
                    "original_query": "频谱分析仪 屏幕",
                }
            }
        )
        query_formatter.expand_query_intents = Mock(return_value=[
            {
                "search_text": "设备显示屏 曲线图",
                "media_terms": ["screen"],
                "identity_terms": [],
                "strict_identity_filter": False,
                "intent_mode": "open",
                "intent_contract": {
                    "core_target": "设备显示屏",
                    "must_keep": ["设备", "屏幕"],
                    "avoid_drift": "泛化设备",
                },
                "contract_satisfied": False,
                "reason": "泛化成更常见的设备屏幕",
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
            {"photo_path": "/tmp/base.jpg", "description": "频谱分析仪屏幕", "score": 0.42, "vector_score": 0.42, "keyword_score": 0.0, "rank": 1},
        ]

        with patch.object(searcher, "_hybrid_search", return_value=weak_results) as hybrid_search, \
             patch.object(searcher.embedding_service, "generate_embedding", return_value=[0.1] * 8):
            results = searcher.search("频谱分析仪 屏幕", top_k=5)

        self.assertEqual(hybrid_search.call_count, 1)
        self.assertEqual(results, weak_results)

    def test_search_allows_open_query_expansion(self) -> None:
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
                    "intent_mode": "open",
                    "intent_contract": {
                        "core_target": "海边日落",
                        "must_keep": ["海边"],
                        "avoid_drift": "",
                    },
                    "time_hint": None,
                    "season": None,
                    "time_period": None,
                    "original_query": "海边日落",
                }
            }
        )
        query_formatter.expand_query_intents = Mock(return_value=[
            {
                "search_text": "海边 傍晚 落日",
                "media_terms": [],
                "identity_terms": [],
                "strict_identity_filter": False,
                "intent_mode": "open",
                "intent_contract": {
                    "core_target": "海边日落",
                    "must_keep": ["海边"],
                    "avoid_drift": "",
                },
                "contract_satisfied": True,
                "reason": "补充常见视觉表达",
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
            {"photo_path": "/tmp/weak.jpg", "description": "海边", "score": 0.41, "vector_score": 0.41, "keyword_score": 0.0, "rank": 1},
        ]
        expanded_results = [
            {"photo_path": "/tmp/better.jpg", "description": "海边日落", "score": 0.74, "vector_score": 0.74, "keyword_score": 0.0, "rank": 1},
        ]

        with patch.object(searcher, "_hybrid_search", side_effect=[weak_results, expanded_results]) as hybrid_search, \
             patch.object(searcher.embedding_service, "generate_embedding", return_value=[0.1] * 8):
            results = searcher.search("海边日落", top_k=5)

        self.assertEqual(hybrid_search.call_count, 2)
        self.assertEqual(results[0], expanded_results[0])
        self.assertEqual(results[1], weak_results[0] | {"rank": 2})

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
                    "intent_mode": "strict",
                    "intent_contract": {
                        "core_target": "陶喆的照片",
                        "must_keep": ["陶喆"],
                        "avoid_drift": "不要扩成其他男歌手",
                    },
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
                "intent_mode": "strict",
                "intent_contract": {
                    "core_target": "陶喆的照片",
                    "must_keep": ["陶喆"],
                    "avoid_drift": "不要扩成其他男歌手",
                },
                "contract_satisfied": True,
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

        self.assertEqual(results[0]["photo_path"], "/tmp/better.jpg")
        self.assertEqual(results[1]["photo_path"], "/tmp/weak.jpg")
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
                    "intent_mode": "strict",
                    "intent_contract": {
                        "core_target": "陶喆的照片",
                        "must_keep": ["陶喆"],
                        "avoid_drift": "不要扩成其他男歌手",
                    },
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
                "strict_identity_filter": True,
                "intent_mode": "strict",
                "intent_contract": {
                    "core_target": "陶喆的照片",
                    "must_keep": ["陶喆"],
                    "avoid_drift": "不要扩成其他男歌手",
                },
                "contract_satisfied": True,
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
            "intent_mode": "strict",
            "intent_contract": {
                "core_target": "陶喆的照片",
                "must_keep": ["陶喆"],
                "avoid_drift": "不要扩成其他男歌手",
            },
            "contract_satisfied": True,
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

        self.assertEqual(results[0]["photo_path"], "/tmp/final.jpg")
        self.assertEqual(len(results), 3)
        debug = searcher.get_last_search_debug()
        self.assertTrue(debug["expansion_triggered"])
        self.assertTrue(debug["reflection_triggered"])
        self.assertEqual(debug["reflection"]["reason"], "前两轮结果仍偏泛化，需要收紧到近景舞台人像")
        self.assertEqual(len(debug["rounds"]), 3)
        self.assertEqual(debug["rounds"][2]["round"], "reflection")
        self.assertEqual(debug["rounds"][2]["top_score"], 0.82)

    def test_search_uses_reflection_when_expansion_result_count_is_still_below_top_k(self) -> None:
        vector_store = Mock()
        vector_store.load.return_value = True
        vector_store.dimension = 8
        vector_store.metric = "cosine"
        vector_store.metadata = []
        vector_store.get_total_items.return_value = 10

        query_formatter = FakeQueryFormatter(
            {
                "河南说唱之神": {
                    "search_text": "",
                    "media_terms": [],
                    "identity_terms": ["河南说唱之神"],
                    "strict_identity_filter": True,
                    "intent_mode": "strict",
                    "intent_contract": {
                        "core_target": "河南说唱之神的照片",
                        "must_keep": ["河南说唱之神"],
                        "avoid_drift": "不要漂移到截图或其他人物",
                    },
                    "time_hint": None,
                    "season": None,
                    "time_period": None,
                    "original_query": "河南说唱之神",
                }
            }
        )
        query_formatter.expansion_mapping["河南说唱之神"] = [
            {
                "search_text": "舞台现场 说唱歌手 演出照",
                "media_terms": ["stage_performance"],
                "identity_terms": ["河南说唱之神"],
                "strict_identity_filter": True,
                "intent_mode": "strict",
                "intent_contract": {
                    "core_target": "河南说唱之神的照片",
                    "must_keep": ["河南说唱之神"],
                    "avoid_drift": "不要漂移到截图或其他人物",
                },
                "contract_satisfied": True,
                "time_hint": None,
                "season": None,
                "time_period": None,
                "original_query": "河南说唱之神",
                "reason": "补充演出现场语义",
            }
        ]
        query_formatter.reflection_mapping["河南说唱之神"] = {
            "search_text": "舞台近景 现场表演 人像特写",
            "media_terms": ["stage_performance"],
            "identity_terms": ["河南说唱之神"],
            "strict_identity_filter": True,
            "intent_mode": "strict",
            "intent_contract": {
                "core_target": "河南说唱之神的照片",
                "must_keep": ["河南说唱之神"],
                "avoid_drift": "不要漂移到截图或其他人物",
            },
            "contract_satisfied": True,
            "time_hint": None,
            "season": None,
            "time_period": None,
            "original_query": "河南说唱之神",
            "reason": "结果数量仍不足，补充更直接的人像现场语义",
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
                "photo_path": "/tmp/netease.jpg",
                "description": "网易云截图",
                "score": 0.61,
                "vector_score": 0.61,
                "keyword_score": 0.0,
                "rank": 1,
                "metadata": {"photo_path": "/tmp/netease.jpg", "identity_names": []},
            },
        ]
        still_sparse_results = [
            {
                "photo_path": "/tmp/show-1.jpg",
                "description": "现场照片 1",
                "score": 0.74,
                "vector_score": 0.74,
                "keyword_score": 0.0,
                "rank": 1,
                "metadata": {"photo_path": "/tmp/show-1.jpg", "identity_names": ["河南说唱之神"]},
            },
        ]
        reflected_results = [
            {
                "photo_path": "/tmp/show-2.jpg",
                "description": "现场照片 2",
                "score": 0.72,
                "vector_score": 0.72,
                "keyword_score": 0.0,
                "rank": 1,
                "metadata": {"photo_path": "/tmp/show-2.jpg", "identity_names": ["河南说唱之神"]},
            },
            {
                "photo_path": "/tmp/show-3.jpg",
                "description": "现场照片 3",
                "score": 0.69,
                "vector_score": 0.69,
                "keyword_score": 0.0,
                "rank": 2,
                "metadata": {"photo_path": "/tmp/show-3.jpg", "identity_names": ["河南说唱之神"]},
            },
        ]

        with patch.object(searcher, "_hybrid_search", side_effect=[weak_results, still_sparse_results, reflected_results]), \
             patch.object(searcher.embedding_service, "generate_embedding", return_value=[0.1] * 8):
            results = searcher.search("河南说唱之神", top_k=4)

        self.assertEqual(len(results), 4)
        self.assertEqual(
            [item["photo_path"] for item in results],
            ["/tmp/show-1.jpg", "/tmp/show-2.jpg", "/tmp/show-3.jpg", "/tmp/netease.jpg"],
        )
        debug = searcher.get_last_search_debug()
        self.assertTrue(debug["expansion_triggered"])
        self.assertTrue(debug["reflection_triggered"])

    def test_search_continues_reflection_rounds_until_top_k_is_filled(self) -> None:
        vector_store = Mock()
        vector_store.load.return_value = True
        vector_store.dimension = 8
        vector_store.metric = "cosine"
        vector_store.metadata = []
        vector_store.get_total_items.return_value = 20

        query_formatter = FakeQueryFormatter(
            {
                "河南说唱之神演出照片": {
                    "search_text": "说唱歌手 舞台演出",
                    "media_terms": ["stage_performance"],
                    "identity_terms": ["河南说唱之神"],
                    "strict_identity_filter": False,
                    "intent_mode": "open",
                    "intent_contract": {
                        "core_target": "河南说唱之神演出照片",
                        "must_keep": ["演出"],
                        "avoid_drift": "不要扩成生活照或截图",
                    },
                    "time_hint": None,
                    "season": None,
                    "time_period": None,
                    "original_query": "河南说唱之神演出照片",
                }
            }
        )
        query_formatter.expand_query_intents = Mock(return_value=[])
        query_formatter.reflect_on_weak_results = Mock(side_effect=[
            {
                "search_text": "舞台近景 现场表演",
                "media_terms": ["stage_performance"],
                "identity_terms": ["河南说唱之神"],
                "strict_identity_filter": False,
                "intent_mode": "open",
                "intent_contract": {
                    "core_target": "河南说唱之神演出照片",
                    "must_keep": ["演出"],
                    "avoid_drift": "不要扩成生活照或截图",
                },
                "contract_satisfied": True,
                "reason": "第一轮结果数量不足，补充近景现场表达",
            },
            {
                "search_text": "舞台特写 麦克风 现场人像",
                "media_terms": ["stage_performance"],
                "identity_terms": ["河南说唱之神"],
                "strict_identity_filter": False,
                "intent_mode": "open",
                "intent_contract": {
                    "core_target": "河南说唱之神演出照片",
                    "must_keep": ["演出"],
                    "avoid_drift": "不要扩成生活照或截图",
                },
                "contract_satisfied": True,
                "reason": "继续补充更直接的人像舞台特写",
            },
            {},
        ])

        searcher = Searcher(
            embedding=FakeEmbeddingService(dimension=8),
            time_parser=FakeTimeParser(),
            vector_store=vector_store,
            keyword_store=Mock(),
            query_formatter=query_formatter,
        )

        base_results = [
            {
                "photo_path": "/tmp/show-a.jpg",
                "description": "演出照片 A",
                "score": 0.84,
                "vector_score": 0.84,
                "keyword_score": 0.0,
                "rank": 0,
                "metadata": {"photo_path": "/tmp/show-a.jpg", "media_types": ["stage_performance"]},
            },
            {
                "photo_path": "/tmp/screen.jpg",
                "description": "截图",
                "score": 0.31,
                "vector_score": 0.31,
                "keyword_score": 0.0,
                "rank": 0,
                "metadata": {"photo_path": "/tmp/screen.jpg", "media_types": ["screen"]},
            },
        ]
        reflected_round_one = [
            {
                "photo_path": "/tmp/show-b.jpg",
                "description": "演出照片 B",
                "score": 0.76,
                "vector_score": 0.76,
                "keyword_score": 0.0,
                "rank": 0,
                "metadata": {"photo_path": "/tmp/show-b.jpg", "media_types": ["stage_performance"]},
            },
        ]
        reflected_round_two = [
            {
                "photo_path": "/tmp/show-c.jpg",
                "description": "演出照片 C",
                "score": 0.71,
                "vector_score": 0.71,
                "keyword_score": 0.0,
                "rank": 0,
                "metadata": {"photo_path": "/tmp/show-c.jpg", "media_types": ["stage_performance"]},
            },
            {
                "photo_path": "/tmp/show-d.jpg",
                "description": "演出照片 D",
                "score": 0.66,
                "vector_score": 0.66,
                "keyword_score": 0.0,
                "rank": 0,
                "metadata": {"photo_path": "/tmp/show-d.jpg", "media_types": ["stage_performance"]},
            },
        ]

        with patch.object(searcher, "_hybrid_search", side_effect=[base_results, reflected_round_one, reflected_round_two]), \
             patch.object(searcher.embedding_service, "generate_embedding", return_value=[0.1] * 8):
            results = searcher.search("河南说唱之神演出照片", top_k=4)

        self.assertEqual(
            [item["photo_path"] for item in results],
            ["/tmp/show-a.jpg", "/tmp/show-b.jpg", "/tmp/show-c.jpg", "/tmp/show-d.jpg"],
        )
        self.assertGreaterEqual(query_formatter.reflect_on_weak_results.call_count, 2)
        debug = searcher.get_last_search_debug()
        self.assertTrue(debug["reflection_triggered"])
        self.assertEqual(debug["rounds"][-1]["round"], "reflection")

    def test_reflection_merge_does_not_replace_existing_results_when_reflected_round_is_still_sparse(self) -> None:
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
            query_formatter=FakeQueryFormatter(),
        )

        current_results = [
            {
                "photo_path": "/tmp/a.jpg",
                "description": "a",
                "score": 0.77,
                "rank": 1,
                "metadata": {"photo_path": "/tmp/a.jpg"},
            },
            {
                "photo_path": "/tmp/b.jpg",
                "description": "b",
                "score": 0.71,
                "rank": 2,
                "metadata": {"photo_path": "/tmp/b.jpg"},
            },
        ]
        reflected_results = [
            {
                "photo_path": "/tmp/c.jpg",
                "description": "c",
                "score": 0.74,
                "rank": 1,
                "metadata": {"photo_path": "/tmp/c.jpg"},
            },
        ]

        searcher.query_formatter.reflection_mapping["test query"] = {
            "search_text": "refined",
            "media_terms": [],
            "identity_terms": [],
            "strict_identity_filter": False,
            "intent_mode": "open",
            "contract_satisfied": True,
            "reason": "数量不足",
        }
        debug = searcher._empty_search_debug()
        with patch.object(searcher, "_run_single_search_round", return_value=reflected_results):
            results = searcher._maybe_reflect_query_results(
                query="test query",
                base_intent={"intent_mode": "open"},
                current_results=current_results,
                normalized_top_k=4,
                constraints={},
                has_filter=False,
                debug=debug,
            )

        self.assertEqual(
            [item["photo_path"] for item in results],
            ["/tmp/a.jpg", "/tmp/c.jpg", "/tmp/b.jpg"],
        )

    def test_finalize_results_fills_to_top_k_after_threshold_filtering(self) -> None:
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
        )

        combined_results = [
            {
                "photo_path": "/tmp/a.jpg",
                "description": "a",
                "score": 0.95,
                "vector_score": 0.95,
                "keyword_score": 0.0,
                "rank": 0,
                "metadata": {"photo_path": "/tmp/a.jpg"},
            },
            {
                "photo_path": "/tmp/b.jpg",
                "description": "b",
                "score": 0.76,
                "vector_score": 0.76,
                "keyword_score": 0.0,
                "rank": 0,
                "metadata": {"photo_path": "/tmp/b.jpg"},
            },
            {
                "photo_path": "/tmp/c.jpg",
                "description": "c",
                "score": 0.71,
                "vector_score": 0.71,
                "keyword_score": 0.0,
                "rank": 0,
                "metadata": {"photo_path": "/tmp/c.jpg"},
            },
            {
                "photo_path": "/tmp/d.jpg",
                "description": "d",
                "score": 0.69,
                "vector_score": 0.69,
                "keyword_score": 0.0,
                "rank": 0,
                "metadata": {"photo_path": "/tmp/d.jpg"},
            },
        ]

        with patch.object(searcher, "_calculate_dynamic_threshold", return_value=0.75):
            results = searcher._finalize_results(
                combined_results=combined_results,
                normalized_top_k=4,
                has_filter=False,
                constraints={},
            )

        self.assertEqual(len(results), 4)
        self.assertEqual([item["photo_path"] for item in results], [
            "/tmp/a.jpg",
            "/tmp/b.jpg",
            "/tmp/c.jpg",
            "/tmp/d.jpg",
        ])
        self.assertEqual([item["rank"] for item in results], [1, 2, 3, 4])

    def test_finalize_results_prefers_scores_above_point_four_but_backfills_when_needed(self) -> None:
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
        )

        combined_results = [
            {
                "photo_path": "/tmp/a.jpg",
                "description": "a",
                "score": 0.88,
                "vector_score": 0.88,
                "keyword_score": 0.0,
                "rank": 0,
                "metadata": {"photo_path": "/tmp/a.jpg"},
            },
            {
                "photo_path": "/tmp/b.jpg",
                "description": "b",
                "score": 0.52,
                "vector_score": 0.52,
                "keyword_score": 0.0,
                "rank": 0,
                "metadata": {"photo_path": "/tmp/b.jpg"},
            },
            {
                "photo_path": "/tmp/c.jpg",
                "description": "c",
                "score": 0.39,
                "vector_score": 0.39,
                "keyword_score": 0.0,
                "rank": 0,
                "metadata": {"photo_path": "/tmp/c.jpg"},
            },
            {
                "photo_path": "/tmp/d.jpg",
                "description": "d",
                "score": 0.22,
                "vector_score": 0.22,
                "keyword_score": 0.0,
                "rank": 0,
                "metadata": {"photo_path": "/tmp/d.jpg"},
            },
        ]

        with patch.object(searcher, "_calculate_dynamic_threshold", return_value=0.1):
            results = searcher._finalize_results(
                combined_results=combined_results,
                normalized_top_k=3,
                has_filter=False,
                constraints={},
            )

        self.assertEqual([item["photo_path"] for item in results], [
            "/tmp/a.jpg",
            "/tmp/b.jpg",
            "/tmp/c.jpg",
        ])
        self.assertEqual([item["rank"] for item in results], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
