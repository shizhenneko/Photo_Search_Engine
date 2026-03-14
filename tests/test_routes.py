import os
import sys
import tempfile
import time
import unittest
import json
from typing import Any
from pathlib import Path
from io import BytesIO
from unittest.mock import patch

from PIL import Image

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from api.routes import _apply_rerank_pipeline, _calculate_rerank_search_pool_k, register_routes
from core.indexer import Indexer
from core.searcher import Searcher
from tests.helpers import (
    FakeEmbeddingService,
    FakeQueryFormatter,
    FakeTextRerankService,
    FakeTimeParser,
    FakeVisualRerankService,
)
from utils.vector_store import VectorStore
from utils.vision_llm_service import LocalVisionLLMService


def _create_image(path: str, size: tuple[int, int] = (64, 48)) -> None:
    image = Image.new("RGB", size, color=(255, 0, 0))
    image.save(path)


def _create_test_app():
    from flask import Flask

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.secret_key = "test-secret-key"

    tmp = tempfile.mkdtemp()
    photo_dir = os.path.join(tmp, "photos")
    data_dir = os.path.join(tmp, "data")
    os.makedirs(photo_dir)
    os.makedirs(data_dir)

    index_path = os.path.join(data_dir, "photo_search.index")
    metadata_path = os.path.join(data_dir, "metadata.json")
    vector_store = VectorStore(dimension=8, index_path=index_path, metadata_path=metadata_path)

    vision = LocalVisionLLMService()
    embedding = FakeEmbeddingService(dimension=8)
    time_parser = FakeTimeParser()
    query_formatter = FakeQueryFormatter()

    indexer = Indexer(
        photo_dir=photo_dir,
        vision=vision,
        embedding=embedding,
        vector_store=vector_store,
        data_dir=data_dir,
    )
    searcher = Searcher(
        embedding=embedding,
        time_parser=time_parser,
        vector_store=vector_store,
        query_formatter=query_formatter,
        data_dir=data_dir,
    )

    config = {
        "PHOTO_DIR": photo_dir,
        "DATA_DIR": data_dir,
        "TOP_K": 5,
    }

    register_routes(
        app,
        indexer,
        searcher,
        config,
        text_rerank_service=FakeTextRerankService(),
        visual_rerank_service=FakeVisualRerankService(),
    )

    return app, tmp, indexer, searcher


class RouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app, self.tmpdir, self.indexer, self.searcher = _create_test_app()
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        import shutil

        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def _index_photos(self, count: int = 3) -> list[str]:
        photo_dir = os.path.join(self.tmpdir, "photos")
        paths = []
        for index in range(count):
            path = os.path.join(photo_dir, f"photo_{index}.jpg")
            _create_image(path)
            paths.append(path)
        response = self.client.post("/init_index", json={})
        self.assertEqual(response.status_code, 200)
        for _ in range(100):
            status = self.client.get("/index_status").get_json()
            if status["status"] in {"success", "ready"}:
                break
            time.sleep(0.05)
        else:
            self.fail("index build did not finish in time")
        self.searcher.load_index()
        return paths

    def test_index_page_renders(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("搜图", html)
        self.assertIn("准备好就能搜", html)
        self.assertIn("文本搜图", html)
        self.assertIn("以图搜图", html)
        self.assertIn("增量索引", html)
        self.assertIn("全量重建", html)

    def test_init_index_defaults_to_incremental_mode(self) -> None:
        with patch.object(self.indexer, "start_build_in_background", return_value={"status": "processing"}) as start_build:
            response = self.client.post("/init_index", json={})
        self.assertEqual(response.status_code, 200)
        start_build.assert_called_once_with(force_rebuild=False)

    def test_init_index_supports_full_rebuild_mode(self) -> None:
        with patch.object(self.indexer, "start_build_in_background", return_value={"status": "processing"}) as start_build:
            response = self.client.post("/init_index", json={"mode": "full"})
        self.assertEqual(response.status_code, 200)
        start_build.assert_called_once_with(force_rebuild=True)

    def test_init_index_recovers_from_stale_lock(self) -> None:
        with open(self.indexer._lock_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "pid": 99999999,
                    "created_at": "2026-03-14T00:00:00",
                    "updated_at": "2026-03-14T00:00:00",
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
        with open(self.indexer._status_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "status": "processing",
                    "message": "索引构建中",
                    "total_count": 10,
                    "indexed_count": 4,
                    "failed_count": 0,
                    "fallback_ratio": 0.0,
                    "index_path": self.indexer.vector_store.index_path,
                    "elapsed_time": 30.0,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

        with patch.object(self.indexer, "start_build_in_background", return_value={"status": "processing"}) as start_build:
            response = self.client.post("/init_index", json={})

        self.assertEqual(response.status_code, 200)
        start_build.assert_called_once_with(force_rebuild=False)

    def test_search_photos_returns_photo_path_and_url(self) -> None:
        self._index_photos()
        response = self.client.post(
            "/search_photos",
            json={
                "query": "照片搜索内容",
                "top_k": 2,
                "enable_text_rerank": True,
                "enable_visual_rerank": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("photo_path", data["results"][0])
        self.assertIn("photo_url", data["results"][0])
        self.assertIn("match_summary", data["results"][0])
        self.assertIn("search_debug", data)
        self.assertIn("rounds", data["search_debug"])
        self.assertTrue(data["text_reranked"])
        self.assertTrue(data["visual_reranked"])

    def test_search_by_image_returns_results(self) -> None:
        paths = self._index_photos()
        response = self.client.post(
            "/search_by_image",
            json={
                "image_path": paths[0],
                "top_k": 2,
                "enable_visual_rerank": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["query_image_path"], paths[0])
        self.assertIn("search_debug", data)
        self.assertLessEqual(len(data["results"]), 2)
        self.assertTrue(data["visual_reranked"])

    def test_search_by_image_requires_path(self) -> None:
        response = self.client.post("/search_by_image", json={"image_path": ""})
        self.assertEqual(response.status_code, 400)
        self.assertIn("图片路径不能为空", response.get_json()["message"])

    def test_search_by_uploaded_image_returns_results(self) -> None:
        self._index_photos()
        upload_buffer = BytesIO()
        Image.new("RGB", (40, 40), color=(0, 128, 255)).save(upload_buffer, format="JPEG")
        upload_buffer.seek(0)
        payload = {
            "image": (upload_buffer, "query.jpg"),
            "top_k": "2",
            "query_hint": "优先找同场景",
        }
        response = self.client.post(
            "/search_by_uploaded_image",
            data=payload,
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["query_image_name"], "query.jpg")
        self.assertTrue(bool(data["query_image_path"]))
        self.assertIn("search_debug", data)
        self.assertLessEqual(len(data["results"]), 2)

    def test_index_page_contains_search_planning_panel(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("检索规划", html)
        self.assertIn("planner-panel", html)

    def test_search_by_uploaded_image_requires_file(self) -> None:
        response = self.client.post(
            "/search_by_uploaded_image",
            data={},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("请上传图片文件", response.get_json()["message"])

    def test_rerank_pipeline_uses_full_candidate_pool_before_final_cut(self) -> None:
        class RecordingTextRerankService:
            def __init__(self) -> None:
                self.calls = []

            def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
                self.calls.append({"query": query, "count": len(candidates), "top_k": top_k})
                return list(reversed(candidates))[:top_k]

            def is_enabled(self) -> bool:
                return True

        class RecordingVisualRerankService:
            def __init__(self) -> None:
                self.calls = []

            def rerank(self, query: str, candidates: list[dict[str, Any]], rerank_top_k: int) -> list[dict[str, Any]]:
                self.calls.append({"query": query, "count": len(candidates), "top_k": rerank_top_k})
                return sorted(candidates, key=lambda item: item.get("photo_path", ""))[:rerank_top_k]

            def is_enabled(self) -> bool:
                return True

        results = [
            {"photo_path": f"/tmp/photo_{index}.jpg", "score": 0.9 - index * 0.1}
            for index in range(5)
        ]
        text_service = RecordingTextRerankService()
        visual_service = RecordingVisualRerankService()

        reranked_results, rerank_state = _apply_rerank_pipeline(
            results=results,
            top_k=5,
            rerank_top_k=2,
            enable_text_rerank=True,
            enable_visual_rerank=True,
            text_query="请给我一张河南说唱之神的演出照片",
            reference_image_path=None,
            text_rerank_service=text_service,
            visual_rerank_service=visual_service,
        )

        self.assertEqual(text_service.calls[0]["count"], 5)
        self.assertEqual(text_service.calls[0]["top_k"], 5)
        self.assertEqual(visual_service.calls[0]["count"], 5)
        self.assertEqual(visual_service.calls[0]["top_k"], 5)
        self.assertEqual(len(reranked_results), 2)
        self.assertTrue(rerank_state["text_reranked"])
        self.assertTrue(rerank_state["visual_reranked"])

    def test_rerank_pipeline_does_not_trim_results_when_rerank_is_disabled(self) -> None:
        results = [
            {"photo_path": f"/tmp/photo_{index}.jpg", "score": 0.95 - index * 0.01}
            for index in range(12)
        ]

        final_results, rerank_state = _apply_rerank_pipeline(
            results=results,
            top_k=12,
            rerank_top_k=8,
            enable_text_rerank=False,
            enable_visual_rerank=False,
            text_query="河南说唱之神演出",
            reference_image_path=None,
            text_rerank_service=None,
            visual_rerank_service=None,
        )

        self.assertEqual(len(final_results), 12)
        self.assertEqual([item["rank"] for item in final_results], list(range(1, 13)))
        self.assertFalse(rerank_state["text_reranked"])
        self.assertFalse(rerank_state["visual_reranked"])

    def test_calculate_rerank_search_pool_k_expands_candidate_pool_for_visual_rerank(self) -> None:
        self.assertEqual(
            _calculate_rerank_search_pool_k(
                top_k=5,
                rerank_top_k=5,
                enable_text_rerank=False,
                enable_visual_rerank=True,
            ),
            15,
        )

    def test_search_photos_expands_base_pool_when_rerank_enabled(self) -> None:
        with patch.object(self.searcher, "search", return_value=[]) as search_mock:
            response = self.client.post(
                "/search_photos",
                json={
                    "query": "河南说唱之神",
                    "top_k": 5,
                    "rerank_top_k": 5,
                    "enable_visual_rerank": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        search_mock.assert_called_once_with("河南说唱之神", 15)

    def test_search_by_image_expands_base_pool_when_rerank_enabled(self) -> None:
        with patch.object(self.searcher, "search_by_image_path", return_value=[]) as search_mock:
            response = self.client.post(
                "/search_by_image",
                json={
                    "image_path": "/tmp/query.jpg",
                    "top_k": 4,
                    "rerank_top_k": 4,
                    "enable_visual_rerank": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        search_mock.assert_called_once_with("/tmp/query.jpg", 12)

    def test_open_photo_location(self) -> None:
        paths = self._index_photos(1)
        with patch("api.routes.open_in_file_manager") as opener:
            response = self.client.post("/open_photo_location", json={"image_path": paths[0]})
        self.assertEqual(response.status_code, 200)
        opener.assert_called_once()

    def test_open_photo_location_missing_file(self) -> None:
        with patch("api.routes.open_in_file_manager", side_effect=FileNotFoundError("文件不存在")):
            response = self.client.post("/open_photo_location", json={"image_path": "/tmp/missing.jpg"})
        self.assertEqual(response.status_code, 404)

    def test_photo_get_success(self) -> None:
        photo_dir = os.path.join(self.tmpdir, "photos")
        img_path = os.path.join(photo_dir, "test.jpg")
        _create_image(img_path, size=(100, 80))

        response = self.client.get(f"/photo?path={img_path}")
        self.assertEqual(response.status_code, 200)
        self.assertIn("image/jpeg", response.content_type)

    def test_search_photos_no_query(self) -> None:
        response = self.client.post("/search_photos", json={"query": ""})
        self.assertEqual(response.status_code, 400)
        self.assertIn("查询内容不能为空", response.get_json()["message"])

    def test_search_rejected_while_indexing(self) -> None:
        lock_path = os.path.join(self.tmpdir, "data", "indexing.lock")
        with open(lock_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "pid": os.getpid(),
                    "created_at": "2026-03-14T00:00:00",
                    "updated_at": "2026-03-14T00:00:00",
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

        response = self.client.post("/search_photos", json={"query": "照片搜索内容"})
        self.assertEqual(response.status_code, 409)
        self.assertIn("索引仍在构建中", response.get_json()["message"])

    def test_search_by_image_rejected_while_indexing(self) -> None:
        lock_path = os.path.join(self.tmpdir, "data", "indexing.lock")
        with open(lock_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "pid": os.getpid(),
                    "created_at": "2026-03-14T00:00:00",
                    "updated_at": "2026-03-14T00:00:00",
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

        response = self.client.post("/search_by_image", json={"image_path": "/tmp/test.jpg"})
        self.assertEqual(response.status_code, 409)
        self.assertIn("索引仍在构建中", response.get_json()["message"])


if __name__ == "__main__":
    unittest.main()
