"""
API路由单元测试

测试所有HTTP接口：
- GET / - 渲染前端页面
- POST /init_index - 触发索引构建
- POST /search_photos - 执行照片搜索
- GET /index_status - 获取索引状态
- GET /photo - 返回图片文件
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import get_config
from PIL import Image

from core.indexer import Indexer
from core.searcher import Searcher
from utils.embedding_service import T5EmbeddingService
from utils.time_parser import TimeParser
from utils.vector_store import VectorStore
from utils.vision_llm_service import LocalVisionLLMService


def _create_image(path: str, size: tuple[int, int] = (64, 48)) -> None:
    """创建测试图片"""
    image = Image.new("RGB", size, color=(255, 0, 0))
    image.save(path)


def _create_test_app():
    """创建测试用Flask应用"""
    from api.routes import register_routes
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

    vector_store = VectorStore(dimension=768, index_path=index_path, metadata_path=metadata_path)
    vision = LocalVisionLLMService()
    embedding = T5EmbeddingService(model_name="sentence-t5-base", device="cuda")
    time_parser = TimeParser(api_key="test-key")

    indexer = Indexer(photo_dir=photo_dir, vision=vision, embedding=embedding, vector_store=vector_store, data_dir=data_dir)
    searcher = Searcher(embedding=embedding, time_parser=time_parser, vector_store=vector_store, data_dir=data_dir)

    config = get_config().copy()
    config["PHOTO_DIR"] = photo_dir
    config["DATA_DIR"] = data_dir

    register_routes(app, indexer, searcher, config)

    return app, tmp, indexer, searcher


class RouteTests(unittest.TestCase):
    """路由测试类"""

    def setUp(self) -> None:
        """每个测试前创建应用"""
        self.app, self.tmpdir, self.indexer, self.searcher = _create_test_app()
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        """每个测试后清理临时目录"""
        import shutil

        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def test_index_page_renders(self) -> None:
        """测试GET / - 渲染前端页面"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("照片搜索引擎", response.get_data(as_text=True))
        self.assertIn("初始化索引", response.get_data(as_text=True))
        self.assertIn("search", response.get_data(as_text=True))

    def test_init_index_no_photos(self) -> None:
        """测试POST /init_index - 无照片时返回失败"""
        response = self.client.post("/init_index", json={})
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["status"], "failed")
        self.assertIn("未找到可索引的图片文件", data["message"])

    def test_init_index_with_photos(self) -> None:
        """测试POST /init_index - 有照片时成功构建"""
        photo_dir = os.path.join(self.tmpdir, "photos")
        for i in range(3):
            _create_image(os.path.join(photo_dir, f"photo_{i}.jpg"))

        response = self.client.post("/init_index", json={})
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["indexed_count"], 3)
        self.assertGreater(data["elapsed_time"], 0)

    def test_init_index_already_processing(self) -> None:
        """测试POST /init_index - 正在构建时返回processing"""
        photo_dir = os.path.join(self.tmpdir, "photos")
        data_dir = os.path.join(self.tmpdir, "data")

        _create_image(os.path.join(photo_dir, "photo.jpg"))

        lock_path = os.path.join(data_dir, "indexing.lock")
        with open(lock_path, "w", encoding="utf-8") as f:
            f.write("locked")

        response = self.client.post("/init_index", json={})
        self.assertEqual(response.status_code, 400)

        data = response.get_json()
        self.assertEqual(data["status"], "processing")

    def test_index_status_initial(self) -> None:
        """测试GET /index_status - 初始状态"""
        response = self.client.get("/index_status")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["status"], "idle")
        self.assertEqual(data["indexed_count"], 0)

    def test_index_status_after_build(self) -> None:
        """测试GET /index_status - 构建完成后状态"""
        photo_dir = os.path.join(self.tmpdir, "photos")
        for i in range(3):
            _create_image(os.path.join(photo_dir, f"photo_{i}.jpg"))

        self.client.post("/init_index", json={})

        response = self.client.get("/index_status")
        data = response.get_json()
        self.assertEqual(data["status"], "ready")

    def test_search_photos_no_query(self) -> None:
        """测试POST /search_photos - 空查询"""
        response = self.client.post("/search_photos", json={"query": ""})
        self.assertEqual(response.status_code, 400)

        data = response.get_json()
        self.assertEqual(data["status"], "error")
        self.assertIn("查询内容不能为空", data["message"])

    def test_search_photos_short_query(self) -> None:
        """测试POST /search_photos - 查询过短"""
        response = self.client.post("/search_photos", json={"query": "abc"})
        self.assertEqual(response.status_code, 400)

        data = response.get_json()
        self.assertEqual(data["status"], "error")
        self.assertIn("查询内容不合法", data["message"])

    def test_search_photos_no_index(self) -> None:
        """测试POST /search_photos - 索引未加载"""
        response = self.client.post("/search_photos", json={"query": "valid query text here"})
        self.assertEqual(response.status_code, 400)

        data = response.get_json()
        self.assertEqual(data["status"], "error")
        self.assertIn("索引未加载", data["message"])

    def test_search_photos_with_index(self) -> None:
        """测试POST /search_photos - 有索引时成功搜索"""
        photo_dir = os.path.join(self.tmpdir, "photos")
        data_dir = os.path.join(self.tmpdir, "data")

        for i in range(3):
            _create_image(os.path.join(photo_dir, f"photo_{i}.jpg"))

        self.client.post("/init_index", json={})

        marker_path = os.path.join(data_dir, "index_ready.marker")
        self.assertTrue(os.path.exists(marker_path))

        self.searcher.load_index()

        response = self.client.post("/search_photos", json={"query": "照片 test", "top_k": 2})
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIsInstance(data["results"], list)
        self.assertGreaterEqual(len(data["results"]), 0)

    def test_search_photos_custom_top_k(self) -> None:
        """测试POST /search_photos - 自定义top_k"""
        photo_dir = os.path.join(self.tmpdir, "photos")
        data_dir = os.path.join(self.tmpdir, "data")

        for i in range(5):
            _create_image(os.path.join(photo_dir, f"photo_{i}.jpg"))

        self.client.post("/init_index", json={})
        self.searcher.load_index()

        response = self.client.post("/search_photos", json={"query": "照片搜索内容", "top_k": 2})
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertLessEqual(len(data["results"]), 2)

    def test_photo_get_missing_path(self) -> None:
        """测试GET /photo - 缺少path参数"""
        response = self.client.get("/photo")
        self.assertEqual(response.status_code, 400)
        self.assertIn("缺少path参数", response.get_data(as_text=True))

    def test_photo_get_file_not_found(self) -> None:
        """测试GET /photo - 文件不存在"""
        response = self.client.get("/photo?path=/nonexistent/image.jpg")
        self.assertEqual(response.status_code, 404)
        self.assertIn("文件不存在", response.get_data(as_text=True))

    def test_photo_get_success(self) -> None:
        """测试GET /photo - 成功返回图片"""
        photo_dir = os.path.join(self.tmpdir, "photos")
        img_path = os.path.join(photo_dir, "test.jpg")
        _create_image(img_path, size=(100, 80))

        response = self.client.get(f"/photo?path={img_path}")
        self.assertEqual(response.status_code, 200)
        self.assertIn("image/jpeg", response.content_type)

    def test_photo_get_unsupported_format(self) -> None:
        """测试GET /photo - 不支持的格式"""
        photo_dir = os.path.join(self.tmpdir, "photos")
        txt_path = os.path.join(photo_dir, "test.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("not an image")

        response = self.client.get(f"/photo?path={txt_path}")
        self.assertEqual(response.status_code, 400)
        self.assertIn("不支持的文件格式", response.get_data(as_text=True))

    def test_photo_get_path_traversal_protection(self) -> None:
        """测试GET /photo - 路径遍历防护"""
        response = self.client.get("/photo?path=../../../etc/passwd")
        self.assertEqual(response.status_code, 403)
        self.assertIn("拒绝访问", response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
