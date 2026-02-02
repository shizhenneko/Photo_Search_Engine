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

from utils.vision_llm_service import OpenRouterVisionLLMService, LocalVisionLLMService


class VisionServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """加载配置"""
        cls.config = get_config()
        cls.has_api_key = bool(cls.config.get("OPENROUTER_API_KEY"))

    def test_local_vision_service(self) -> None:
        """测试本地Vision服务（离线模式）"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "local.jpg")
            image = Image.new("RGB", (80, 60), color=(1, 2, 3))
            image.save(path)
            service = LocalVisionLLMService()
            result = service.generate_description(path)
            self.assertIn("80x60", result)

    def test_local_vision_service_batch(self) -> None:
        """测试本地Vision服务批量生成"""
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in range(3):
                path = os.path.join(tmp, f"local{i}.jpg")
                image = Image.new("RGB", (100 + i * 10, 50 + i * 10), color=(i, i, i))
                image.save(path)
                paths.append(path)

            service = LocalVisionLLMService()
            results = service.generate_description_batch(paths)

            self.assertEqual(len(results), len(paths))
            for i, result in enumerate(results):
                expected_size = f"{100 + i * 10}x{50 + i * 10}"
                self.assertIn(expected_size, result)

    @unittest.skipIf(
        not bool(os.getenv("OPENROUTER_API_KEY")),
        "OPENROUTER_API_KEY未设置，跳过集成测试"
    )
    def test_openrouter_vision_service_init(self) -> None:
        """测试OpenRouter Vision服务初始化（集成测试）"""
        api_key = self.config["OPENROUTER_API_KEY"]
        base_url = self.config.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        service = OpenRouterVisionLLMService(
            api_key=api_key,
            base_url=base_url,
        )

        self.assertEqual(service.api_key, api_key)
        self.assertEqual(service.base_url, base_url)

    @unittest.skipIf(
        not bool(os.getenv("OPENROUTER_API_KEY")),
        "OPENROUTER_API_KEY未设置，跳过集成测试"
    )
    def test_openrouter_vision_service_requires_api_key(self) -> None:
        """测试OpenRouter Vision服务需要API密钥"""
        with self.assertRaises(ValueError):
            OpenRouterVisionLLMService(api_key="")

    def test_base64_image_encoding(self) -> None:
        """测试Base64图片编码"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.jpg")
            image = Image.new("RGB", (2000, 1500), color=(100, 150, 200))
            image.save(path)

            service = OpenRouterVisionLLMService(
                api_key="test-key",
                use_base64=True,
                image_max_size=1024,
                image_quality=85,
                image_format="WEBP",
            )

            base64_url = service._get_image_base64(path)

            self.assertIn("data:image/webp;base64,", base64_url)
            self.assertGreater(len(base64_url), 100)

    def test_image_resize_optimization(self) -> None:
        """测试图片压缩优化"""
        from utils.image_parser import resize_and_optimize_image

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "large.jpg")
            image = Image.new("RGB", (3000, 2000), color=(50, 100, 150))
            image.save(path, quality=95)

            optimized_bytes = resize_and_optimize_image(
                path, max_size=1024, quality=85, format="WEBP"
            )

            original_size = os.path.getsize(path)

            optimized_size = len(optimized_bytes)

            self.assertLess(optimized_size, original_size)

    def test_default_config_values(self) -> None:
        """测试默认配置值"""
        service = OpenRouterVisionLLMService(api_key="test-key")

        self.assertTrue(service.use_base64)
        self.assertEqual(service.image_max_size, 1024)
        self.assertEqual(service.image_quality, 85)
        self.assertEqual(service.image_format, "WEBP")

    @unittest.skipIf(
        not bool(os.getenv("OPENROUTER_API_KEY")),
        "OPENROUTER_API_KEY未设置，跳过集成测试"
    )
    def test_openrouter_vision_service_generate_description(self) -> None:
        """测试OpenRouter Vision服务生成描述（集成测试）"""
        api_key = self.config["OPENROUTER_API_KEY"]
        base_url = self.config.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        service = OpenRouterVisionLLMService(
            api_key=api_key,
            base_url=base_url,
            server_host="localhost",
            server_port=5000,
            max_retries=2,
            use_public_image_url=True,
        )

        result = service.generate_description("dummy_path.jpg")

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)
        self.assertTrue(50 <= len(result) <= 200)

    @unittest.skipIf(
        not bool(os.getenv("OPENROUTER_API_KEY")),
        "OPENROUTER_API_KEY未设置，跳过集成测试"
    )
    def test_openrouter_vision_service_generate_description_batch(self) -> None:
        """测试批量生成描述（集成测试）"""
        api_key = self.config["OPENROUTER_API_KEY"]
        base_url = self.config.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        service = OpenRouterVisionLLMService(
            api_key=api_key,
            base_url=base_url,
            server_host="localhost",
            server_port=5000,
            max_retries=2,
            use_public_image_url=True,
        )

        results = service.generate_description_batch(["dummy1.jpg", "dummy2.jpg", "dummy3.jpg"])

        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 10)

    def test_get_image_url_encoding(self) -> None:
        """测试图片URL路径编码"""
        service = OpenRouterVisionLLMService(
            api_key="test-key",
            server_host="localhost",
            server_port=5000,
        )

        test_path = "C:/Users/Test/我的照片/image.jpg"
        url = service._get_image_url(test_path)

        self.assertIn("localhost:5000", url)
        self.assertIn("/photo?path=", url)


if __name__ == "__main__":
    unittest.main()
