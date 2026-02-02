import os
import sys
import unittest
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import get_config

from utils.embedding_service import OpenAIEmbeddingService, T5EmbeddingService


class EmbeddingServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """加载配置"""
        cls.config = get_config()
        cls.has_openrouter_key = bool(cls.config.get("OPENROUTER_API_KEY"))

    def test_config_loading(self) -> None:
        """测试配置从.env正确加载"""
        self.assertIsNotNone(self.config)
        self.assertIn("OPENROUTER_API_KEY", self.config)
        self.assertIn("EMBEDDING_MODEL_NAME", self.config)

    def test_openai_embedding_service_requires_api_key(self) -> None:
        """测试OpenAI Embedding服务需要API密钥"""
        with self.assertRaises(ValueError):
            OpenAIEmbeddingService(api_key="")

    @unittest.skipIf(
        not bool(os.getenv("OPENROUTER_API_KEY")),
        "OPENROUTER_API_KEY未设置，跳过集成测试"
    )
    def test_openai_embedding_real_api(self) -> None:
        """测试真实OpenRouter API嵌入生成（集成测试）"""
        api_key = self.config["OPENROUTER_API_KEY"]
        base_url = self.config.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        service = OpenAIEmbeddingService(
            api_key=api_key,
            model_name="openai/text-embedding-3-small",
            base_url=base_url,
        )

        result = service.generate_embedding("测试文本")
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        self.assertTrue(all(isinstance(x, float) for x in result))

    @unittest.skipIf(
        not bool(os.getenv("OPENROUTER_API_KEY")),
        "OPENROUTER_API_KEY未设置，跳过集成测试"
    )
    def test_openai_embedding_batch_real_api(self) -> None:
        """测试批量嵌入生成（集成测试）"""
        api_key = self.config["OPENROUTER_API_KEY"]
        base_url = self.config.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        service = OpenAIEmbeddingService(
            api_key=api_key,
            model_name="openai/text-embedding-3-small",
            base_url=base_url,
        )

        texts = ["测试文本1", "测试文本2", "测试文本3"]
        results = service.generate_embedding_batch(texts)

        self.assertEqual(len(results), len(texts))
        for result in results:
            self.assertIsInstance(result, list)
            self.assertTrue(len(result) > 0)

    def test_t5_embedding_service_init(self) -> None:
        """测试T5 Embedding服务初始化"""
        try:
            service = T5EmbeddingService(model_name="sentence-t5-base", device="cuda")
            self.assertIsNotNone(service.model)
        except ImportError:
            self.skipTest("未安装sentence-transformers")

    @unittest.skipIf(
        not bool(os.getenv("OPENROUTER_API_KEY")),
        "OPENROUTER_API_KEY未设置，跳过集成测试"
    )
    def test_embedding_vector_dimension_consistency(self) -> None:
        """测试相同文本生成相同维度的向量（集成测试）"""
        api_key = self.config["OPENROUTER_API_KEY"]
        base_url = self.config.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        service = OpenAIEmbeddingService(
            api_key=api_key,
            model_name="openai/text-embedding-3-small",
            base_url=base_url,
        )

        result1 = service.generate_embedding("相同文本")
        result2 = service.generate_embedding("相同文本")

        self.assertEqual(len(result1), len(result2))
        # 向量应该非常接近（浮点数精度）
        self.assertTrue(
            all(abs(a - b) < 1e-6 for a, b in zip(result1, result2))
        )


if __name__ == "__main__":
    unittest.main()
