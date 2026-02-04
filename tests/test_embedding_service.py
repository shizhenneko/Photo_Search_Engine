import os
import sys
import unittest
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import get_config
import torch
from unittest.mock import Mock

from utils.embedding_service import OpenAIEmbeddingService, T5EmbeddingService, VolcanoEmbeddingService


class EmbeddingServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """加载配置"""
        cls.config = get_config()
        cls.has_openrouter_key = bool(cls.config.get("OPENROUTER_API_KEY"))
        cls.cuda_available = torch.cuda.is_available()

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
            if self.cuda_available:
                service = T5EmbeddingService(model_name="sentence-t5-base", device="cuda")
            else:
                service = T5EmbeddingService(model_name="sentence-t5-base", device="cpu")
            self.assertIsNotNone(service.model)
        except ImportError:
            self.skipTest("未安装sentence-transformers")

    def test_t5_embedding_device_selection(self) -> None:
        """测试T5 Embedding设备选择（GPU优先）"""
        try:
            if self.cuda_available:
                service = T5EmbeddingService(model_name="sentence-t5-base")
                self.assertEqual(service.model.device.type, "cuda")
            else:
                service = T5EmbeddingService(model_name="sentence-t5-base")
                self.assertEqual(service.model.device.type, "cpu")
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
        # OpenAI embeddings may have slight variations, only check dimension consistency
        # not exact vector match (which would require deterministic model behavior)


class TestVolcanoEmbeddingService(unittest.TestCase):
    """火山引擎 Embedding 服务测试。"""
    
    def test_init_requires_api_key(self) -> None:
        """测试初始化必须提供 API 密钥。"""
        with self.assertRaises(ValueError) as context:
            VolcanoEmbeddingService(api_key="")
        self.assertIn("VOLCANO_API_KEY", str(context.exception))
    
    def test_dimension_is_4096(self) -> None:
        """测试向量维度固定为 4096。"""
        # Mock 客户端
        mock_client = Mock()
        service = VolcanoEmbeddingService(
            api_key="test-key",
            client=mock_client,
        )
        self.assertEqual(service.dimension, 4096)
    
    def test_generate_embedding_returns_list(self) -> None:
        """测试生成嵌入返回列表。"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 4096)]
        mock_client.embeddings.create.return_value = mock_response
        
        service = VolcanoEmbeddingService(
            api_key="test-key",
            client=mock_client,
        )
        
        result = service.generate_embedding("测试文本")
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4096)
    
    def test_generate_embedding_retries_on_failure(self) -> None:
        """测试失败后自动重试。"""
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = [
            Exception("网络错误"),
            Exception("网络错误"),
            Mock(data=[Mock(embedding=[0.1] * 4096)]),  # 第三次成功
        ]
        
        service = VolcanoEmbeddingService(
            api_key="test-key",
            max_retries=3,
            client=mock_client,
        )
        
        result = service.generate_embedding("测试文本")
        self.assertEqual(len(result), 4096)
        self.assertEqual(mock_client.embeddings.create.call_count, 3)
    
    def test_generate_embedding_raises_after_max_retries(self) -> None:
        """测试超过最大重试次数后抛出异常。"""
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("持续失败")
        
        service = VolcanoEmbeddingService(
            api_key="test-key",
            max_retries=3,
            client=mock_client,
        )
        
        with self.assertRaises(ValueError) as context:
            service.generate_embedding("测试文本")
        self.assertIn("向量生成失败", str(context.exception))
    
    def test_generate_embedding_batch(self) -> None:
        """测试批量生成嵌入。"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * 4096),
            Mock(embedding=[0.2] * 4096),
        ]
        mock_client.embeddings.create.return_value = mock_response
        
        service = VolcanoEmbeddingService(
            api_key="test-key",
            client=mock_client,
        )
        
        texts = ["文本1", "文本2"]
        results = service.generate_embedding_batch(texts)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(len(results[0]), 4096)
    
    @unittest.skipIf(
        not bool(os.getenv("VOLCANO_API_KEY")),
        "VOLCANO_API_KEY 未设置，跳过集成测试"
    )
    def test_real_api_integration(self) -> None:
        """集成测试：真实 API 调用（需要配置环境变量）。"""
        service = VolcanoEmbeddingService(
            api_key=os.getenv("VOLCANO_API_KEY"),
            base_url=os.getenv("VOLCANO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        )
        
        result = service.generate_embedding("测试火山引擎嵌入服务")
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4096)
        self.assertTrue(all(isinstance(x, float) for x in result))


if __name__ == "__main__":
    unittest.main()
