import unittest
from unittest.mock import Mock

from utils.embedding_service import TextRerankService, TumuerEmbeddingService
from utils.rerank_service import VisualRerankService


class EmbeddingServiceTests(unittest.TestCase):
    def test_tumuer_embedding_requires_api_key(self) -> None:
        with self.assertRaises(ValueError):
            TumuerEmbeddingService(api_key="", model_name="model", base_url="https://router.tumuer.me/v1")

    def test_tumuer_embedding_returns_list(self) -> None:
        mock_client = Mock()
        mock_client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=[0.1, 0.2, 0.3])]
        )
        service = TumuerEmbeddingService(
            api_key="test-key",
            model_name="Qwen/Qwen3-Embedding-8B",
            base_url="https://router.tumuer.me/v1",
            client=mock_client,
        )
        result = service.generate_embedding("测试文本")
        self.assertEqual(result, [0.1, 0.2, 0.3])
        self.assertEqual(service.dimension, 3)
        _, kwargs = mock_client.embeddings.create.call_args
        self.assertNotIn("dimensions", kwargs)

    def test_tumuer_embedding_batch(self) -> None:
        mock_client = Mock()
        mock_client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=[0.1, 0.2]), Mock(embedding=[0.3, 0.4])]
        )
        service = TumuerEmbeddingService(
            api_key="test-key",
            model_name="Qwen/Qwen3-Embedding-8B",
            base_url="https://router.tumuer.me/v1",
            client=mock_client,
        )
        results = service.generate_embedding_batch(["文本1", "文本2"])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[1], [0.3, 0.4])

    def test_tumuer_embedding_omits_dimensions_when_unknown(self) -> None:
        mock_client = Mock()
        mock_client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=[0.1, 0.2, 0.3])]
        )
        service = TumuerEmbeddingService(
            api_key="test-key",
            model_name="Qwen/Qwen3-Embedding-8B",
            base_url="https://router.tumuer.me/v1",
            client=mock_client,
            dimension=None,
        )
        service.generate_embedding("测试文本")
        _, kwargs = mock_client.embeddings.create.call_args
        self.assertNotIn("dimensions", kwargs)


class TextRerankServiceTests(unittest.TestCase):
    def test_requires_api_key(self) -> None:
        with self.assertRaises(ValueError):
            TextRerankService(api_key="", model_name="model", base_url="https://router.tumuer.me/v1")

    def test_rerank_reorders_candidates(self) -> None:
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.90},
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_session.post.return_value = mock_response

        service = TextRerankService(
            api_key="test-key",
            model_name="Qwen/Qwen3-Reranker-8B",
            base_url="https://router.tumuer.me/v1",
            session=mock_session,
        )
        candidates = [
            {"photo_path": "/a.jpg", "description": "A"},
            {"photo_path": "/b.jpg", "description": "B"},
        ]
        results = service.rerank("查询", candidates, 2)
        self.assertEqual(results[0]["photo_path"], "/b.jpg")
        self.assertIn("text_rerank_score", results[0])


class VisualRerankServiceTests(unittest.TestCase):
    def test_create_completion_falls_back_to_string_content(self) -> None:
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [
            ValueError("structured content not supported"),
            Mock(choices=[Mock(message=Mock(content='{"ranking":[1]}'))]),
        ]
        service = VisualRerankService(
            api_key="test-key",
            model_name="gpt-5.4",
            base_url="https://www.su8.codes/codex/v1",
            client=mock_client,
        )

        result = service._create_completion([{"type": "text", "text": "test"}])
        self.assertIsNotNone(result)
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)
        second_call = mock_client.chat.completions.create.call_args_list[1]
        self.assertIsInstance(second_call.kwargs["messages"][0]["content"], str)


if __name__ == "__main__":
    unittest.main()
