import unittest
import os
import tempfile
from unittest.mock import Mock

from PIL import Image

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

    def test_chat_backend_works_with_openai_compatible_client(self) -> None:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content='{"ranking":[{"index":2,"score":0.91},{"index":1,"score":0.82}]}'))]
        )
        service = TextRerankService(
            api_key="test-key",
            model_name="kimi-k2",
            base_url="https://api.moonshot.cn/v1",
            client=mock_client,
            backend="chat",
        )

        candidates = [
            {"photo_path": "/a.jpg", "description": "A"},
            {"photo_path": "/b.jpg", "description": "B"},
        ]
        results = service.rerank("查询", candidates, 2)

        self.assertEqual(results[0]["photo_path"], "/b.jpg")
        self.assertEqual(results[0]["text_rerank_score"], 0.91)

    def test_local_ollama_rerank_does_not_require_api_key(self) -> None:
        mock_client = Mock()
        service = TextRerankService(
            api_key="",
            model_name="qwen2.5:7b-instruct",
            base_url="http://localhost:11434",
            client=mock_client,
            backend="chat",
        )

        self.assertTrue(service.is_enabled())
        self.assertEqual(service.api_key, "ollama")


class VisualRerankServiceTests(unittest.TestCase):
    def test_create_completion_falls_back_to_string_content_after_structured_attempts_fail(self) -> None:
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [
            ValueError("structured content not supported"),
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
        self.assertEqual(mock_client.chat.completions.create.call_count, 3)
        fallback_call = mock_client.chat.completions.create.call_args_list[2]
        self.assertIsInstance(fallback_call.kwargs["messages"][0]["content"], str)

    def test_prompt_requests_direct_visual_relevance_over_carrier_text(self) -> None:
        service = VisualRerankService(
            api_key="test-key",
            model_name="gpt-5.4",
            base_url="https://www.su8.codes/codex/v1",
            client=Mock(),
        )

        prompt = service._build_prompt("请给我一张河南说唱之神的演出照片", 3)

        self.assertIn("直接呈现了用户要找的主体", prompt)
        self.assertIn("不要因为图片里出现相关文字", prompt)
        self.assertIn("嵌入式屏幕或二次载体", prompt)
        self.assertNotIn("截图、海报、专辑封面", prompt)

    def test_rerank_scores_all_candidates_and_then_returns_global_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index in range(5):
                path = os.path.join(tmp, f"candidate_{index}.jpg")
                Image.new("RGB", (32, 32), color=(10 * index, 20, 30)).save(path)
                paths.append(path)

            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = [
                Mock(choices=[Mock(message=Mock(content='{"ranking":[2,1]}'))]),
                Mock(choices=[Mock(message=Mock(content='{"ranking":[2,1]}'))]),
                Mock(choices=[Mock(message=Mock(content='{"ranking":[4,1,2,3,5]}'))]),
            ]
            service = VisualRerankService(
                api_key="test-key",
                model_name="gpt-5.4",
                base_url="https://www.su8.codes/codex/v1",
                max_images=2,
                client=mock_client,
            )

            candidates = [
                {"photo_path": paths[0], "description": "a"},
                {"photo_path": paths[1], "description": "b"},
                {"photo_path": paths[2], "description": "c"},
                {"photo_path": paths[3], "description": "d"},
                {"photo_path": paths[4], "description": "e"},
            ]

            results = service.rerank("请给我一张河南说唱之神的演出照片", candidates, 2)

            self.assertEqual(mock_client.chat.completions.create.call_count, 3)
            self.assertEqual([item["photo_path"] for item in results], [paths[2], paths[1]])
            self.assertIn("visual_rerank_score", results[0])
            self.assertGreater(results[0]["visual_rerank_score"], results[1]["visual_rerank_score"])

    def test_rerank_preserves_unprocessable_candidates_as_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            valid_path = os.path.join(tmp, "candidate_valid.jpg")
            Image.new("RGB", (32, 32), color=(20, 40, 60)).save(valid_path)

            mock_client = Mock()
            service = VisualRerankService(
                api_key="test-key",
                model_name="gpt-5.4",
                base_url="https://www.su8.codes/codex/v1",
                client=mock_client,
            )
            service._rerank_in_batches = Mock(
                return_value=[
                    {
                        "photo_path": valid_path,
                        "description": "valid",
                        "score": 0.91,
                        "visual_rerank_score": 1.0,
                    }
                ]
            )

            candidates = [
                {"photo_path": valid_path, "description": "valid", "score": 0.91},
                {"photo_path": os.path.join(tmp, "missing.jpg"), "description": "missing", "score": 0.89},
                {"photo_path": "", "description": "blank", "score": 0.88},
            ]

            results = service.rerank("请给我一张河南说唱之神的演出照片", candidates, 3)

            self.assertEqual(len(results), 3)
            self.assertEqual(results[0]["photo_path"], valid_path)
            self.assertEqual(results[1]["photo_path"], os.path.join(tmp, "missing.jpg"))
            self.assertEqual(results[2]["photo_path"], "")

    def test_rerank_by_reference_image_preserves_unprocessable_candidates_as_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reference_path = os.path.join(tmp, "reference.jpg")
            valid_path = os.path.join(tmp, "candidate_valid.jpg")
            Image.new("RGB", (32, 32), color=(10, 20, 30)).save(reference_path)
            Image.new("RGB", (32, 32), color=(40, 50, 60)).save(valid_path)

            mock_client = Mock()
            service = VisualRerankService(
                api_key="test-key",
                model_name="gpt-5.4",
                base_url="https://www.su8.codes/codex/v1",
                client=mock_client,
            )
            service._rerank_in_batches = Mock(
                return_value=[
                    {
                        "photo_path": valid_path,
                        "description": "valid",
                        "score": 0.86,
                        "visual_rerank_score": 1.0,
                    }
                ]
            )

            candidates = [
                {"photo_path": valid_path, "description": "valid", "score": 0.86},
                {"photo_path": os.path.join(tmp, "missing.jpg"), "description": "missing", "score": 0.84},
            ]

            results = service.rerank_by_reference_image(reference_path, candidates, 2)

            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["photo_path"], valid_path)
            self.assertEqual(results[1]["photo_path"], os.path.join(tmp, "missing.jpg"))


if __name__ == "__main__":
    unittest.main()
