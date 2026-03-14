import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import main


class MainModuleTests(unittest.TestCase):
    def test_validate_required_config_missing_photo_dir(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            main._validate_required_config({"SU8_API_KEY": "key", "EMBEDDING_API_KEY": "emb"})
        self.assertIn("PHOTO_DIR环境变量未设置", str(ctx.exception))

    def test_validate_required_config_missing_su8_key(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            main._validate_required_config({"PHOTO_DIR": "C:/photos", "EMBEDDING_API_KEY": "emb"})
        self.assertIn("SU8_API_KEY环境变量未设置", str(ctx.exception))

    def test_validate_required_config_missing_embedding_key(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            main._validate_required_config({"PHOTO_DIR": "C:/photos", "SU8_API_KEY": "key"})
        self.assertIn("EMBEDDING_API_KEY环境变量未设置", str(ctx.exception))

    def test_initialize_services_wires_dependencies(self) -> None:
        config = {
            "PHOTO_DIR": "C:/photos",
            "DATA_DIR": "C:/data",
            "SU8_API_KEY": "su8-key",
            "SU8_BASE_URL": "https://www.su8.codes/codex/v1",
            "VISION_MODEL": "gpt-5.4",
            "VISION_REASONING_EFFORT": "medium",
            "STRUCTURED_ANALYSIS_ENABLED": True,
            "ENHANCED_ANALYSIS_ENABLED": True,
            "TAG_MIN_CONFIDENCE": 0.65,
            "IDENTITY_TEXT_MIN_CONFIDENCE": 0.7,
            "IDENTITY_VISUAL_MIN_CONFIDENCE": 0.92,
            "TIME_PARSE_MODEL": "gpt-5.1",
            "TIME_PARSE_REASONING_EFFORT": "low",
            "EMBEDDING_API_KEY": "embed-key",
            "EMBEDDING_BASE_URL": "https://router.tumuer.me/v1",
            "EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-8B",
            "EMBEDDING_DIMENSION": 4096,
            "TEXT_RERANK_API_KEY": "rerank-key",
            "TEXT_RERANK_BASE_URL": "https://router.tumuer.me/v1",
            "TEXT_RERANK_MODEL": "Qwen/Qwen3-Reranker-8B",
            "VISUAL_RERANK_ENABLED": True,
            "VISUAL_RERANK_MODEL": "gpt-5.4",
            "BATCH_SIZE": 8,
            "MAX_RETRIES": 2,
            "TIMEOUT": 15,
            "TOP_K": 5,
            "INDEX_PATH": "C:/data/photo_search.index",
            "METADATA_PATH": "C:/data/metadata.json",
            "USE_BASE64": True,
            "IMAGE_MAX_SIZE": 1024,
            "IMAGE_QUALITY": 85,
            "IMAGE_FORMAT": "WEBP",
            "VECTOR_METRIC": "cosine",
            "VECTOR_WEIGHT": 0.8,
            "KEYWORD_WEIGHT": 0.2,
        }

        with patch.object(main, "VectorStore") as vector_store_cls, \
            patch.object(main, "SU8VisionLLMService") as vision_cls, \
            patch.object(main, "TumuerEmbeddingService") as embedding_cls, \
            patch.object(main, "TextRerankService") as text_rerank_cls, \
            patch.object(main, "TimeParser") as time_parser_cls, \
            patch.object(main, "Indexer") as indexer_cls, \
            patch.object(main, "Searcher") as searcher_cls, \
            patch("main.normalize_local_path", return_value="/mnt/c/photos"), \
            patch("utils.rerank_service.VisualRerankService") as visual_rerank_cls:

            vector_store = MagicMock()
            vision_service = MagicMock()
            embedding_service = MagicMock()
            time_parser = MagicMock()
            text_rerank_service = MagicMock()
            visual_rerank_service = MagicMock()
            indexer = MagicMock()
            searcher = MagicMock()

            vector_store_cls.return_value = vector_store
            vision_cls.return_value = vision_service
            embedding_cls.return_value = embedding_service
            time_parser_cls.return_value = time_parser
            text_rerank_cls.return_value = text_rerank_service
            visual_rerank_cls.return_value = visual_rerank_service
            indexer_cls.return_value = indexer
            searcher_cls.return_value = searcher

            result = main.initialize_services(config)

            vector_store_cls.assert_called_once_with(
                dimension=4096,
                index_path="C:/data/photo_search.index",
                metadata_path="C:/data/metadata.json",
                metric="cosine",
            )
            embedding_cls.assert_called_once()
            vision_cls.assert_called_once()
            time_parser_cls.assert_called_once()
            text_rerank_cls.assert_called_once()
            indexer_cls.assert_called_once()
            searcher_cls.assert_called_once()
            self.assertEqual(result, (indexer, searcher, text_rerank_service, visual_rerank_service))

    def test_initialize_services_falls_back_to_su8_base_url_for_query_formatter(self) -> None:
        config = {
            "PHOTO_DIR": "C:/photos",
            "DATA_DIR": "C:/data",
            "SU8_API_KEY": "su8-key",
            "SU8_BASE_URL": "https://www.su8.codes/codex/v1",
            "VISION_MODEL": "gpt-5.4",
            "VISION_REASONING_EFFORT": "medium",
            "STRUCTURED_ANALYSIS_ENABLED": True,
            "ENHANCED_ANALYSIS_ENABLED": True,
            "TAG_MIN_CONFIDENCE": 0.65,
            "IDENTITY_TEXT_MIN_CONFIDENCE": 0.7,
            "IDENTITY_VISUAL_MIN_CONFIDENCE": 0.92,
            "TIME_PARSE_MODEL": "gpt-5.1",
            "TIME_PARSE_REASONING_EFFORT": "low",
            "QUERY_FORMAT_ENABLED": True,
            "QUERY_FORMAT_API_KEY": "su8-key",
            "QUERY_FORMAT_BASE_URL": "",
            "QUERY_FORMAT_MODEL": "gpt-5.1",
            "QUERY_FORMAT_REASONING_EFFORT": "low",
            "EMBEDDING_API_KEY": "embed-key",
            "EMBEDDING_BASE_URL": "https://router.tumuer.me/v1",
            "EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-8B",
            "EMBEDDING_DIMENSION": 4096,
            "TEXT_RERANK_API_KEY": "",
            "VISUAL_RERANK_ENABLED": False,
            "BATCH_SIZE": 8,
            "MAX_RETRIES": 2,
            "TIMEOUT": 15,
            "TOP_K": 5,
            "INDEX_PATH": "C:/data/photo_search.index",
            "METADATA_PATH": "C:/data/metadata.json",
            "USE_BASE64": True,
            "IMAGE_MAX_SIZE": 1024,
            "IMAGE_QUALITY": 85,
            "IMAGE_FORMAT": "WEBP",
            "VECTOR_METRIC": "cosine",
            "VECTOR_WEIGHT": 0.8,
            "KEYWORD_WEIGHT": 0.2,
        }

        with patch.object(main, "VectorStore") as vector_store_cls, \
            patch.object(main, "SU8VisionLLMService") as vision_cls, \
            patch.object(main, "TumuerEmbeddingService") as embedding_cls, \
            patch.object(main, "TimeParser") as time_parser_cls, \
            patch.object(main, "Indexer") as indexer_cls, \
            patch.object(main, "Searcher") as searcher_cls, \
            patch("main.normalize_local_path", return_value="/mnt/c/photos"), \
            patch("utils.query_formatter.QueryFormatter") as query_formatter_cls:

            vector_store_cls.return_value = MagicMock()
            vision_cls.return_value = MagicMock()
            embedding_cls.return_value = MagicMock()
            time_parser_cls.return_value = MagicMock()
            indexer_cls.return_value = MagicMock()
            searcher_cls.return_value = MagicMock()
            query_formatter_cls.return_value = MagicMock()

            main.initialize_services(config)

            _, kwargs = query_formatter_cls.call_args
            self.assertEqual(kwargs["base_url"], "https://www.su8.codes/codex/v1")

    def test_create_app_registers_routes_and_errors(self) -> None:
        indexer = MagicMock()
        searcher = MagicMock()
        config = {"SECRET_KEY": "test-secret"}

        with patch.object(main, "register_routes") as register_routes:
            app = main.create_app(indexer, searcher, config)
            register_routes.assert_called_once()

            client = app.test_client()
            response = client.get("/missing")
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.get_json()["message"], "接口不存在")

    def test_main_runs_app(self) -> None:
        config = {
            "PHOTO_DIR": "C:/photos",
            "SU8_API_KEY": "key",
            "EMBEDDING_API_KEY": "emb",
            "SERVER_HOST": "127.0.0.1",
            "SERVER_PORT": 5001,
        }
        mock_app = MagicMock()

        with patch.object(main, "load_config", return_value=config), \
            patch.object(main, "initialize_services", return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock())), \
            patch.object(main, "create_app", return_value=mock_app):
            main.main()

        mock_app.run.assert_called_once_with(host="127.0.0.1", port=5001, debug=False)


if __name__ == "__main__":
    unittest.main()
