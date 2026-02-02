import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from unittest.mock import MagicMock, patch

import main


class MainModuleTests(unittest.TestCase):
    def test_validate_required_config_missing_photo_dir(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            main._validate_required_config({"OPENROUTER_API_KEY": "key"})
        self.assertIn("PHOTO_DIR环境变量未设置", str(ctx.exception))

    def test_validate_required_config_missing_api_key(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            main._validate_required_config({"PHOTO_DIR": "C:/photos"})
        self.assertIn("OPENROUTER_API_KEY环境变量未设置", str(ctx.exception))

    def test_initialize_services_wires_dependencies(self) -> None:
        config = {
            "PHOTO_DIR": "C:/photos",
            "DATA_DIR": "C:/data",
            "OPENROUTER_API_KEY": "test-key",
            "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
            "VISION_MODEL_NAME": "openai/gpt-4o",
            "EMBEDDING_MODEL_NAME": "sentence-t5-base",
            "EMBEDDING_DIMENSION": 768,
            "TIME_PARSE_MODEL_NAME": "openai/gpt-3.5-turbo",
            "SERVER_HOST": "localhost",
            "SERVER_PORT": 5000,
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
        }

        with patch.object(main, "VectorStore") as vector_store_cls, \
            patch.object(main, "OpenRouterVisionLLMService") as vision_cls, \
            patch.object(main, "T5EmbeddingService") as embedding_cls, \
            patch.object(main, "TimeParser") as time_parser_cls, \
            patch.object(main, "Indexer") as indexer_cls, \
            patch.object(main, "Searcher") as searcher_cls:

            vector_store = MagicMock()
            vision_service = MagicMock()
            embedding_service = MagicMock()
            time_parser = MagicMock()
            indexer = MagicMock()
            searcher = MagicMock()

            vector_store_cls.return_value = vector_store
            vision_cls.return_value = vision_service
            embedding_cls.return_value = embedding_service
            time_parser_cls.return_value = time_parser
            indexer_cls.return_value = indexer
            searcher_cls.return_value = searcher

            result = main.initialize_services(config)

            vector_store_cls.assert_called_once_with(
                dimension=768,
                index_path="C:/data/photo_search.index",
                metadata_path="C:/data/metadata.json",
            )
            vision_cls.assert_called_once()
            embedding_cls.assert_called_once_with(model_name="sentence-t5-base")
            time_parser_cls.assert_called_once()
            indexer_cls.assert_called_once_with(
                photo_dir="C:/photos",
                vision=vision_service,
                embedding=embedding_service,
                vector_store=vector_store,
                data_dir="C:/data",
                batch_size=8,
                max_retries=2,
                timeout=15,
            )
            searcher_cls.assert_called_once_with(
                embedding=embedding_service,
                time_parser=time_parser,
                vector_store=vector_store,
                data_dir="C:/data",
                top_k=5,
            )
            self.assertEqual(result, (indexer, searcher))

    def test_create_app_registers_routes_and_errors(self) -> None:
        indexer = MagicMock()
        searcher = MagicMock()
        config = {"SECRET_KEY": "test-secret"}

        with patch.object(main, "register_routes") as register_routes:
            app = main.create_app(indexer, searcher, config)
            register_routes.assert_called_once_with(app, indexer, searcher, config)

            client = app.test_client()
            response = client.get("/missing")
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.get_json()["message"], "接口不存在")

    def test_main_runs_app(self) -> None:
        config = {
            "PHOTO_DIR": "C:/photos",
            "OPENROUTER_API_KEY": "key",
            "SERVER_HOST": "127.0.0.1",
            "SERVER_PORT": 5001,
        }
        mock_app = MagicMock()

        with patch.object(main, "load_config", return_value=config), \
            patch.object(main, "initialize_services", return_value=(MagicMock(), MagicMock())), \
            patch.object(main, "create_app", return_value=mock_app):
            main.main()

        mock_app.run.assert_called_once_with(host="127.0.0.1", port=5001, debug=False)


if __name__ == "__main__":
    unittest.main()
