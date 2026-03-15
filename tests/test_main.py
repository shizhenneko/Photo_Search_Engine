import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

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
        self.assertIn("LLM_API_KEY环境变量未设置", str(ctx.exception))

    def test_validate_required_config_missing_embedding_key(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            main._validate_required_config({"PHOTO_DIR": "C:/photos", "SU8_API_KEY": "key"})
        self.assertIn("EMBEDDING_API_KEY环境变量未设置", str(ctx.exception))

    def test_validate_required_config_allows_empty_keys_for_local_ollama(self) -> None:
        main._validate_required_config(
            {
                "PHOTO_DIR": "C:/photos",
                "LLM_BASE_URL": "http://localhost:11434",
                "EMBEDDING_BASE_URL": "http://localhost:11434",
            }
        )

    def test_initialize_services_wires_dependencies(self) -> None:
        config = {
            "PHOTO_DIR": "C:/photos",
            "DATA_DIR": "C:/data",
            "LLM_API_KEY": "llm-key",
            "LLM_BASE_URL": "https://www.su8.codes/codex/v1",
            "SU8_API_KEY": "su8-key",
            "SU8_BASE_URL": "https://www.su8.codes/codex/v1",
            "VISION_API_KEY": "vision-key",
            "VISION_BASE_URL": "https://api.moonshot.cn/v1",
            "VISION_MODEL": "gpt-5.4",
            "VISION_REASONING_EFFORT": "medium",
            "VISION_ENHANCED_REASONING_EFFORT": "low",
            "VISION_BASE_MAX_TOKENS": 700,
            "VISION_ENHANCED_MAX_TOKENS": 420,
            "VISION_REPAIR_MAX_TOKENS": 420,
            "STRUCTURED_ANALYSIS_ENABLED": True,
            "ENHANCED_ANALYSIS_ENABLED": True,
            "TAG_MIN_CONFIDENCE": 0.65,
            "IDENTITY_TEXT_MIN_CONFIDENCE": 0.7,
            "IDENTITY_VISUAL_MIN_CONFIDENCE": 0.92,
            "TIME_PARSE_MODEL": "gpt-5.1",
            "TIME_PARSE_REASONING_EFFORT": "low",
            "TIME_PARSE_API_KEY": "time-key",
            "TIME_PARSE_BASE_URL": "https://time.example/v1",
            "EMBEDDING_API_KEY": "embed-key",
            "EMBEDDING_BASE_URL": "https://router.tumuer.me/v1",
            "EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-8B",
            "EMBEDDING_DIMENSION": 4096,
            "TEXT_RERANK_API_KEY": "rerank-key",
            "TEXT_RERANK_BASE_URL": "https://router.tumuer.me/v1",
            "TEXT_RERANK_MODEL": "Qwen/Qwen3-Reranker-8B",
            "VISUAL_RERANK_ENABLED": True,
            "VISUAL_RERANK_API_KEY": "vision-rerank-key",
            "VISUAL_RERANK_BASE_URL": "https://api.moonshot.cn/v1",
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
            "QUERY_MAX_REFLECTION_ROUNDS": 3,
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

            _, vision_kwargs = vision_cls.call_args
            self.assertEqual(vision_kwargs["api_key"], "vision-key")
            self.assertEqual(vision_kwargs["base_url"], "https://api.moonshot.cn/v1")
            _, time_parser_kwargs = time_parser_cls.call_args
            self.assertEqual(time_parser_kwargs["api_key"], "time-key")
            self.assertEqual(time_parser_kwargs["base_url"], "https://time.example/v1")
            _, visual_rerank_kwargs = visual_rerank_cls.call_args
            self.assertEqual(visual_rerank_kwargs["api_key"], "vision-rerank-key")
            self.assertEqual(visual_rerank_kwargs["base_url"], "https://api.moonshot.cn/v1")
            vector_store_cls.assert_called_once_with(
                dimension=4096,
                index_path="C:/data/photo_search.index",
                metadata_path="C:/data/metadata.json",
                metric="cosine",
                index_type="flat",
                hnsw_m=32,
                hnsw_ef_construction=200,
                hnsw_ef_search=96,
            )
            embedding_cls.assert_called_once()
            vision_cls.assert_called_once()
            time_parser_cls.assert_called_once()
            text_rerank_cls.assert_called_once()
            indexer_cls.assert_called_once()
            searcher_cls.assert_called_once()
            _, indexer_kwargs = indexer_cls.call_args
            self.assertEqual(indexer_kwargs["background_mode"], "process")
            self.assertEqual(indexer_kwargs["worker_python_executable"], sys.executable)
            self.assertTrue(str(indexer_kwargs["worker_entrypoint"]).endswith("main.py"))
            self.assertEqual(result, (indexer, searcher, text_rerank_service, visual_rerank_service))

    def test_initialize_services_falls_back_to_su8_base_url_for_query_formatter(self) -> None:
        config = {
            "PHOTO_DIR": "C:/photos",
            "DATA_DIR": "C:/data",
            "SU8_API_KEY": "su8-key",
            "SU8_BASE_URL": "https://www.su8.codes/codex/v1",
            "VISION_MODEL": "gpt-5.4",
            "VISION_REASONING_EFFORT": "medium",
            "VISION_ENHANCED_REASONING_EFFORT": "low",
            "VISION_BASE_MAX_TOKENS": 700,
            "VISION_ENHANCED_MAX_TOKENS": 420,
            "VISION_REPAIR_MAX_TOKENS": 420,
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

    def test_initialize_services_uses_llm_fallback_for_time_parse(self) -> None:
        config = {
            "PHOTO_DIR": "C:/photos",
            "DATA_DIR": "C:/data",
            "LLM_API_KEY": "generic-key",
            "LLM_BASE_URL": "https://llm.example/v1",
            "VISION_MODEL": "gpt-5.4",
            "TIME_PARSE_MODEL": "gpt-5.1",
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
            patch("main.normalize_local_path", return_value="/mnt/c/photos"):

            vector_store_cls.return_value = MagicMock()
            vision_cls.return_value = MagicMock()
            embedding_cls.return_value = MagicMock()
            time_parser_cls.return_value = MagicMock()
            indexer_cls.return_value = MagicMock()
            searcher_cls.return_value = MagicMock()

            main.initialize_services(config)

            _, kwargs = time_parser_cls.call_args
            self.assertEqual(kwargs["api_key"], "generic-key")
            self.assertEqual(kwargs["base_url"], "https://llm.example/v1")

    def test_initialize_services_enables_local_ollama_optional_services_without_keys(self) -> None:
        config = {
            "PHOTO_DIR": "C:/photos",
            "DATA_DIR": "C:/data",
            "LLM_API_KEY": "",
            "LLM_BASE_URL": "http://localhost:11434",
            "SU8_API_KEY": "",
            "SU8_BASE_URL": "http://localhost:11434",
            "VISION_API_KEY": "",
            "VISION_BASE_URL": "",
            "VISION_MODEL": "qwen2.5vl:7b",
            "VISION_REASONING_EFFORT": "medium",
            "VISION_ENHANCED_REASONING_EFFORT": "low",
            "VISION_BASE_MAX_TOKENS": 700,
            "VISION_ENHANCED_MAX_TOKENS": 420,
            "VISION_REPAIR_MAX_TOKENS": 420,
            "STRUCTURED_ANALYSIS_ENABLED": True,
            "ENHANCED_ANALYSIS_ENABLED": True,
            "TAG_MIN_CONFIDENCE": 0.65,
            "IDENTITY_TEXT_MIN_CONFIDENCE": 0.7,
            "IDENTITY_VISUAL_MIN_CONFIDENCE": 0.92,
            "TIME_PARSE_MODEL": "qwen2.5:7b-instruct",
            "TIME_PARSE_REASONING_EFFORT": "low",
            "QUERY_FORMAT_ENABLED": True,
            "QUERY_FORMAT_API_KEY": "",
            "QUERY_FORMAT_BASE_URL": "",
            "QUERY_FORMAT_MODEL": "qwen2.5:7b-instruct",
            "QUERY_FORMAT_REASONING_EFFORT": "low",
            "EMBEDDING_API_KEY": "",
            "EMBEDDING_BASE_URL": "http://localhost:11434",
            "EMBEDDING_MODEL": "nomic-embed-text",
            "EMBEDDING_DIMENSION": 768,
            "TEXT_RERANK_API_KEY": "",
            "TEXT_RERANK_BASE_URL": "http://localhost:11434",
            "TEXT_RERANK_MODEL": "qwen3-reranker:8b",
            "TEXT_RERANK_BACKEND": "chat",
            "VISUAL_RERANK_ENABLED": True,
            "VISUAL_RERANK_API_KEY": "",
            "VISUAL_RERANK_BASE_URL": "",
            "VISUAL_RERANK_MODEL": "qwen2.5vl:7b",
            "BATCH_SIZE": 4,
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
            "ELASTICSEARCH_HOST": "",
        }

        with patch.object(main, "VectorStore") as vector_store_cls, \
            patch.object(main, "SU8VisionLLMService") as vision_cls, \
            patch.object(main, "TumuerEmbeddingService") as embedding_cls, \
            patch.object(main, "TimeParser") as time_parser_cls, \
            patch.object(main, "Indexer") as indexer_cls, \
            patch.object(main, "Searcher") as searcher_cls, \
            patch("main.normalize_local_path", return_value="/mnt/c/photos"), \
            patch("utils.query_formatter.QueryFormatter") as query_formatter_cls, \
            patch.object(main, "TextRerankService") as text_rerank_cls, \
            patch("utils.rerank_service.VisualRerankService") as visual_rerank_cls:

            vector_store_cls.return_value = MagicMock()
            vision_cls.return_value = MagicMock()
            embedding_cls.return_value = MagicMock()
            time_parser_cls.return_value = MagicMock()
            indexer_cls.return_value = MagicMock()
            searcher_cls.return_value = MagicMock()
            query_formatter_cls.return_value = MagicMock()
            text_rerank_cls.return_value = MagicMock()
            visual_rerank_cls.return_value = MagicMock()

            _, _, text_rerank_service, visual_rerank_service = main.initialize_services(config)

            query_formatter_cls.assert_called_once()
            text_rerank_cls.assert_called_once()
            visual_rerank_cls.assert_called_once()
            self.assertIsNotNone(text_rerank_service)
            self.assertIsNotNone(visual_rerank_service)

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
            patch.object(main, "create_app", return_value=mock_app), \
            patch.object(main, "_can_bind", return_value=True):
            main.main()

        mock_app.run.assert_called_once_with(host="127.0.0.1", port=5001, debug=False)

    def test_run_index_worker_uses_lock_already_held(self) -> None:
        config = {
            "PHOTO_DIR": "C:/photos",
            "SU8_API_KEY": "key",
            "EMBEDDING_API_KEY": "emb",
        }
        indexer = MagicMock()
        indexer.build_index.return_value = {"status": "success"}

        with patch.object(main, "load_config", return_value=config), \
            patch.object(main, "initialize_services", return_value=(indexer, MagicMock(), None, None)):
            exit_code = main.run_index_worker(force_rebuild=True)

        indexer.build_index.assert_called_once_with(force_rebuild=True, lock_already_held=True)
        self.assertEqual(exit_code, 0)

    def test_main_switches_to_fallback_port_before_start_when_port_unavailable(self) -> None:
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
            patch.object(main, "create_app", return_value=mock_app), \
            patch.object(main, "_can_bind", return_value=False), \
            patch.object(main, "_find_available_port", return_value=5010):
            main.main()

        mock_app.run.assert_called_once_with(host="127.0.0.1", port=5010, debug=False)

    def test_main_retries_with_fallback_port_on_bind_error(self) -> None:
        config = {
            "PHOTO_DIR": "C:/photos",
            "SU8_API_KEY": "key",
            "EMBEDDING_API_KEY": "emb",
            "SERVER_HOST": "127.0.0.1",
            "SERVER_PORT": 5001,
        }
        mock_app = MagicMock()
        bind_error = OSError("access denied")
        bind_error.winerror = 10013
        mock_app.run.side_effect = [bind_error, None]

        with patch.object(main, "load_config", return_value=config), \
            patch.object(main, "initialize_services", return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock())), \
            patch.object(main, "create_app", return_value=mock_app), \
            patch.object(main, "_can_bind", return_value=True), \
            patch.object(main, "_find_available_port", return_value=5010):
            main.main()

        self.assertEqual(
            mock_app.run.call_args_list,
            [
                call(host="127.0.0.1", port=5001, debug=False),
                call(host="127.0.0.1", port=5010, debug=False),
            ],
        )

    def test_validate_required_config_accepts_generic_llm_key(self) -> None:
        main._validate_required_config(
            {
                "PHOTO_DIR": "C:/photos",
                "LLM_API_KEY": "generic-key",
                "EMBEDDING_API_KEY": "emb",
            }
        )


if __name__ == "__main__":
    unittest.main()
