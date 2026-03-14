from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:
    from utils.embedding_service import TextRerankService
    from utils.rerank_service import VisualRerankService

from flask import Flask

from api.routes import register_routes
from config import get_config
from core.indexer import Indexer
from core.searcher import Searcher
from utils.embedding_service import TextRerankService, TumuerEmbeddingService
from utils.path_utils import normalize_local_path
from utils.time_parser import TimeParser
from utils.vector_store import VectorStore
from utils.vision_llm_service import SU8VisionLLMService


def load_config() -> Dict[str, object]:
    return get_config()


def initialize_services(
    config: Dict[str, object],
) -> Tuple[Indexer, Searcher, Optional["TextRerankService"], Optional["VisualRerankService"]]:
    data_dir = str(config.get("DATA_DIR", "./data"))
    os.makedirs(data_dir, exist_ok=True)

    embedding_service = TumuerEmbeddingService(
        api_key=str(config.get("EMBEDDING_API_KEY", "")),
        model_name=str(config.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")),
        base_url=str(config.get("EMBEDDING_BASE_URL", "https://router.tumuer.me/v1")),
        timeout=int(config.get("TIMEOUT", 45)),
        max_retries=int(config.get("MAX_RETRIES", 3)),
        dimension=int(config.get("EMBEDDING_DIMENSION", 4096)),
    )

    vector_store = VectorStore(
        dimension=int(config.get("EMBEDDING_DIMENSION", 4096)),
        index_path=str(config.get("INDEX_PATH", os.path.join(data_dir, "photo_search.index"))),
        metadata_path=str(config.get("METADATA_PATH", os.path.join(data_dir, "metadata.json"))),
        metric=str(config.get("VECTOR_METRIC", "cosine")),
    )

    vision_service = SU8VisionLLMService(
        api_key=str(config.get("SU8_API_KEY", "")),
        model_name=str(config.get("VISION_MODEL", "gpt-5.4")),
        base_url=str(config.get("SU8_BASE_URL", "https://www.su8.codes/codex/v1")),
        reasoning_effort=str(config.get("VISION_REASONING_EFFORT", "medium")),
        timeout=int(config.get("TIMEOUT", 45)),
        max_retries=int(config.get("MAX_RETRIES", 3)),
        use_base64=bool(config.get("USE_BASE64", True)),
        image_max_size=int(config.get("IMAGE_MAX_SIZE", 1024)),
        image_quality=int(config.get("IMAGE_QUALITY", 85)),
        image_format=str(config.get("IMAGE_FORMAT", "WEBP")),
    )
    enhanced_analysis_enabled = bool(config.get("ENHANCED_ANALYSIS_ENABLED", True))
    setattr(vision_service, "enhanced_analysis_enabled", enhanced_analysis_enabled)

    time_parser = TimeParser(
        api_key=str(config.get("SU8_API_KEY", "")),
        model_name=str(config.get("TIME_PARSE_MODEL", "gpt-5.1")),
        base_url=str(config.get("SU8_BASE_URL", "https://www.su8.codes/codex/v1")),
        reasoning_effort=str(config.get("TIME_PARSE_REASONING_EFFORT", "low")),
        timeout=int(config.get("TIMEOUT", 45)),
        max_retries=int(config.get("MAX_RETRIES", 3)),
    )

    keyword_store = None
    if config.get("ELASTICSEARCH_HOST"):
        try:
            from utils.keyword_store import KeywordStore

            keyword_store = KeywordStore(
                host=str(config.get("ELASTICSEARCH_HOST")),
                port=int(config.get("ELASTICSEARCH_PORT", 9200)),
                index_name=str(config.get("ELASTICSEARCH_INDEX", "photo_keywords")),
                username=str(config.get("ELASTICSEARCH_USERNAME")) if config.get("ELASTICSEARCH_USERNAME") else None,
                password=str(config.get("ELASTICSEARCH_PASSWORD")) if config.get("ELASTICSEARCH_PASSWORD") else None,
            )
        except Exception as exc:
            print(f"Warning: Failed to initialize Elasticsearch: {exc}. Keyword search will be disabled.")

    query_formatter = None
    if config.get("QUERY_FORMAT_ENABLED", True) and config.get("QUERY_FORMAT_API_KEY"):
        try:
            from utils.query_formatter import QueryFormatter

            query_formatter = QueryFormatter(
                api_key=str(config.get("QUERY_FORMAT_API_KEY", "")),
                model_name=str(config.get("QUERY_FORMAT_MODEL", "gpt-5.1")),
                base_url=str(
                    config.get("QUERY_FORMAT_BASE_URL")
                    or config.get("SU8_BASE_URL", "https://www.su8.codes/codex/v1")
                ),
                reasoning_effort=str(config.get("QUERY_FORMAT_REASONING_EFFORT", "low")),
                timeout=int(config.get("TIMEOUT", 45)),
                max_retries=int(config.get("MAX_RETRIES", 3)),
            )
        except Exception as exc:
            print(f"Warning: Failed to initialize QueryFormatter: {exc}. Query formatting will be disabled.")

    indexer = Indexer(
        photo_dir=normalize_local_path(str(config.get("PHOTO_DIR", ""))),
        vision=vision_service,
        embedding=embedding_service,
        vector_store=vector_store,
        keyword_store=keyword_store,
        data_dir=data_dir,
        batch_size=int(config.get("BATCH_SIZE", 8)),
        max_retries=int(config.get("MAX_RETRIES", 3)),
        timeout=int(config.get("TIMEOUT", 45)),
    )

    searcher = Searcher(
        embedding=embedding_service,
        time_parser=time_parser,
        vector_store=vector_store,
        keyword_store=keyword_store,
        query_formatter=query_formatter,
        data_dir=data_dir,
        top_k=int(config.get("TOP_K", 12)),
        vector_weight=float(config.get("VECTOR_WEIGHT", 0.8)),
        keyword_weight=float(config.get("KEYWORD_WEIGHT", 0.2)),
        query_expansion_enabled=bool(config.get("QUERY_EXPANSION_ENABLED", True)),
        query_expansion_max_alternatives=int(config.get("QUERY_EXPANSION_MAX_ALTERNATIVES", 2)),
    )

    text_rerank_service: Optional[TextRerankService] = None
    if config.get("TEXT_RERANK_API_KEY"):
        try:
            text_rerank_service = TextRerankService(
                api_key=str(config.get("TEXT_RERANK_API_KEY", "")),
                model_name=str(config.get("TEXT_RERANK_MODEL", "Qwen/Qwen3-Reranker-8B")),
                base_url=str(config.get("TEXT_RERANK_BASE_URL", "https://router.tumuer.me/v1")),
                timeout=int(config.get("TEXT_RERANK_TIMEOUT", 45)),
                max_retries=int(config.get("MAX_RETRIES", 3)),
            )
        except Exception as exc:
            print(f"Warning: Failed to initialize text rerank service: {exc}")

    visual_rerank_service = None
    if config.get("VISUAL_RERANK_ENABLED", True) and config.get("SU8_API_KEY"):
        try:
            from utils.rerank_service import VisualRerankService

            visual_rerank_service = VisualRerankService(
                api_key=str(config.get("SU8_API_KEY", "")),
                model_name=str(config.get("VISUAL_RERANK_MODEL", config.get("VISION_MODEL", "gpt-5.4"))),
                base_url=str(config.get("SU8_BASE_URL", "https://www.su8.codes/codex/v1")),
                reasoning_effort=str(config.get("VISUAL_RERANK_REASONING_EFFORT", "medium")),
                timeout=int(config.get("VISUAL_RERANK_TIMEOUT", 60)),
                max_retries=int(config.get("MAX_RETRIES", 3)),
                image_max_size=int(config.get("RERANK_IMAGE_MAX_SIZE", 512)),
                image_quality=int(config.get("RERANK_IMAGE_QUALITY", 75)),
                image_format=str(config.get("RERANK_IMAGE_FORMAT", "WEBP")),
                max_images=int(config.get("RERANK_MAX_IMAGES", 12)),
            )
        except Exception as exc:
            print(f"Warning: Failed to initialize visual rerank service: {exc}")

    return indexer, searcher, text_rerank_service, visual_rerank_service


def create_app(
    indexer: Indexer,
    searcher: Searcher,
    config: Dict[str, object],
    text_rerank_service: Optional["TextRerankService"] = None,
    visual_rerank_service: Optional["VisualRerankService"] = None,
) -> Flask:
    app = Flask(__name__)
    app.secret_key = str(config.get("SECRET_KEY", "dev-secret-key"))

    register_routes(
        app,
        indexer,
        searcher,
        config,
        text_rerank_service=text_rerank_service,
        visual_rerank_service=visual_rerank_service,
    )

    @app.errorhandler(404)
    def not_found(error: object) -> tuple[dict, int]:
        return {"status": "error", "message": "接口不存在"}, 404

    @app.errorhandler(500)
    def internal_error(error: object) -> tuple[dict, int]:
        return {"status": "error", "message": "服务器内部错误"}, 500

    return app


def _validate_required_config(config: Dict[str, object]) -> None:
    if not config.get("PHOTO_DIR"):
        raise ValueError("PHOTO_DIR环境变量未设置")
    if not config.get("SU8_API_KEY"):
        raise ValueError("SU8_API_KEY环境变量未设置")
    if not config.get("EMBEDDING_API_KEY"):
        raise ValueError("EMBEDDING_API_KEY环境变量未设置")


def main() -> None:
    config = load_config()
    _validate_required_config(config)

    indexer, searcher, text_rerank_service, visual_rerank_service = initialize_services(config)
    app = create_app(
        indexer,
        searcher,
        config,
        text_rerank_service=text_rerank_service,
        visual_rerank_service=visual_rerank_service,
    )

    host = str(config.get("SERVER_HOST", "127.0.0.1"))
    port = int(config.get("SERVER_PORT", 10001))
    print(f"启动服务器: http://{host}:{port}")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
