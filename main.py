from __future__ import annotations

import os
from typing import Dict, Tuple

from flask import Flask

from api.routes import register_routes
from config import get_config
from core.indexer import Indexer
from core.searcher import Searcher
from utils.embedding_service import T5EmbeddingService
from utils.time_parser import TimeParser
from utils.vector_store import VectorStore
from utils.vision_llm_service import OpenRouterVisionLLMService


def load_config() -> Dict[str, object]:
    """
    加载配置（从config.py读取）。

    Returns:
        Dict[str, object]: 配置字典
    """
    return get_config()


def initialize_services(config: Dict[str, object]) -> Tuple[Indexer, Searcher]:
    """
    初始化所有服务实例。

    Args:
        config: 配置字典

    Returns:
        Tuple[Indexer, Searcher]: 索引构建器与检索器实例
    """
    data_dir = str(config.get("DATA_DIR", "./data"))
    os.makedirs(data_dir, exist_ok=True)

    # 修改：使用火山引擎 Embedding 服务，如果未配置 API Key 则回退到本地模型（或者根据需求报错）
    # 注意：根据BUG_FIX.md，这里我们优先使用VolcanoService
    volcano_api_key = str(config.get("VOLCANO_API_KEY", ""))
    
    if volcano_api_key and volcano_api_key != "None":
        from utils.embedding_service import VolcanoEmbeddingService
        embedding_service = VolcanoEmbeddingService(
            api_key=volcano_api_key,
            model_name=str(config.get("VOLCANO_EMBEDDING_MODEL", "doubao-embedding-large-text-240915")),
            base_url=str(config.get("VOLCANO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")),
            timeout=int(config.get("TIMEOUT", 30)),
            max_retries=int(config.get("MAX_RETRIES", 3)),
        )
    else:
        # Fallback or original
        print("Warning: VOLCANO_API_KEY not set. Falling back to local T5EmbeddingService.")
        embedding_service = T5EmbeddingService(
            model_name=str(config.get("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5"))
        )

    vector_store = VectorStore(
        dimension=int(config.get("EMBEDDING_DIMENSION", 4096)),
        index_path=str(config.get("INDEX_PATH", os.path.join(data_dir, "photo_search.index"))),
        metadata_path=str(config.get("METADATA_PATH", os.path.join(data_dir, "metadata.json"))),
        metric=str(config.get("VECTOR_METRIC", "cosine")),
    )

    vision_service = OpenRouterVisionLLMService(
        api_key=str(config.get("OPENROUTER_API_KEY", "")),
        model_name=str(config.get("VISION_MODEL_NAME", "openai/gpt-4o")),
        base_url=str(config.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")),
        server_host=str(config.get("SERVER_HOST", "localhost")),
        server_port=int(config.get("SERVER_PORT", 5000)),
        timeout=int(config.get("TIMEOUT", 30)),
        max_retries=int(config.get("MAX_RETRIES", 3)),
        use_base64=bool(config.get("USE_BASE64", True)),
        image_max_size=int(config.get("IMAGE_MAX_SIZE", 1024)),
        image_quality=int(config.get("IMAGE_QUALITY", 85)),
        image_format=str(config.get("IMAGE_FORMAT", "WEBP")),
    )

    time_parser = TimeParser(
        api_key=str(config.get("OPENROUTER_API_KEY", "")),
        model_name=str(config.get("TIME_PARSE_MODEL_NAME", "openai/gpt-3.5-turbo")),
        base_url=str(config.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")),
        timeout=int(config.get("TIMEOUT", 30)),
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
        except Exception as e:
            print(f"Warning: Failed to initialize Elasticsearch: {e}. Keyword search will be disabled.")

    query_formatter = None
    qf_api_key = str(config.get("QUERY_FORMAT_API_KEY", ""))
    if qf_api_key and qf_api_key != "None":
        try:
            from utils.query_formatter import QueryFormatter
            query_formatter = QueryFormatter(
                api_key=qf_api_key,
                model_name=str(config.get("QUERY_FORMAT_MODEL", "doubao-pro-32k")),
                base_url=str(config.get("QUERY_FORMAT_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")),
                timeout=int(config.get("TIMEOUT", 30)),
                max_retries=int(config.get("MAX_RETRIES", 3)),
            )
        except Exception as e:
             print(f"Warning: Failed to initialize QueryFormatter: {e}. Query formatting will be disabled.")

    indexer = Indexer(
        photo_dir=str(config.get("PHOTO_DIR", "")),
        vision=vision_service,
        embedding=embedding_service,
        vector_store=vector_store,
        keyword_store=keyword_store,
        data_dir=data_dir,
        batch_size=int(config.get("BATCH_SIZE", 10)),
        max_retries=int(config.get("MAX_RETRIES", 3)),
        timeout=int(config.get("TIMEOUT", 30)),
    )

    searcher = Searcher(
        embedding=embedding_service,
        time_parser=time_parser,
        vector_store=vector_store,
        keyword_store=keyword_store,
        query_formatter=query_formatter,
        data_dir=data_dir,
        top_k=int(config.get("TOP_K", 10)),
        vector_weight=float(config.get("VECTOR_WEIGHT", 0.8)),
        keyword_weight=float(config.get("KEYWORD_WEIGHT", 0.2)),
    )

    return indexer, searcher


def create_app(indexer: Indexer, searcher: Searcher, config: Dict[str, object]) -> Flask:
    """
    创建并配置Flask应用。

    Args:
        indexer: 索引构建器实例
        searcher: 检索器实例
        config: 配置字典

    Returns:
        Flask: 配置好的Flask应用实例
    """
    app = Flask(__name__)
    app.secret_key = str(config.get("SECRET_KEY", "dev-secret-key"))

    register_routes(app, indexer, searcher, config)

    @app.errorhandler(404)
    def not_found(error: object) -> tuple[dict, int]:
        return {"status": "error", "message": "接口不存在"}, 404

    @app.errorhandler(500)
    def internal_error(error: object) -> tuple[dict, int]:
        return {"status": "error", "message": "服务器内部错误"}, 500

    return app


def _validate_required_config(config: Dict[str, object]) -> None:
    """
    校验必要配置，缺失时抛出异常。

    Args:
        config: 配置字典

    Raises:
        ValueError: 必要配置缺失
    """
    if not config.get("PHOTO_DIR"):
        raise ValueError("PHOTO_DIR环境变量未设置")
    if not config.get("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY环境变量未设置")


def main() -> None:
    """
    主函数：初始化服务并启动Flask应用。
    """
    config = load_config()
    _validate_required_config(config)

    indexer, searcher = initialize_services(config)
    app = create_app(indexer, searcher, config)

    host = str(config.get("SERVER_HOST", "localhost"))
    port = int(config.get("SERVER_PORT", 5000))

    print(f"启动服务器: http://{host}:{port}")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
