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

    vector_store = VectorStore(
        dimension=int(config.get("EMBEDDING_DIMENSION", 768)),
        index_path=str(config.get("INDEX_PATH", os.path.join(data_dir, "photo_search.index"))),
        metadata_path=str(config.get("METADATA_PATH", os.path.join(data_dir, "metadata.json"))),
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

    embedding_service = T5EmbeddingService(
        model_name=str(config.get("EMBEDDING_MODEL_NAME", "sentence-t5-base"))
    )

    time_parser = TimeParser(
        api_key=str(config.get("OPENROUTER_API_KEY", "")),
        model_name=str(config.get("TIME_PARSE_MODEL_NAME", "openai/gpt-3.5-turbo")),
        base_url=str(config.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")),
        timeout=int(config.get("TIMEOUT", 30)),
        max_retries=int(config.get("MAX_RETRIES", 3)),
    )

    indexer = Indexer(
        photo_dir=str(config.get("PHOTO_DIR", "")),
        vision=vision_service,
        embedding=embedding_service,
        vector_store=vector_store,
        data_dir=data_dir,
        batch_size=int(config.get("BATCH_SIZE", 10)),
        max_retries=int(config.get("MAX_RETRIES", 3)),
        timeout=int(config.get("TIMEOUT", 30)),
    )

    searcher = Searcher(
        embedding=embedding_service,
        time_parser=time_parser,
        vector_store=vector_store,
        data_dir=data_dir,
        top_k=int(config.get("TOP_K", 10)),
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
