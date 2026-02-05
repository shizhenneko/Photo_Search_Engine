from __future__ import annotations

import os
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency at runtime
    load_dotenv = None

_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数") from exc


def load_config() -> Dict[str, Any]:
    """
    加载配置（从环境变量读取）。

    Returns:
        Dict[str, Any]: 配置字典
    """
    if load_dotenv is not None:
        load_dotenv()

    data_dir = os.getenv("DATA_DIR", "./data")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY") or openrouter_api_key

    config: Dict[str, Any] = {
        "PHOTO_DIR": os.getenv("PHOTO_DIR"),
        "DATA_DIR": data_dir,
        "OPENROUTER_API_KEY": openrouter_api_key,
        "OPENAI_API_KEY": openai_api_key,
        "OPENROUTER_BASE_URL": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "VISION_MODEL_NAME": os.getenv("VISION_MODEL_NAME", "openai/gpt-4o"),
        "EMBEDDING_MODEL_NAME": os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5"),
        "EMBEDDING_DIMENSION": _get_int("EMBEDDING_DIMENSION", 4096),

        # 阿里百炼 Embedding 配置
        "EMBEDDING_API_KEY": os.getenv("EMBEDDING_API_KEY"),
        "EMBEDDING_BASE_URL": os.getenv("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),

        "TIME_PARSE_MODEL_NAME": os.getenv("TIME_PARSE_MODEL_NAME", "openai/gpt-3.5-turbo"),
        "SERVER_HOST": os.getenv("SERVER_HOST", "localhost"),

        # 火山引擎 Embedding 配置
        "VOLCANO_API_KEY": os.getenv("VOLCANO_API_KEY"),
        "VOLCANO_BASE_URL": os.getenv("VOLCANO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        "VOLCANO_EMBEDDING_MODEL": os.getenv("VOLCANO_EMBEDDING_MODEL", "doubao-embedding-large-text-240915"),

        # Elasticsearch 配置
        "ELASTICSEARCH_HOST": os.getenv("ELASTICSEARCH_HOST", "localhost"),
        "ELASTICSEARCH_PORT": _get_int("ELASTICSEARCH_PORT", 9200),
        "ELASTICSEARCH_INDEX": os.getenv("ELASTICSEARCH_INDEX", "photo_keywords"),
        "ELASTICSEARCH_USERNAME": os.getenv("ELASTICSEARCH_USERNAME"),
        "ELASTICSEARCH_PASSWORD": os.getenv("ELASTICSEARCH_PASSWORD"),

        # 混合检索权重
        "VECTOR_WEIGHT": float(os.getenv("VECTOR_WEIGHT", 0.8)),
        "KEYWORD_WEIGHT": float(os.getenv("KEYWORD_WEIGHT", 0.2)),

        # 查询格式化 LLM 配置
        "QUERY_FORMAT_API_KEY": os.getenv("QUERY_FORMAT_API_KEY"),
        "QUERY_FORMAT_BASE_URL": os.getenv("QUERY_FORMAT_BASE_URL"),
        "QUERY_FORMAT_MODEL": os.getenv("QUERY_FORMAT_MODEL"),

        "SERVER_PORT": _get_int("SERVER_PORT", 5000),
        "BATCH_SIZE": _get_int("BATCH_SIZE", 10),
        "MAX_RETRIES": _get_int("MAX_RETRIES", 3),
        "TIMEOUT": _get_int("TIMEOUT", 30),
        "TOP_K": _get_int("TOP_K", 10),
        "INDEX_PATH": os.getenv("INDEX_PATH", os.path.join(data_dir, "photo_search.index")),
        "METADATA_PATH": os.getenv("METADATA_PATH", os.path.join(data_dir, "metadata.json")),
        "VECTOR_METRIC": os.getenv("VECTOR_METRIC", "cosine"),  # 确保默认使用余弦相似度，适合中文语义检索
        "SECRET_KEY": os.getenv("SECRET_KEY", "dev-secret-key"),
        "USE_BASE64": os.getenv("USE_BASE64", "true").lower() in ("true", "1", "yes"),
        "IMAGE_MAX_SIZE": _get_int("IMAGE_MAX_SIZE", 1024),
        "IMAGE_QUALITY": _get_int("IMAGE_QUALITY", 85),
        "IMAGE_FORMAT": os.getenv("IMAGE_FORMAT", "WEBP").upper(),

        # Rerank 配置
        "RERANK_ENABLED": os.getenv("RERANK_ENABLED", "true").lower() in ("true", "1", "yes"),
        "RERANK_MODEL_NAME": os.getenv("RERANK_MODEL_NAME"),  # 默认使用VISION_MODEL_NAME
        "RERANK_IMAGE_MAX_SIZE": _get_int("RERANK_IMAGE_MAX_SIZE", 512),
        "RERANK_IMAGE_QUALITY": _get_int("RERANK_IMAGE_QUALITY", 75),
        "RERANK_MAX_IMAGES": _get_int("RERANK_MAX_IMAGES", 10),
        "RERANK_TIMEOUT": _get_int("RERANK_TIMEOUT", 60),
    }

    # RERANK_MODEL_NAME 默认使用 VISION_MODEL_NAME
    if not config["RERANK_MODEL_NAME"]:
        config["RERANK_MODEL_NAME"] = config["VISION_MODEL_NAME"]

    return config


def get_config() -> Dict[str, Any]:
    """
    获取配置单例。

    Returns:
        Dict[str, Any]: 配置字典
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = load_config()
    return _CONFIG_CACHE
