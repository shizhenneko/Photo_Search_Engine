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
        "EMBEDDING_DIMENSION": _get_int("EMBEDDING_DIMENSION", 512),
        "TIME_PARSE_MODEL_NAME": os.getenv("TIME_PARSE_MODEL_NAME", "openai/gpt-3.5-turbo"),
        "SERVER_HOST": os.getenv("SERVER_HOST", "localhost"),
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
    }

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
