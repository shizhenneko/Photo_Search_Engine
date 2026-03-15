from __future__ import annotations

import os
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional at runtime
    load_dotenv = None

_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数") from exc


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是数字") from exc


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def load_config() -> Dict[str, Any]:
    """从环境变量加载运行配置。"""
    if load_dotenv is not None:
        load_dotenv()

    data_dir = os.getenv("DATA_DIR", "./data")
    runtime_data_dir = os.getenv("RUNTIME_DATA_DIR", data_dir)
    llm_api_key = (
        os.getenv("LLM_API_KEY")
        or os.getenv("SU8_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    llm_base_url = os.getenv("LLM_BASE_URL") or os.getenv("SU8_BASE_URL", "https://www.su8.codes/codex/v1")

    vision_api_key = os.getenv("VISION_API_KEY") or llm_api_key
    vision_base_url = os.getenv("VISION_BASE_URL") or llm_base_url
    time_parse_api_key = os.getenv("TIME_PARSE_API_KEY") or llm_api_key
    time_parse_base_url = os.getenv("TIME_PARSE_BASE_URL") or llm_base_url
    visual_rerank_api_key = os.getenv("VISUAL_RERANK_API_KEY") or vision_api_key
    visual_rerank_base_url = os.getenv("VISUAL_RERANK_BASE_URL") or vision_base_url
    query_format_api_key = os.getenv("QUERY_FORMAT_API_KEY") or llm_api_key
    query_format_base_url = os.getenv("QUERY_FORMAT_BASE_URL") or llm_base_url
    embedding_api_key = os.getenv("EMBEDDING_API_KEY") or llm_api_key
    embedding_base_url = os.getenv("EMBEDDING_BASE_URL") or "https://router.tumuer.me/v1"
    text_rerank_api_key = os.getenv("TEXT_RERANK_API_KEY") or embedding_api_key
    text_rerank_base_url = os.getenv("TEXT_RERANK_BASE_URL") or embedding_base_url
    query_max_expansion_rounds = _get_int(
        "QUERY_MAX_EXPANSION_ROUNDS",
        _get_int("QUERY_EXPANSION_MAX_ALTERNATIVES", 2),
    )
    query_max_reflection_rounds = _get_int("QUERY_MAX_REFLECTION_ROUNDS", 2)

    config: Dict[str, Any] = {
        "PHOTO_DIR": os.getenv("PHOTO_DIR"),
        "DATA_DIR": data_dir,
        "RUNTIME_DATA_DIR": runtime_data_dir,
        "INDEX_PATH": os.getenv("INDEX_PATH", os.path.join(runtime_data_dir, "photo_search.index")),
        "METADATA_PATH": os.getenv("METADATA_PATH", os.path.join(runtime_data_dir, "metadata.json")),
        "VECTOR_METRIC": os.getenv("VECTOR_METRIC", "cosine"),
        "VECTOR_INDEX_TYPE": os.getenv("VECTOR_INDEX_TYPE", "flat"),
        "HNSW_M": _get_int("HNSW_M", 32),
        "HNSW_EF_CONSTRUCTION": _get_int("HNSW_EF_CONSTRUCTION", 200),
        "HNSW_EF_SEARCH": _get_int("HNSW_EF_SEARCH", 96),
        "VECTOR_WEIGHT": _get_float("VECTOR_WEIGHT", 0.8),
        "KEYWORD_WEIGHT": _get_float("KEYWORD_WEIGHT", 0.2),
        "TOP_K": _get_int("TOP_K", 12),
        "BATCH_SIZE": _get_int("BATCH_SIZE", 8),
        "MAX_RETRIES": _get_int("MAX_RETRIES", 3),
        "TIMEOUT": _get_int("TIMEOUT", 45),
        "INDEX_BACKGROUND_MODE": os.getenv("INDEX_BACKGROUND_MODE", "process").strip().lower(),
        "SERVER_HOST": os.getenv("SERVER_HOST", "127.0.0.1"),
        "SERVER_PORT": _get_int("SERVER_PORT", 10001),
        "SECRET_KEY": os.getenv("SECRET_KEY", "dev-secret-key"),
        "USE_BASE64": _get_bool("USE_BASE64", True),
        "IMAGE_MAX_SIZE": _get_int("IMAGE_MAX_SIZE", 1024),
        "IMAGE_QUALITY": _get_int("IMAGE_QUALITY", 85),
        "IMAGE_FORMAT": os.getenv("IMAGE_FORMAT", "WEBP").upper(),
        "LLM_API_KEY": llm_api_key,
        "LLM_BASE_URL": llm_base_url,
        "SU8_API_KEY": llm_api_key,
        "SU8_BASE_URL": llm_base_url,
        "VISION_API_KEY": vision_api_key,
        "VISION_BASE_URL": vision_base_url,
        "VISION_MODEL": os.getenv("VISION_MODEL", "gpt-5.4"),
        "VISION_REASONING_EFFORT": os.getenv("VISION_REASONING_EFFORT", "medium"),
        "VISION_ENHANCED_REASONING_EFFORT": os.getenv("VISION_ENHANCED_REASONING_EFFORT", "low"),
        "VISION_BASE_MAX_TOKENS": _get_int("VISION_BASE_MAX_TOKENS", 700),
        "VISION_ENHANCED_MAX_TOKENS": _get_int("VISION_ENHANCED_MAX_TOKENS", 420),
        "VISION_REPAIR_MAX_TOKENS": _get_int("VISION_REPAIR_MAX_TOKENS", 420),
        "STRUCTURED_ANALYSIS_ENABLED": _get_bool("STRUCTURED_ANALYSIS_ENABLED", True),
        "ENHANCED_ANALYSIS_ENABLED": _get_bool("ENHANCED_ANALYSIS_ENABLED", True),
        "TAG_MIN_CONFIDENCE": _get_float("TAG_MIN_CONFIDENCE", 0.65),
        "IDENTITY_TEXT_MIN_CONFIDENCE": _get_float("IDENTITY_TEXT_MIN_CONFIDENCE", 0.7),
        "IDENTITY_VISUAL_MIN_CONFIDENCE": _get_float("IDENTITY_VISUAL_MIN_CONFIDENCE", 0.92),
        "TIME_PARSE_API_KEY": time_parse_api_key,
        "TIME_PARSE_BASE_URL": time_parse_base_url,
        "TIME_PARSE_MODEL": os.getenv("TIME_PARSE_MODEL", "gpt-5.1"),
        "TIME_PARSE_REASONING_EFFORT": os.getenv("TIME_PARSE_REASONING_EFFORT", "low"),
        "TIME_PARSE_STRATEGY": os.getenv("TIME_PARSE_STRATEGY", "local_first"),
        "QUERY_FORMAT_ENABLED": _get_bool("QUERY_FORMAT_ENABLED", True),
        "QUERY_FORMAT_API_KEY": query_format_api_key,
        "QUERY_FORMAT_BASE_URL": query_format_base_url,
        "QUERY_FORMAT_MODEL": os.getenv("QUERY_FORMAT_MODEL", "gpt-5.1"),
        "QUERY_FORMAT_REASONING_EFFORT": os.getenv("QUERY_FORMAT_REASONING_EFFORT", "low"),
        "QUERY_EXPANSION_ENABLED": _get_bool("QUERY_EXPANSION_ENABLED", True),
        "QUERY_EXPANSION_MAX_ALTERNATIVES": query_max_expansion_rounds,
        "QUERY_MAX_EXPANSION_ROUNDS": query_max_expansion_rounds,
        "QUERY_MULTI_ROUND_ENABLED": _get_bool("QUERY_MULTI_ROUND_ENABLED", False),
        "QUERY_REFLECTION_ENABLED": _get_bool("QUERY_REFLECTION_ENABLED", False),
        "QUERY_MAX_REFLECTION_ROUNDS": query_max_reflection_rounds,
        "QUERY_DYNAMIC_THRESHOLD_FLOOR": _get_float("QUERY_DYNAMIC_THRESHOLD_FLOOR", 0.05),
        "QUERY_STRICT_FLOOR_MIN": _get_float("QUERY_STRICT_FLOOR_MIN", 0.22),
        "QUERY_BROAD_FLOOR_MIN": _get_float("QUERY_BROAD_FLOOR_MIN", 0.12),
        "QUERY_CACHE_ENABLED": _get_bool("QUERY_CACHE_ENABLED", True),
        "QUERY_CACHE_SIZE": _get_int("QUERY_CACHE_SIZE", 2000),
        "EMBEDDING_CACHE_ENABLED": _get_bool("EMBEDDING_CACHE_ENABLED", True),
        "EMBEDDING_CACHE_SIZE": _get_int("EMBEDDING_CACHE_SIZE", 5000),
        "DISK_CACHE_ENABLED": _get_bool("DISK_CACHE_ENABLED", False),
        "EMBEDDING_API_KEY": embedding_api_key,
        "EMBEDDING_BASE_URL": embedding_base_url,
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B"),
        "EMBEDDING_DIMENSION": _get_int("EMBEDDING_DIMENSION", 4096),
        "TEXT_RERANK_API_KEY": text_rerank_api_key,
        "TEXT_RERANK_BASE_URL": text_rerank_base_url,
        "TEXT_RERANK_MODEL": os.getenv("TEXT_RERANK_MODEL", "Qwen/Qwen3-Reranker-8B"),
        "TEXT_RERANK_BACKEND": os.getenv("TEXT_RERANK_BACKEND", "auto"),
        "TEXT_RERANK_TIMEOUT": _get_int("TEXT_RERANK_TIMEOUT", 45),
        "VISUAL_RERANK_ENABLED": _get_bool("VISUAL_RERANK_ENABLED", True),
        "VISUAL_RERANK_API_KEY": visual_rerank_api_key,
        "VISUAL_RERANK_BASE_URL": visual_rerank_base_url,
        "VISUAL_RERANK_MODEL": os.getenv("VISUAL_RERANK_MODEL", os.getenv("VISION_MODEL", "gpt-5.4")),
        "VISUAL_RERANK_REASONING_EFFORT": os.getenv("VISUAL_RERANK_REASONING_EFFORT", "medium"),
        "VISUAL_RERANK_TIMEOUT": _get_int("VISUAL_RERANK_TIMEOUT", 60),
        "RERANK_IMAGE_MAX_SIZE": _get_int("RERANK_IMAGE_MAX_SIZE", 512),
        "RERANK_IMAGE_QUALITY": _get_int("RERANK_IMAGE_QUALITY", 75),
        "RERANK_IMAGE_FORMAT": os.getenv("RERANK_IMAGE_FORMAT", "WEBP").upper(),
        "RERANK_MAX_IMAGES": _get_int("RERANK_MAX_IMAGES", 12),
        "ELASTICSEARCH_HOST": os.getenv("ELASTICSEARCH_HOST", "localhost"),
        "ELASTICSEARCH_PORT": _get_int("ELASTICSEARCH_PORT", 9200),
        "ELASTICSEARCH_INDEX": os.getenv("ELASTICSEARCH_INDEX", "photo_keywords"),
        "ELASTICSEARCH_USERNAME": os.getenv("ELASTICSEARCH_USERNAME"),
        "ELASTICSEARCH_PASSWORD": os.getenv("ELASTICSEARCH_PASSWORD"),
        "SEARCH_VALIDATE_FILE_EXISTS": _get_bool("SEARCH_VALIDATE_FILE_EXISTS", False),
        "DEFAULT_SEARCH_MODE": os.getenv("DEFAULT_SEARCH_MODE", "balanced").strip().lower(),
    }

    return config


def get_config() -> Dict[str, Any]:
    """获取缓存后的配置。"""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = load_config()
    return _CONFIG_CACHE
