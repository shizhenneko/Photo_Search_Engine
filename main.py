from __future__ import annotations

import argparse
import errno
import os
import socket
import sys
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
from utils.llm_compat import requires_api_key
from utils.path_utils import normalize_local_path
from utils.time_parser import TimeParser
from utils.vector_store import VectorStore
from utils.vision_llm_service import SU8VisionLLMService


def load_config() -> Dict[str, object]:
    return get_config()


def _has_usable_api_config(api_key: object, base_url: object) -> bool:
    normalized_api_key = str(api_key or "").strip()
    normalized_base_url = str(base_url or "").strip()
    if normalized_api_key:
        return True
    if not normalized_base_url:
        return False
    return not requires_api_key(normalized_base_url)


def initialize_services(
    config: Dict[str, object],
) -> Tuple[Indexer, Searcher, Optional["TextRerankService"], Optional["VisualRerankService"]]:
    data_dir = str(config.get("DATA_DIR", "./data"))
    runtime_data_dir = str(config.get("RUNTIME_DATA_DIR", data_dir))
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(runtime_data_dir, exist_ok=True)

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
        index_path=str(config.get("INDEX_PATH", os.path.join(runtime_data_dir, "photo_search.index"))),
        metadata_path=str(config.get("METADATA_PATH", os.path.join(runtime_data_dir, "metadata.json"))),
        metric=str(config.get("VECTOR_METRIC", "cosine")),
        index_type=str(config.get("VECTOR_INDEX_TYPE", "flat")),
        hnsw_m=int(config.get("HNSW_M", 32)),
        hnsw_ef_construction=int(config.get("HNSW_EF_CONSTRUCTION", 200)),
        hnsw_ef_search=int(config.get("HNSW_EF_SEARCH", 96)),
    )

    vision_service = SU8VisionLLMService(
        api_key=str(
            config.get("VISION_API_KEY")
            or config.get("LLM_API_KEY")
            or config.get("SU8_API_KEY", "")
        ),
        model_name=str(config.get("VISION_MODEL", "gpt-5.4")),
        base_url=str(
            config.get("VISION_BASE_URL")
            or config.get("LLM_BASE_URL")
            or config.get("SU8_BASE_URL", "https://www.su8.codes/codex/v1")
        ),
        reasoning_effort=str(config.get("VISION_REASONING_EFFORT", "medium")),
        enhanced_reasoning_effort=str(config.get("VISION_ENHANCED_REASONING_EFFORT", "low")),
        timeout=int(config.get("TIMEOUT", 45)),
        max_retries=int(config.get("MAX_RETRIES", 3)),
        use_base64=bool(config.get("USE_BASE64", True)),
        image_max_size=int(config.get("IMAGE_MAX_SIZE", 1024)),
        image_quality=int(config.get("IMAGE_QUALITY", 85)),
        image_format=str(config.get("IMAGE_FORMAT", "WEBP")),
        base_max_output_tokens=int(config.get("VISION_BASE_MAX_TOKENS", 700)),
        enhanced_max_output_tokens=int(config.get("VISION_ENHANCED_MAX_TOKENS", 420)),
        repair_max_output_tokens=int(config.get("VISION_REPAIR_MAX_TOKENS", 420)),
    )
    enhanced_analysis_enabled = bool(config.get("ENHANCED_ANALYSIS_ENABLED", True))
    setattr(vision_service, "enhanced_analysis_enabled", enhanced_analysis_enabled)

    time_parser = TimeParser(
        api_key=str(
            config.get("TIME_PARSE_API_KEY")
            or config.get("LLM_API_KEY")
            or config.get("SU8_API_KEY", "")
        ),
        model_name=str(config.get("TIME_PARSE_MODEL", "gpt-5.1")),
        base_url=str(
            config.get("TIME_PARSE_BASE_URL")
            or config.get("LLM_BASE_URL")
            or config.get("SU8_BASE_URL", "https://www.su8.codes/codex/v1")
        ),
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
    query_format_base_url = str(
        config.get("QUERY_FORMAT_BASE_URL")
        or config.get("LLM_BASE_URL", "https://www.su8.codes/codex/v1")
        or config.get("SU8_BASE_URL", "https://www.su8.codes/codex/v1")
    )
    if config.get("QUERY_FORMAT_ENABLED", True) and _has_usable_api_config(
        config.get("QUERY_FORMAT_API_KEY"),
        query_format_base_url,
    ):
        try:
            from utils.query_formatter import QueryFormatter

            query_formatter = QueryFormatter(
                api_key=str(config.get("QUERY_FORMAT_API_KEY", "")),
                model_name=str(config.get("QUERY_FORMAT_MODEL", "gpt-5.1")),
                base_url=query_format_base_url,
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
        background_mode=str(config.get("INDEX_BACKGROUND_MODE", "process")),
        worker_python_executable=sys.executable,
        worker_entrypoint=os.path.abspath(__file__),
        worker_log_path=os.path.join(data_dir, "index_worker.log"),
        worker_cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    searcher = Searcher(
        embedding=embedding_service,
        time_parser=time_parser,
        vector_store=vector_store,
        keyword_store=keyword_store,
        query_formatter=query_formatter,
        data_dir=runtime_data_dir,
        top_k=int(config.get("TOP_K", 12)),
        vector_weight=float(config.get("VECTOR_WEIGHT", 0.8)),
        keyword_weight=float(config.get("KEYWORD_WEIGHT", 0.2)),
        query_expansion_enabled=bool(config.get("QUERY_EXPANSION_ENABLED", True)),
        query_expansion_max_alternatives=int(config.get("QUERY_EXPANSION_MAX_ALTERNATIVES", 2)),
        query_multi_round_enabled=bool(config.get("QUERY_MULTI_ROUND_ENABLED", False)),
        query_reflection_enabled=bool(config.get("QUERY_REFLECTION_ENABLED", False)),
        query_max_reflection_rounds=int(config.get("QUERY_MAX_REFLECTION_ROUNDS", 2)),
        query_dynamic_threshold_floor=float(config.get("QUERY_DYNAMIC_THRESHOLD_FLOOR", 0.05)),
        query_strict_floor_min=float(config.get("QUERY_STRICT_FLOOR_MIN", 0.22)),
        query_broad_floor_min=float(config.get("QUERY_BROAD_FLOOR_MIN", 0.12)),
        time_parse_strategy=str(config.get("TIME_PARSE_STRATEGY", "local_first")),
        validate_file_exists=bool(config.get("SEARCH_VALIDATE_FILE_EXISTS", False)),
        query_cache_enabled=bool(config.get("QUERY_CACHE_ENABLED", True)),
        query_cache_size=int(config.get("QUERY_CACHE_SIZE", 2000)),
        embedding_cache_enabled=bool(config.get("EMBEDDING_CACHE_ENABLED", True)),
        embedding_cache_size=int(config.get("EMBEDDING_CACHE_SIZE", 5000)),
        default_search_mode=str(config.get("DEFAULT_SEARCH_MODE", "balanced")),
    )

    text_rerank_service: Optional[TextRerankService] = None
    text_rerank_base_url = str(config.get("TEXT_RERANK_BASE_URL", "https://router.tumuer.me/v1"))
    if _has_usable_api_config(config.get("TEXT_RERANK_API_KEY"), text_rerank_base_url):
        try:
            text_rerank_service = TextRerankService(
                api_key=str(config.get("TEXT_RERANK_API_KEY", "")),
                model_name=str(config.get("TEXT_RERANK_MODEL", "Qwen/Qwen3-Reranker-8B")),
                base_url=text_rerank_base_url,
                timeout=int(config.get("TEXT_RERANK_TIMEOUT", 45)),
                max_retries=int(config.get("MAX_RETRIES", 3)),
                backend=str(config.get("TEXT_RERANK_BACKEND", "auto")),
            )
        except Exception as exc:
            print(f"Warning: Failed to initialize text rerank service: {exc}")

    visual_rerank_service = None
    visual_rerank_api_key = (
        config.get("VISUAL_RERANK_API_KEY")
        or config.get("VISION_API_KEY")
        or config.get("LLM_API_KEY", "")
        or config.get("SU8_API_KEY", "")
    )
    visual_rerank_base_url = str(
        config.get("VISUAL_RERANK_BASE_URL")
        or config.get("VISION_BASE_URL")
        or config.get("LLM_BASE_URL", "https://www.su8.codes/codex/v1")
        or config.get("SU8_BASE_URL", "https://www.su8.codes/codex/v1")
    )
    if config.get("VISUAL_RERANK_ENABLED", True) and _has_usable_api_config(
        visual_rerank_api_key,
        visual_rerank_base_url,
    ):
        try:
            from utils.rerank_service import VisualRerankService

            visual_rerank_service = VisualRerankService(
                api_key=str(visual_rerank_api_key),
                model_name=str(config.get("VISUAL_RERANK_MODEL", config.get("VISION_MODEL", "gpt-5.4"))),
                base_url=visual_rerank_base_url,
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
    llm_api_key = config.get("LLM_API_KEY") or config.get("SU8_API_KEY")
    llm_base_url = config.get("LLM_BASE_URL") or config.get("SU8_BASE_URL")
    if not _has_usable_api_config(llm_api_key, llm_base_url):
        raise ValueError("LLM_API_KEY环境变量未设置")
    if not _has_usable_api_config(config.get("EMBEDDING_API_KEY"), config.get("EMBEDDING_BASE_URL")):
        raise ValueError("EMBEDDING_API_KEY环境变量未设置")


def _socket_family_for_host(host: str) -> int:
    return socket.AF_INET6 if ":" in host else socket.AF_INET


def _can_bind(host: str, port: int) -> bool:
    family = _socket_family_for_host(host)
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _find_available_port(host: str, preferred_port: int, max_attempts: int = 20) -> int:
    for offset in range(1, max_attempts + 1):
        candidate = preferred_port + offset
        if _can_bind(host, candidate):
            return candidate

    family = _socket_family_for_host(host)
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _is_port_bind_error(exc: OSError) -> bool:
    win_error = getattr(exc, "winerror", None)
    return bool(
        win_error in {10013, 10048}
        or exc.errno in {errno.EACCES, errno.EADDRINUSE}
    )


def _resolve_server_port(host: str, preferred_port: int) -> tuple[int, bool]:
    if _can_bind(host, preferred_port):
        return preferred_port, False
    fallback_port = _find_available_port(host, preferred_port)
    return fallback_port, True


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
    requested_port = int(config.get("SERVER_PORT", 10001))
    port, used_fallback = _resolve_server_port(host, requested_port)
    if used_fallback:
        print(f"Warning: 端口 {requested_port} 无法绑定，自动切换到端口 {port}。")
    print(f"启动服务器: http://{host}:{port}")
    try:
        app.run(host=host, port=port, debug=False)
    except OSError as exc:
        if not _is_port_bind_error(exc):
            raise
        fallback_port = _find_available_port(host, port)
        if fallback_port == port:
            raise
        print(f"Warning: 端口 {port} 无法绑定 ({exc})，自动切换到端口 {fallback_port}。")
        print(f"启动服务器: http://{host}:{fallback_port}")
        app.run(host=host, port=fallback_port, debug=False)


def run_index_worker(*, force_rebuild: bool = False) -> int:
    config = load_config()
    _validate_required_config(config)
    indexer, _, _, _ = initialize_services(config)
    result = indexer.build_index(force_rebuild=force_rebuild, lock_already_held=True)
    return 0 if result.get("status") in {"success", "ready"} else 1


def run_cli(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--index-worker", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    args, _ = parser.parse_known_args(argv)
    if args.index_worker:
        return run_index_worker(force_rebuild=bool(args.force_rebuild))
    main()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
