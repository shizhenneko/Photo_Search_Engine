from __future__ import annotations

import os
import tempfile
import time
from io import BytesIO
from typing import Any, Dict, TYPE_CHECKING
from urllib.parse import quote, unquote

from flask import Flask, jsonify, render_template, request, send_file
from jinja2 import FileSystemLoader

from utils.path_utils import ensure_display_path, normalize_local_path, open_in_file_manager
from utils.image_parser import is_valid_image

if TYPE_CHECKING:
    from core.indexer import Indexer
    from core.searcher import Searcher
    from utils.embedding_service import TextRerankService
    from utils.rerank_service import VisualRerankService


def _enrich_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for item in results:
        result = dict(item)
        photo_path = result.get("photo_path", "")
        normalized_path = normalize_local_path(photo_path)
        result["photo_path"] = ensure_display_path(photo_path)
        result["photo_url"] = f"/photo?path={quote(normalized_path)}" if normalized_path else ""
        result["file_name"] = os.path.basename(normalized_path) if normalized_path else ""
        result["match_summary"] = dict(result.get("match_summary") or {})
        enriched.append(result)
    return enriched


def _calculate_rerank_search_pool_k(
    *,
    top_k: int,
    rerank_top_k: int,
    enable_text_rerank: bool,
    enable_visual_rerank: bool,
) -> int:
    normalized_top_k = max(1, min(int(top_k), 50))
    normalized_rerank_top_k = max(1, min(int(rerank_top_k), normalized_top_k))
    if not (enable_text_rerank or enable_visual_rerank):
        return normalized_top_k

    desired = max(normalized_top_k, normalized_rerank_top_k)
    multiplier = 3 if enable_visual_rerank else 2
    adaptive_extra = max(4, desired // 2)
    return min(50, max(desired * multiplier, desired + adaptive_extra))


def _apply_rerank_pipeline(
    *,
    results: list[dict[str, Any]],
    top_k: int,
    rerank_top_k: int,
    enable_text_rerank: bool,
    enable_visual_rerank: bool,
    text_query: str | None,
    reference_image_path: str | None,
    text_rerank_service: "TextRerankService | None",
    visual_rerank_service: "VisualRerankService | None",
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    normalized_top_k = max(1, min(int(top_k), 50))
    normalized_rerank_top_k = max(1, min(int(rerank_top_k), normalized_top_k))
    rerank_state = {
        "text_reranked": False,
        "visual_reranked": False,
    }
    reranked_results = list(results)
    full_candidate_count = len(reranked_results)

    if enable_text_rerank and text_query and text_rerank_service and text_rerank_service.is_enabled():
        reranked_results = text_rerank_service.rerank(text_query, reranked_results, full_candidate_count)
        rerank_state["text_reranked"] = True

    if enable_visual_rerank and visual_rerank_service and visual_rerank_service.is_enabled():
        try:
            if reference_image_path:
                reranked_results = visual_rerank_service.rerank_by_reference_image(
                    reference_image_path,
                    reranked_results,
                    full_candidate_count,
                )
            elif text_query:
                reranked_results = visual_rerank_service.rerank(text_query, reranked_results, full_candidate_count)
            rerank_state["visual_reranked"] = True
        except Exception as exc:
            # 视觉重排是可选增强能力，失败时不应让搜索整体失败。
            print(f"Warning: visual rerank skipped: {exc}")

    final_limit = normalized_rerank_top_k if any(rerank_state.values()) else normalized_top_k
    reranked_results = reranked_results[:final_limit]
    for rank, item in enumerate(reranked_results, start=1):
        item["rank"] = rank

    return reranked_results, rerank_state


def register_routes(
    app: Flask,
    indexer: "Indexer",
    searcher: "Searcher",
    config: Dict[str, Any],
    text_rerank_service: "TextRerankService" = None,
    visual_rerank_service: "VisualRerankService" = None,
) -> None:
    templates_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "templates"))
    if os.path.isdir(templates_dir):
        if isinstance(app.jinja_loader, FileSystemLoader):
            if templates_dir not in app.jinja_loader.searchpath:
                app.jinja_loader.searchpath.append(templates_dir)
        else:
            app.jinja_loader = FileSystemLoader([templates_dir])

    @app.route("/")
    def index() -> Any:
        return render_template("index.html")

    @app.route("/init_index", methods=["POST"])
    def init_index() -> Any:
        try:
            status = indexer.get_status()
            if status.get("status") == "processing":
                return jsonify(status), 400
            data = request.get_json(silent=True) or {}
            mode = str(data.get("mode") or "incremental").strip().lower()
            force_rebuild = mode == "full"
            return jsonify(indexer.start_build_in_background(force_rebuild=force_rebuild))
        except Exception as exc:
            return jsonify(
                {
                    "status": "failed",
                    "message": f"索引构建异常: {exc}",
                    "total_count": 0,
                    "indexed_count": 0,
                    "failed_count": 0,
                    "fallback_ratio": 0.0,
                    "elapsed_time": 0.0,
                }
            ), 500

    @app.route("/search_photos", methods=["POST"])
    def search_photos() -> Any:
        start_time = time.time()
        try:
            current_status = indexer.get_status()
            if current_status.get("status") == "processing":
                return jsonify(
                    {
                        "status": "error",
                        "message": "索引仍在构建中，请稍后再搜索",
                        "results": [],
                        "total_results": 0,
                        "elapsed_time": round(time.time() - start_time, 4),
                        "text_reranked": False,
                        "visual_reranked": False,
                    }
                ), 409

            data = request.get_json()
            if data is None:
                return jsonify({"status": "error", "message": "请求体必须为JSON格式"}), 400

            query = (data.get("query") or "").strip()
            if not query:
                return jsonify({"status": "error", "message": "查询内容不能为空"}), 400

            search_mode = str(data.get("search_mode") or config.get("DEFAULT_SEARCH_MODE", "balanced")).strip().lower()
            top_k = min(int(data.get("top_k", config.get("TOP_K", 12))), 50)
            rerank_top_k = min(max(1, int(data.get("rerank_top_k", top_k))), top_k)
            enable_text_rerank = bool(data.get("enable_text_rerank", False))
            enable_visual_rerank = bool(data.get("enable_visual_rerank", False))
            search_pool_k = _calculate_rerank_search_pool_k(
                top_k=top_k,
                rerank_top_k=rerank_top_k,
                enable_text_rerank=enable_text_rerank,
                enable_visual_rerank=enable_visual_rerank,
            )
            results = searcher.search(query, search_pool_k, search_mode=search_mode)
            results, rerank_state = _apply_rerank_pipeline(
                results=results,
                top_k=top_k,
                rerank_top_k=rerank_top_k,
                enable_text_rerank=enable_text_rerank,
                enable_visual_rerank=enable_visual_rerank,
                text_query=query,
                reference_image_path=None,
                text_rerank_service=text_rerank_service,
                visual_rerank_service=visual_rerank_service,
            )

            enriched_results = _enrich_results(results)
            return jsonify(
                {
                    "status": "success",
                    "results": enriched_results,
                    "total_results": len(enriched_results),
                    "elapsed_time": round(time.time() - start_time, 4),
                    "search_debug": searcher.get_last_search_debug(),
                    **rerank_state,
                }
            )
        except ValueError as exc:
            return jsonify(
                {
                    "status": "error",
                    "message": str(exc),
                    "results": [],
                    "total_results": 0,
                    "elapsed_time": round(time.time() - start_time, 4),
                    "text_reranked": False,
                    "visual_reranked": False,
                }
            ), 400
        except Exception as exc:
            return jsonify(
                {
                    "status": "error",
                    "message": f"搜索异常: {exc}",
                    "results": [],
                    "total_results": 0,
                    "elapsed_time": round(time.time() - start_time, 4),
                    "text_reranked": False,
                    "visual_reranked": False,
                }
            ), 500

    @app.route("/search_by_image", methods=["POST"])
    def search_by_image() -> Any:
        start_time = time.time()
        try:
            current_status = indexer.get_status()
            if current_status.get("status") == "processing":
                return jsonify(
                    {
                        "status": "error",
                        "message": "索引仍在构建中，请稍后再搜索",
                        "results": [],
                        "total_results": 0,
                        "elapsed_time": round(time.time() - start_time, 4),
                        "text_reranked": False,
                        "visual_reranked": False,
                    }
                ), 409

            data = request.get_json()
            if data is None:
                return jsonify({"status": "error", "message": "请求体必须为JSON格式"}), 400

            image_path = normalize_local_path((data.get("image_path") or "").strip())
            if not image_path:
                return jsonify({"status": "error", "message": "图片路径不能为空"}), 400

            top_k = min(int(data.get("top_k", config.get("TOP_K", 12))), 50)
            rerank_top_k = min(max(1, int(data.get("rerank_top_k", top_k))), top_k)
            enable_text_rerank = bool(data.get("enable_text_rerank", False))
            enable_visual_rerank = bool(data.get("enable_visual_rerank", False))
            query_hint = (data.get("query_hint") or "").strip() or None
            search_pool_k = _calculate_rerank_search_pool_k(
                top_k=top_k,
                rerank_top_k=rerank_top_k,
                enable_text_rerank=enable_text_rerank,
                enable_visual_rerank=enable_visual_rerank,
            )
            results = searcher.search_by_image_path(image_path, search_pool_k)
            results, rerank_state = _apply_rerank_pipeline(
                results=results,
                top_k=top_k,
                rerank_top_k=rerank_top_k,
                enable_text_rerank=enable_text_rerank,
                enable_visual_rerank=enable_visual_rerank,
                text_query=query_hint,
                reference_image_path=image_path,
                text_rerank_service=text_rerank_service,
                visual_rerank_service=visual_rerank_service,
            )

            enriched_results = _enrich_results(results)
            return jsonify(
                {
                    "status": "success",
                    "query_image_path": ensure_display_path(image_path),
                    "results": enriched_results,
                    "total_results": len(enriched_results),
                    "elapsed_time": round(time.time() - start_time, 4),
                    "search_debug": searcher.get_last_search_debug(),
                    **rerank_state,
                }
            )
        except ValueError as exc:
            return jsonify(
                {
                    "status": "error",
                    "message": str(exc),
                    "results": [],
                    "total_results": 0,
                    "elapsed_time": round(time.time() - start_time, 4),
                    "text_reranked": False,
                    "visual_reranked": False,
                }
            ), 400
        except Exception as exc:
            return jsonify(
                {
                    "status": "error",
                    "message": f"以图搜图异常: {exc}",
                    "results": [],
                    "total_results": 0,
                    "elapsed_time": round(time.time() - start_time, 4),
                    "text_reranked": False,
                    "visual_reranked": False,
                }
            ), 500

    @app.route("/search_by_uploaded_image", methods=["POST"])
    def search_by_uploaded_image() -> Any:
        start_time = time.time()
        temp_path = ""
        try:
            current_status = indexer.get_status()
            if current_status.get("status") == "processing":
                return jsonify(
                    {
                        "status": "error",
                        "message": "索引仍在构建中，请稍后再搜索",
                        "results": [],
                        "total_results": 0,
                        "elapsed_time": round(time.time() - start_time, 4),
                        "text_reranked": False,
                        "visual_reranked": False,
                    }
                ), 409

            uploaded = request.files.get("image")
            if uploaded is None or not uploaded.filename:
                return jsonify({"status": "error", "message": "请上传图片文件"}), 400

            suffix = os.path.splitext(uploaded.filename)[1] or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                uploaded.save(temp_file)
                temp_path = temp_file.name

            if not is_valid_image(temp_path):
                raise ValueError("上传的文件不是有效图片")

            top_k = min(int(request.form.get("top_k", config.get("TOP_K", 12))), 50)
            rerank_top_k = min(max(1, int(request.form.get("rerank_top_k", top_k))), top_k)
            enable_text_rerank = str(request.form.get("enable_text_rerank", "")).lower() in {"true", "1", "on"}
            enable_visual_rerank = str(request.form.get("enable_visual_rerank", "")).lower() in {"true", "1", "on"}
            query_hint = (request.form.get("query_hint") or "").strip() or None
            analysis = indexer.generate_analysis(temp_path)
            search_pool_k = _calculate_rerank_search_pool_k(
                top_k=top_k,
                rerank_top_k=rerank_top_k,
                enable_text_rerank=enable_text_rerank,
                enable_visual_rerank=enable_visual_rerank,
            )
            results = searcher.search_by_uploaded_image(temp_path, analysis=analysis, top_k=search_pool_k)
            results, rerank_state = _apply_rerank_pipeline(
                results=results,
                top_k=top_k,
                rerank_top_k=rerank_top_k,
                enable_text_rerank=enable_text_rerank,
                enable_visual_rerank=enable_visual_rerank,
                text_query=query_hint,
                reference_image_path=temp_path,
                text_rerank_service=text_rerank_service,
                visual_rerank_service=visual_rerank_service,
            )

            enriched_results = _enrich_results(results)
            return jsonify(
                {
                    "status": "success",
                    "query_image_path": ensure_display_path(temp_path),
                    "query_image_name": uploaded.filename,
                    "results": enriched_results,
                    "total_results": len(enriched_results),
                    "elapsed_time": round(time.time() - start_time, 4),
                    "search_debug": searcher.get_last_search_debug(),
                    **rerank_state,
                }
            )
        except ValueError as exc:
            return jsonify(
                {
                    "status": "error",
                    "message": str(exc),
                    "results": [],
                    "total_results": 0,
                    "elapsed_time": round(time.time() - start_time, 4),
                    "text_reranked": False,
                    "visual_reranked": False,
                }
            ), 400
        except Exception as exc:
            return jsonify(
                {
                    "status": "error",
                    "message": f"上传图片检索异常: {exc}",
                    "results": [],
                    "total_results": 0,
                    "elapsed_time": round(time.time() - start_time, 4),
                    "text_reranked": False,
                    "visual_reranked": False,
                }
            ), 500
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    @app.route("/open_photo_location", methods=["POST"])
    def open_photo_location() -> Any:
        try:
            data = request.get_json()
            if data is None:
                return jsonify({"status": "error", "message": "请求体必须为JSON格式"}), 400

            image_path = (data.get("image_path") or "").strip()
            if not image_path:
                return jsonify({"status": "error", "message": "图片路径不能为空"}), 400

            open_in_file_manager(image_path)
            return jsonify({"status": "success", "message": "已尝试打开文件所在位置"})
        except FileNotFoundError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 404
        except Exception as exc:
            return jsonify({"status": "error", "message": f"打开文件位置失败: {exc}"}), 500

    @app.route("/index_status", methods=["GET"])
    def index_status() -> Any:
        try:
            return jsonify(indexer.get_status())
        except Exception as exc:
            return jsonify(
                {
                    "status": "failed",
                    "message": f"获取状态失败: {exc}",
                    "total_count": 0,
                    "indexed_count": 0,
                    "failed_count": 0,
                    "elapsed_time": 0.0,
                }
            ), 500

    @app.route("/photo")
    def get_photo() -> Any:
        try:
            path = request.args.get("path", "")
            if not path:
                return "缺少path参数", 400

            decoded_path = unquote(path)
            normalized_path = normalize_local_path(decoded_path)

            if ".." in os.path.normpath(decoded_path).split(os.sep):
                return "拒绝访问：非法路径", 403
            if not os.path.isabs(normalized_path):
                return "路径必须为绝对路径", 400
            if not os.path.isfile(normalized_path):
                return f"文件不存在: {decoded_path}", 404

            _, ext = os.path.splitext(normalized_path)
            supported_formats = {".jpg", ".jpeg", ".png", ".webp"}
            if ext.lower() not in supported_formats:
                return "不支持的文件格式", 400

            with open(normalized_path, "rb") as file:
                content = file.read()

            mime_type = "image/webp"
            if ext.lower() in {".jpg", ".jpeg"}:
                mime_type = "image/jpeg"
            elif ext.lower() == ".png":
                mime_type = "image/png"

            return send_file(
                BytesIO(content),
                mimetype=mime_type,
                as_attachment=False,
                download_name=os.path.basename(normalized_path),
            )
        except Exception as exc:
            return f"获取图片失败: {exc}", 500

    @app.errorhandler(404)
    def not_found(error: Any) -> Any:
        return jsonify({"status": "error", "message": "接口不存在"}), 404

    @app.errorhandler(500)
    def internal_error(error: Any) -> Any:
        return jsonify({"status": "error", "message": "服务器内部错误"}), 500
