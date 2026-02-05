"""
API路由模块，提供HTTP接口连接前端和核心业务逻辑。

路由清单：
- GET / - 渲染前端页面
- POST /init_index - 触发索引构建
- POST /search_photos - 执行照片搜索
- GET /index_status - 获取索引状态
- GET /photo - 返回图片文件（供前端和Vision LLM使用）
"""

from __future__ import annotations

import os
from io import BytesIO
from typing import Any, Dict, TYPE_CHECKING
from urllib.parse import quote, unquote

from flask import Flask, jsonify, request, send_file, render_template
from jinja2 import FileSystemLoader

if TYPE_CHECKING:
    from core.indexer import Indexer
    from core.searcher import Searcher
    from utils.rerank_service import RerankService


def register_routes(
    app: Flask,
    indexer: "Indexer",
    searcher: "Searcher",
    config: Dict[str, Any],
    rerank_service: "RerankService" = None,
) -> None:
    """
    注册所有API路由。

    Args:
        app: Flask应用实例
        indexer: 索引构建器实例
        searcher: 检索器实例
        config: 配置字典
        rerank_service: Rerank服务实例（可选）
    """

    templates_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "templates")
    )
    if os.path.isdir(templates_dir):
        if isinstance(app.jinja_loader, FileSystemLoader):
            if templates_dir not in app.jinja_loader.searchpath:
                app.jinja_loader.searchpath.append(templates_dir)
        else:
            app.jinja_loader = FileSystemLoader([templates_dir])

    @app.route("/")
    def index() -> Any:
        """
        渲染前端页面。

        Returns:
            HTML页面
        """
        return render_template("index.html")

    @app.route("/init_index", methods=["POST"])
    def init_index() -> Any:
        """
        触发索引构建。

        如果索引正在构建中，返回processing状态。
        否则启动索引构建流程。

        请求体：无

        响应格式：
        {
            "status": "success" | "processing" | "failed",
            "message": "索引构建成功/进行中/失败",
            "total_count": int,  # 总图片数量
            "indexed_count": int,  # 成功索引数量
            "failed_count": int,  # 失败数量
            "fallback_ratio": float,  # 降级描述占比
            "elapsed_time": float  # 耗时（秒）
        }
        """
        try:
            # 检查是否正在构建中
            status = indexer.get_status()
            if status.get("status") == "processing":
                return jsonify(status), 400

            # 启动索引构建
            result = indexer.build_index()
            return jsonify(result)
        except Exception as e:
            return (
                jsonify(
                    {
                        "status": "failed",
                        "message": f"索引构建异常: {e}",
                        "total_count": 0,
                        "indexed_count": 0,
                        "failed_count": 0,
                        "fallback_ratio": 0.0,
                        "elapsed_time": 0.0,
                    }
                ),
                500,
            )

    @app.route("/search_photos", methods=["POST"])
    def search_photos() -> Any:
        """
        执行照片搜索（支持可选的Rerank精排）。

        请求体：
        {
            "query": str,           # 查询文本
            "top_k": int,           # 返回结果数量（可选，默认10，最大50）
            "enable_rerank": bool,  # 是否启用Rerank精排（可选，默认false）
            "rerank_top_k": int     # Rerank后保留数量（可选，默认5）
        }

        响应格式：
        {
            "status": "success" | "error",
            "results": [...],
            "total_results": int,
            "elapsed_time": float,
            "reranked": bool        # 是否经过Rerank
        }
        """
        import time

        start_time = time.time()

        try:
            # 解析请求参数
            data = request.get_json()
            if data is None:
                return (
                    jsonify({"status": "error", "message": "请求体必须为JSON格式"}),
                    400,
                )

            query = data.get("query", "").strip()
            top_k = min(data.get("top_k", config.get("TOP_K", 10)), 50)
            enable_rerank = data.get("enable_rerank", False)
            rerank_top_k = data.get("rerank_top_k", 5)
            # rerank_top_k 不能超过 top_k
            rerank_top_k = min(max(1, rerank_top_k), top_k)

            # 参数校验
            if not query:
                return (
                    jsonify({"status": "error", "message": "查询内容不能为空"}),
                    400,
                )

            # 1. 获取格式化后的 search_text（用于 rerank）
            search_text = query  # 默认使用原始 query
            if enable_rerank and searcher.query_formatter and searcher.query_formatter.is_enabled():
                try:
                    formatted = searcher.query_formatter.format_query(query)
                    search_text = formatted.get("search_text", query)
                    print(f"[RERANK] Formatted search_text: {search_text}")
                except Exception as e:
                    print(f"[RERANK] QueryFormatter failed, using original query: {e}")

            # 2. 执行搜索
            results = searcher.search(query, top_k)

            # 3. 如果启用 rerank 且有结果且 rerank_service 可用
            reranked = False
            if enable_rerank and rerank_service and rerank_service.is_enabled() and results:
                try:
                    print(f"[RERANK] Starting rerank with search_text: {search_text}, candidates: {len(results)}")
                    results = rerank_service.rerank(search_text, results, rerank_top_k)
                    reranked = True
                    print(f"[RERANK] Rerank completed, results: {len(results)}")
                except Exception as e:
                    print(f"[RERANK] Rerank failed, using original results: {e}")

            # 4. 构建响应（添加 photo_url）
            photo_url_base = "/photo?path="
            enriched_results = []
            for item in results:
                photo_path = item.get("photo_path", "")
                encoded_path = quote(photo_path)
                item["photo_url"] = f"{photo_url_base}{encoded_path}"
                enriched_results.append(item)

            elapsed_time = time.time() - start_time

            return (
                jsonify(
                    {
                        "status": "success",
                        "results": enriched_results,
                        "total_results": len(enriched_results),
                        "elapsed_time": round(elapsed_time, 4),
                        "reranked": reranked,
                    }
                ),
                200,
            )
        except ValueError as e:
            elapsed_time = time.time() - start_time
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": str(e),
                        "results": [],
                        "total_results": 0,
                        "elapsed_time": round(elapsed_time, 4),
                        "reranked": False,
                    }
                ),
                400,
            )
        except Exception as e:
            elapsed_time = time.time() - start_time
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"搜索异常: {e}",
                        "results": [],
                        "total_results": 0,
                        "elapsed_time": round(elapsed_time, 4),
                        "reranked": False,
                    }
                ),
                500,
            )

    @app.route("/index_status", methods=["GET"])
    def index_status() -> Any:
        """
        获取索引构建和加载状态。

        响应格式：
        {
            "status": "idle" | "processing" | "ready" | "failed",
            "message": str,
            "total_count": int,
            "indexed_count": int,
            "failed_count": int,
            "elapsed_time": float
        }
        """
        try:
            status = indexer.get_status()
            return jsonify(status)
        except Exception as e:
            return (
                jsonify(
                    {
                        "status": "failed",
                        "message": f"获取状态失败: {e}",
                        "total_count": 0,
                        "indexed_count": 0,
                        "failed_count": 0,
                        "elapsed_time": 0.0,
                    }
                ),
                500,
            )

    @app.route("/photo")
    def get_photo() -> Any:
        """
        返回图片文件，供前端使用。

        参数：
        - path: 图片绝对路径

        注意：
        1. 本地演示用接口，仅供个人使用
        2. 路径校验（必须是绝对路径且存在）
        3. 路径遍历防护（防止访问系统敏感文件）
        """
        try:
            # 获取路径参数
            path = request.args.get("path", "")
            if not path:
                return "缺少path参数", 400

            # 解码与路径规范化
            decoded_path = unquote(path)
            normalized_path = os.path.abspath(decoded_path)

            # 安全检查：防止路径遍历攻击
            if ".." in os.path.normpath(decoded_path).split(os.sep):
                return "拒绝访问：非法路径", 403

            # 路径必须为绝对路径
            if not os.path.isabs(normalized_path):
                return "路径必须为绝对路径", 400

            # 检查文件是否存在
            if not os.path.isfile(normalized_path):
                return f"文件不存在: {decoded_path}", 404

            # 检查是否为支持的图片格式
            _, ext = os.path.splitext(normalized_path)
            supported_formats = {".jpg", ".jpeg", ".png", ".webp"}
            if ext.lower() not in supported_formats:
                return "不支持的文件格式", 400

            # 读取文件后返回，避免Windows下文件句柄占用
            with open(normalized_path, "rb") as f:
                content = f.read()

            ext_lower = ext.lower()
            if ext_lower in {".jpg", ".jpeg"}:
                mime_type = "image/jpeg"
            elif ext_lower == ".png":
                mime_type = "image/png"
            else:
                mime_type = "image/webp"

            return send_file(
                BytesIO(content),
                mimetype=mime_type,
                as_attachment=False,
                download_name=os.path.basename(normalized_path),
            )
        except Exception as e:
            return f"获取图片失败: {e}", 500

    @app.errorhandler(404)
    def not_found(error: Any) -> Any:
        """404错误处理"""
        return jsonify({"status": "error", "message": "接口不存在"}), 404

    @app.errorhandler(500)
    def internal_error(error: Any) -> Any:
        """500错误处理"""
        return jsonify({"status": "error", "message": "服务器内部错误"}), 500
