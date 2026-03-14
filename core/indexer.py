from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import uuid4

from utils.image_parser import (
    extract_exif_metadata,
    generate_fallback_description,
    get_file_time,
    is_valid_image,
)
from utils.structured_analysis import normalize_analysis_payload
from utils.vector_store import VectorStore

if TYPE_CHECKING:
    from utils.embedding_service import EmbeddingService
    from utils.vision_llm_service import VisionLLMService
    from utils.keyword_store import KeywordStore


class Indexer:
    """
    照片索引构建器，负责扫描、描述生成、向量化和索引持久化。
    """

    def __init__(
        self,
        photo_dir: str,
        vision: "VisionLLMService",
        embedding: "EmbeddingService",
        vector_store: VectorStore,
        keyword_store: Optional["KeywordStore"] = None,
        data_dir: str = "./data",
        batch_size: int = 10,
        max_retries: int = 3,
        timeout: int = 30,
    ) -> None:
        """
        初始化索引构建器，并准备数据目录与状态文件路径。
        """
        if not photo_dir:
            raise ValueError("照片目录不能为空")
        self.photo_dir = os.path.abspath(photo_dir)
        self.vision_llm_service = vision
        self.embedding_service = embedding
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.batch_size = max(1, batch_size)
        self.max_retries = max(1, max_retries)
        self.timeout = max(1, timeout)
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        if hasattr(self.vision_llm_service, "timeout"):
            try:
                self.vision_llm_service.timeout = self.timeout
            except Exception:
                pass

        self._status_path = os.path.join(self.data_dir, "index_status.status")
        self._lock_path = os.path.join(self.data_dir, "indexing.lock")
        self._ready_path = os.path.join(self.data_dir, "index_ready.marker")
        self._timing_log_path = os.path.join(self.data_dir, "index_timing.jsonl")
        self._fallback_count = 0
        self._current_run_id: Optional[str] = None
        self._background_thread: Optional[threading.Thread] = None
        self._background_lock = threading.Lock()
        self._lock_stale_seconds = max(900, self.timeout * self.batch_size * 3)
        # 缓存用于复用的现有结构化分析
        self._cached_analyses: Dict[str, Dict[str, Any]] = {}
        self._status: Dict[str, Any] = {
            "status": "idle",
            "message": "尚未开始索引构建",
            "total_count": 0,
            "indexed_count": 0,
            "failed_count": 0,
            "fallback_ratio": 0.0,
            "index_path": self.vector_store.index_path,
            "elapsed_time": 0.0,
            "timing_log_path": self._timing_log_path,
        }

    def start_build_in_background(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        后台启动索引构建，立即返回当前 processing 状态，避免阻塞 HTTP 请求。
        """
        with self._background_lock:
            self._clear_stale_lock_if_needed()
            if os.path.exists(self._lock_path):
                return self.get_status()

            total_count = len(self.scan_photos())
            indexed_count = 0 if force_rebuild else self.vector_store.get_total_items()
            if not self._create_lock():
                return self.get_status()

            self._remove_ready_marker()
            self._update_status(
                status="processing",
                message="索引构建中",
                total_count=total_count,
                indexed_count=indexed_count,
                failed_count=0,
                fallback_ratio=0.0,
                elapsed_time=0.0,
            )

            def _runner() -> None:
                try:
                    self.build_index(force_rebuild=force_rebuild, lock_already_held=True)
                except Exception as exc:
                    self._update_status(
                        status="failed",
                        message=f"索引构建异常: {exc}",
                        total_count=self._status.get("total_count", 0),
                        indexed_count=self._status.get("indexed_count", 0),
                        failed_count=self._status.get("failed_count", 0),
                        fallback_ratio=self._status.get("fallback_ratio", 0.0),
                        elapsed_time=self._status.get("elapsed_time", 0.0),
                    )
                    self._release_lock()
                finally:
                    with self._background_lock:
                        self._background_thread = None

            self._background_thread = threading.Thread(
                target=_runner,
                name="photo-index-build",
                daemon=True,
            )
            self._background_thread.start()
            return self._status.copy()

    def scan_photos(self) -> List[str]:
        """
        扫描照片目录，返回有效图片路径（递归扫描、按修改时间排序）。
        """
        if not os.path.isdir(self.photo_dir):
            return []

        photo_paths: List[str] = []
        for root, _, files in os.walk(self.photo_dir):
            for name in files:
                path = os.path.abspath(os.path.join(root, name))
                if is_valid_image(path):
                    photo_paths.append(path)

        def _safe_mtime(file_path: str) -> float:
            try:
                return os.path.getmtime(file_path)
            except Exception:
                return 0.0

        photo_paths.sort(key=_safe_mtime)
        return photo_paths

    def generate_analysis(self, photo_path: str) -> Dict[str, Any]:
        """
        调用 Vision LLM 生成结构化分析，失败时退回已有缓存或本地兜底结构。
        """
        if photo_path in self._cached_analyses:
            cached = self._cached_analyses[photo_path]
            if cached and cached.get("retrieval_text"):
                print(f"[INFO] 使用缓存结构化分析: {photo_path}")
                return cached

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                print(f"[INFO] 尝试生成图片结构化分析 (第{attempt+1}/{self.max_retries}次): {photo_path}")
                service_metrics_before = None
                if hasattr(self.vision_llm_service, "get_last_analysis_metrics"):
                    try:
                        service_metrics_before = self.vision_llm_service.get_last_analysis_metrics()
                    except Exception:
                        service_metrics_before = None
                analysis = self.vision_llm_service.analyze_image(photo_path)
                if not analysis or not analysis.get("retrieval_text"):
                    raise ValueError("结构化分析结果为空")
                service_metrics = None
                if hasattr(self.vision_llm_service, "get_last_analysis_metrics"):
                    try:
                        service_metrics = self.vision_llm_service.get_last_analysis_metrics()
                    except Exception:
                        service_metrics = service_metrics_before
                if isinstance(service_metrics, dict):
                    analysis.setdefault("_timing_metrics", {})
                    analysis["_timing_metrics"]["vision_service"] = service_metrics
                print("[INFO] Vision LLM返回结构化分析成功")
                return analysis
            except Exception as exc:
                last_error = exc
                print(f"[WARN] Vision LLM调用失败 (第{attempt+1}次): {exc}")
                time.sleep(0.5)

        print("[FALLBACK] Vision LLM失败，使用本地结构化兜底策略")
        fallback = normalize_analysis_payload(
            {
                "description": generate_fallback_description(photo_path),
                "outer_scene_summary": generate_fallback_description(photo_path),
                "inner_content_summary": "",
                "media_types": ["photo"],
                "tags": [],
                "ocr_text": "",
                "person_roles": [],
                "identity_candidates": [],
                "analysis_flags": {},
            },
            tag_min_confidence=0.65,
            identity_text_threshold=0.7,
            identity_visual_threshold=0.92,
        )
        self._fallback_count += 1
        fallback["_timing_metrics"] = {
            "vision_service": {
                "image_encode_seconds": 0.0,
                "attempts": [],
                "base_analysis_seconds": 0.0,
                "base_parse_seconds": 0.0,
                "base_normalize_seconds": 0.0,
                "enhanced_prompt_seconds": 0.0,
                "enhanced_analysis_seconds": 0.0,
                "enhanced_parse_seconds": 0.0,
                "enhanced_normalize_seconds": 0.0,
                "enhanced_triggered": False,
                "enhanced_succeeded": False,
                "used_fallback": True,
            }
        }
        if last_error is not None:
            _ = last_error
        return fallback

    def _now_iso(self) -> str:
        return datetime.now().isoformat()

    def _new_run_id(self) -> str:
        return f"index-run-{self._now_iso()}-{uuid4().hex[:8]}"

    def _append_timing_log(self, payload: Dict[str, Any]) -> None:
        record = {
            "timestamp": self._now_iso(),
            "run_id": self._current_run_id,
            **payload,
        }
        try:
            with open(self._timing_log_path, "a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[WARN] 写入索引耗时日志失败: {exc}")

    def _log_stage_timing(
        self,
        stage: str,
        elapsed: float,
        *,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "event": "build_stage_timing",
            "stage": stage,
            "elapsed_seconds": round(elapsed, 4),
        }
        if details:
            payload["details"] = details
        self._append_timing_log(payload)

    def _log_photo_timing(
        self,
        photo_path: str,
        steps: Dict[str, float],
        total_elapsed: float,
        *,
        event: str = "photo_timing",
        status: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "event": event,
            "photo_path": photo_path,
            "status": status,
            "total_elapsed_seconds": round(total_elapsed, 4),
            "steps": {name: round(value, 4) for name, value in steps.items()},
        }
        if details:
            payload["details"] = details
        self._append_timing_log(payload)

    def process_batch(self, photo_paths: List[str]) -> List[Dict[str, Any]]:
        """
        批量处理照片（描述+嵌入+元数据），单个失败不影响整批。

        改进：
        - embedding 改为基于 retrieval_text 生成
        - 元数据持久化结构化分析字段
        - 新增 time_info 字段包含详细时间信息
        """
        results: List[Dict[str, Any]] = []
        for photo_path in photo_paths:
            print(f"[INFO] 开始处理图片: {photo_path}")
            photo_start = time.perf_counter()
            step_timings: Dict[str, float] = {}
            try:
                analysis_start = time.perf_counter()
                analysis = self.generate_analysis(photo_path)
                step_timings["generate_analysis"] = time.perf_counter() - analysis_start
                description = str(analysis.get("description") or "")
                retrieval_text = str(analysis.get("retrieval_text") or "").strip()
                print(f"[INFO] 图片结构化分析成功: {description[:50]}...")

                exif_start = time.perf_counter()
                exif_data = extract_exif_metadata(photo_path)
                file_time = get_file_time(photo_path)
                step_timings["extract_exif"] = time.perf_counter() - exif_start

                # 提取详细时间信息用于 ES 过滤
                time_info_start = time.perf_counter()
                time_info = self._extract_time_info(exif_data, file_time)
                step_timings["extract_time_info"] = time.perf_counter() - time_info_start

                print("[INFO] 开始生成embedding向量（基于retrieval_text）...")
                embedding_start = time.perf_counter()
                embedding = self.embedding_service.generate_embedding(retrieval_text)
                step_timings["generate_embedding"] = time.perf_counter() - embedding_start
                print(f"[INFO] Embedding生成成功，维度: {len(embedding)}")
                total_elapsed = time.perf_counter() - photo_start
                self._log_photo_timing(
                    photo_path,
                    step_timings,
                    total_elapsed,
                    status="success",
                    details={
                        "description_length": len(description),
                        "retrieval_text_length": len(retrieval_text),
                        "embedding_dimension": len(embedding),
                        "used_fallback_analysis": bool(analysis.get("analysis_flags", {}).get("fallback")),
                        "analysis_timing_metrics": analysis.get("_timing_metrics", {}),
                    },
                )
                results.append(
                    {
                        "photo_path": photo_path,
                        "description": description,
                        "retrieval_text": retrieval_text,
                        "analysis": analysis,
                        "embedding": embedding,
                        "exif_data": exif_data,
                        "file_time": file_time,
                        "time_info": time_info,  # 新增：详细时间信息
                        "status": "success",
                        "error": None,
                        "step_timings": {name: round(value, 4) for name, value in step_timings.items()},
                        "processing_elapsed": round(total_elapsed, 4),
                    }
                )
                print(f"[SUCCESS] 图片处理成功: {photo_path}")
            except Exception as exc:
                import traceback
                error_trace = traceback.format_exc()
                total_elapsed = time.perf_counter() - photo_start
                self._log_photo_timing(
                    photo_path,
                    step_timings,
                    total_elapsed,
                    status="failed",
                    details={"error": str(exc)},
                )
                print(f"[ERROR] 处理图片失败: {photo_path}")
                print(f"[ERROR] 错误信息: {exc}")
                print(f"[ERROR] 详细堆栈:\n{error_trace}")
                results.append(
                    {
                        "photo_path": photo_path,
                        "description": None,
                        "retrieval_text": None,
                        "analysis": None,
                        "embedding": None,
                        "exif_data": None,
                        "file_time": None,
                        "time_info": None,
                        "status": "failed",
                        "error": f"处理照片失败: {exc}",
                        "step_timings": {name: round(value, 4) for name, value in step_timings.items()},
                        "processing_elapsed": round(total_elapsed, 4),
                    }
                )
        return results

    def process_batch_with_progress(
        self,
        photo_paths: List[str],
        *,
        total_count: int,
        success_count: int,
        failed_count: int,
        start_time: float,
    ) -> List[Dict[str, Any]]:
        """
        顺序处理批次中的图片，并在每张完成后刷新状态，避免长时间无心跳。
        """
        results: List[Dict[str, Any]] = []
        for photo_path in photo_paths:
            self._update_status(
                status="processing",
                message=f"正在处理: {os.path.basename(photo_path)}",
                total_count=total_count,
                indexed_count=success_count,
                failed_count=failed_count,
                fallback_ratio=self._compute_fallback_ratio(success_count),
                elapsed_time=time.time() - start_time,
            )

            item = self.process_batch([photo_path])[0]
            results.append(item)

            if item["status"] == "success":
                success_count += 1
            else:
                failed_count += 1

            self._update_status(
                status="processing",
                message=f"已处理 {success_count + failed_count}/{total_count} 张",
                total_count=total_count,
                indexed_count=success_count,
                failed_count=failed_count,
                fallback_ratio=self._compute_fallback_ratio(success_count),
                elapsed_time=time.time() - start_time,
            )

        return results

    def _extract_time_info(
        self,
        exif_data: Optional[Dict[str, Any]],
        file_time: Optional[str],
    ) -> Dict[str, Any]:
        """
        提取详细的时间信息，用于 Elasticsearch 精确过滤。

        时段细分为7档：
        - 凌晨 (0:00-5:00)
        - 早晨 (5:00-8:00)
        - 上午 (8:00-12:00)
        - 中午 (12:00-14:00)
        - 下午 (14:00-17:00)
        - 傍晚 (17:00-19:00)
        - 夜晚 (19:00-24:00)

        Returns:
            Dict[str, Any]: 包含 year, month, day, hour, season, time_period, weekday
        """
        time_info: Dict[str, Any] = {
            "year": None,
            "month": None,
            "day": None,
            "hour": None,
            "season": None,
            "time_period": None,
            "weekday": None,
            "datetime_str": None,
        }

        # 只信任 EXIF 拍摄时间生成结构化时间标签。
        # 文件修改时间可以保留作展示或调试参考，但不能反推出季节/时段，
        # 否则没有 EXIF 的图片会被错误打标。
        photo_date = self._get_photo_datetime(exif_data, file_time)
        if not photo_date:
            return time_info

        # 基础时间字段
        time_info["year"] = photo_date.year
        time_info["month"] = photo_date.month
        time_info["day"] = photo_date.day
        time_info["hour"] = photo_date.hour
        time_info["datetime_str"] = photo_date.isoformat()

        # 季节
        time_info["season"] = self._month_to_season(photo_date.month)

        # 时段（7档细分）
        hour = photo_date.hour
        if 0 <= hour < 5:
            time_info["time_period"] = "凌晨"
        elif 5 <= hour < 8:
            time_info["time_period"] = "早晨"
        elif 8 <= hour < 12:
            time_info["time_period"] = "上午"
        elif 12 <= hour < 14:
            time_info["time_period"] = "中午"
        elif 14 <= hour < 17:
            time_info["time_period"] = "下午"
        elif 17 <= hour < 19:
            time_info["time_period"] = "傍晚"
        else:  # 19 <= hour < 24
            time_info["time_period"] = "夜晚"

        # 星期
        weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        time_info["weekday"] = weekday_names[photo_date.weekday()]

        return time_info

    def _get_photo_datetime(
        self, exif_data: Optional[Dict[str, Any]], file_time: Optional[str]
    ) -> Optional[datetime]:
        if exif_data:
            raw = exif_data.get("datetime")
            if raw:
                try:
                    return datetime.fromisoformat(raw)
                except Exception:
                    pass
        _ = file_time
        return None

    @staticmethod
    def _month_to_season(month: int) -> Optional[str]:
        if month in {3, 4, 5}:
            return "春天"
        if month in {6, 7, 8}:
            return "夏天"
        if month in {9, 10, 11}:
            return "秋天"
        if month in {12, 1, 2}:
            return "冬天"
        return None

    def build_index(self, force_rebuild: bool = False, lock_already_held: bool = False) -> Dict[str, Any]:
        """
        主流程：扫描、批处理、写入索引并进行验收门槛检查。
        """
        if not lock_already_held and not self._create_lock():
            return self._response_with_message("processing", "索引构建正在进行中")

        start_time = time.time()
        build_perf_start = time.perf_counter()
        self._current_run_id = self._new_run_id()
        self._append_timing_log(
            {
                "event": "build_started",
                "force_rebuild": force_rebuild,
                "batch_size": self.batch_size,
                "photo_dir": self.photo_dir,
            }
        )
        loaded_existing_index = False
        if force_rebuild:
            rebuild_clear_start = time.perf_counter()
            print("[INFO] 全量重建模式，正在清空现有索引...")
            self.vector_store.clear()
            if self.keyword_store:
                try:
                    self.keyword_store.clear()
                except Exception as exc:
                    print(f"[WARN] KeywordStore清理失败: {exc}")
            self._log_stage_timing("clear_existing_index", time.perf_counter() - rebuild_clear_start)
        elif self.vector_store.get_total_items() == 0:
            load_existing_start = time.perf_counter()
            try:
                loaded_existing_index = self.vector_store.load()
            except Exception as exc:
                print(f"[WARN] 现有向量索引加载失败，将执行全量重建: {exc}")
                self.vector_store.clear()
            self._log_stage_timing(
                "load_existing_index",
                time.perf_counter() - load_existing_start,
                details={"loaded_existing_index": loaded_existing_index},
            )

        # 1. 加载现有高质量数据用于缓存
        print("[INFO] 正在加载现有元数据以进行智能复用...")
        cache_load_start = time.perf_counter()
        self._cached_analyses.clear()
        if self.vector_store.metadata:
            for item in self.vector_store.metadata:
                path = item.get("photo_path")
                retrieval_text = item.get("retrieval_text")
                if path and retrieval_text and isinstance(retrieval_text, str):
                    self._cached_analyses[path] = {
                        "description": item.get("description"),
                        "outer_scene_summary": item.get("outer_scene_summary"),
                        "inner_content_summary": item.get("inner_content_summary"),
                        "media_types": item.get("media_types") or [],
                        "tags": item.get("top_tags") or item.get("tags") or [],
                        "ocr_text": item.get("ocr_text") or "",
                        "person_roles": item.get("person_roles") or [],
                        "identity_candidates": item.get("identity_candidates") or [],
                        "identity_names": item.get("identity_names") or [],
                        "identity_evidence": item.get("identity_evidence") or [],
                        "analysis_flags": item.get("analysis_flags") or {},
                        "retrieval_text": retrieval_text,
                    }
        print(f"[INFO] 已缓存 {len(self._cached_analyses)} 条有效结构化分析")
        self._log_stage_timing(
            "prepare_cached_analyses",
            time.perf_counter() - cache_load_start,
            details={"cached_analysis_count": len(self._cached_analyses)},
        )

        self._fallback_count = 0
        existing_count = self.vector_store.get_total_items()
        success_count = existing_count
        failed_count = 0

        try:
            self._remove_ready_marker()
            scan_start = time.perf_counter()
            photo_paths = self.scan_photos()
            self._log_stage_timing(
                "scan_photos",
                time.perf_counter() - scan_start,
                details={"photo_count": len(photo_paths)},
            )
            total_count = len(photo_paths)
            existing_path_lookup_start = time.perf_counter()
            existing_paths = {
                item.get("photo_path")
                for item in self.vector_store.metadata
                if item.get("photo_path")
            }
            new_photo_paths = [
                path
                for path in photo_paths
                if path not in existing_paths and not self.vector_store.has_photo_path(path)
            ]
            self._log_stage_timing(
                "filter_new_photos",
                time.perf_counter() - existing_path_lookup_start,
                details={
                    "existing_count": existing_count,
                    "new_photo_count": len(new_photo_paths),
                },
            )
            self._update_status(
                status="processing",
                message="索引构建中",
                total_count=total_count,
                indexed_count=success_count,
                failed_count=0,
                fallback_ratio=0.0,
                elapsed_time=0.0,
            )

            if total_count == 0:
                return self._response_with_message("failed", "未找到可索引的图片文件")

            if not new_photo_paths:
                elapsed_time = time.time() - start_time
                self._create_ready_marker()
                self._update_status(
                    status="success",
                    message="索引已是最新，无新增图片需要处理",
                    total_count=total_count,
                    indexed_count=success_count,
                    failed_count=0,
                    fallback_ratio=0.0,
                    elapsed_time=elapsed_time,
                )
                self._append_timing_log(
                    {
                        "event": "build_finished",
                        "status": "success",
                        "elapsed_seconds": round(time.perf_counter() - build_perf_start, 4),
                        "details": {
                            "total_count": total_count,
                            "indexed_count": success_count,
                            "failed_count": failed_count,
                            "message": "索引已是最新，无新增图片需要处理",
                        },
                    }
                )
                return self._status.copy()

            if force_rebuild:
                print(f"[INFO] 全量重建模式，本轮重建 {len(photo_paths)} 张图片")
                new_photo_paths = photo_paths
            elif loaded_existing_index:
                print(f"[INFO] 已加载现有索引，当前已有 {existing_count} 张图片")
            if not force_rebuild:
                print(f"[INFO] 增量索引模式，本轮新增 {len(new_photo_paths)} 张图片")

            for start in range(0, len(new_photo_paths), self.batch_size):
                batch = new_photo_paths[start : start + self.batch_size]
                batch_start = time.perf_counter()
                batch_results = self.process_batch_with_progress(
                    batch,
                    total_count=total_count,
                    success_count=success_count,
                    failed_count=failed_count,
                    start_time=start_time,
                )
                batch_success_count = 0

                for item in batch_results:
                    if item["status"] == "success":
                        try:
                            # 更新 metadata 结构，包含 time_info
                            metadata = {
                                "photo_path": item["photo_path"],
                                "description": item["description"],
                                "outer_scene_summary": item["analysis"].get("outer_scene_summary"),
                                "inner_content_summary": item["analysis"].get("inner_content_summary"),
                                "media_types": item["analysis"].get("media_types") or [],
                                "top_tags": item["analysis"].get("tags") or [],
                                "ocr_text": item["analysis"].get("ocr_text") or "",
                                "person_roles": item["analysis"].get("person_roles") or [],
                                "identity_candidates": item["analysis"].get("identity_candidates") or [],
                                "identity_names": item["analysis"].get("identity_names") or [],
                                "identity_evidence": item["analysis"].get("identity_evidence") or [],
                                "analysis_flags": item["analysis"].get("analysis_flags") or {},
                                "retrieval_text": item.get("retrieval_text"),
                                "exif_data": item["exif_data"],
                                "file_time": item["file_time"],
                                "time_info": item.get("time_info"),  # 新增：详细时间信息
                            }
                            vector_write_start = time.perf_counter()
                            self.vector_store.add_item(item["embedding"], metadata)
                            vector_write_elapsed = time.perf_counter() - vector_write_start
                            
                            # Sync to KeywordStore with complete EXIF fields
                            keyword_write_elapsed = 0.0
                            if self.keyword_store is not None:
                                import hashlib
                                doc_id = hashlib.md5(item["photo_path"].encode()).hexdigest()
                                
                                time_info = item.get("time_info") or {}
                                exif_data = item.get("exif_data") or {}
                                
                                # 构建包含独立 EXIF 字段的文档
                                document = {
                                    "photo_path": item["photo_path"],
                                    "description": item["description"],
                                    "outer_scene_summary": item["analysis"].get("outer_scene_summary"),
                                    "inner_content_summary": item["analysis"].get("inner_content_summary"),
                                    "retrieval_text": item.get("retrieval_text"),
                                    "ocr_text": item["analysis"].get("ocr_text") or "",
                                    "file_name": os.path.basename(item["photo_path"]),
                                    "media_types": item["analysis"].get("media_types") or [],
                                    "tags": item["analysis"].get("tags") or [],
                                    "identity_names": item["analysis"].get("identity_names") or [],
                                    "identity_evidence": item["analysis"].get("identity_evidence") or [],
                                    # EXIF 独立字段（用于精确过滤）
                                    "year": time_info.get("year"),
                                    "month": time_info.get("month"),
                                    "day": time_info.get("day"),
                                    "hour": time_info.get("hour"),
                                    "season": time_info.get("season"),
                                    "time_period": time_info.get("time_period"),
                                    "weekday": time_info.get("weekday"),
                                    "camera": exif_data.get("camera"),
                                    "datetime": time_info.get("datetime_str"),
                                }
                                keyword_write_start = time.perf_counter()
                                self.keyword_store.add_document(doc_id, document)
                                keyword_write_elapsed = time.perf_counter() - keyword_write_start

                            persist_steps = {
                                "vector_store_add_item": vector_write_elapsed,
                            }
                            if self.keyword_store is not None:
                                persist_steps["keyword_store_add_document"] = keyword_write_elapsed
                            self._log_photo_timing(
                                item["photo_path"],
                                persist_steps,
                                vector_write_elapsed + keyword_write_elapsed,
                                event="photo_persist_timing",
                                status="success",
                                details={
                                    "batch_start_index": start,
                                },
                            )

                            success_count += 1
                            batch_success_count += 1
                        except Exception as exc:
                            failed_count += 1
                            item["status"] = "failed"
                            item["error"] = f"写入索引失败: {exc}"
                            self._log_photo_timing(
                                item["photo_path"],
                                {},
                                0.0,
                                event="photo_persist_timing",
                                status="failed",
                                details={"error": str(exc), "batch_start_index": start},
                            )
                    else:
                        failed_count += 1

                elapsed_time = time.time() - start_time
                fallback_ratio = self._compute_fallback_ratio(success_count)
                self._update_status(
                    status="processing",
                    message="索引构建中",
                    total_count=total_count,
                    indexed_count=success_count,
                    failed_count=failed_count,
                    fallback_ratio=fallback_ratio,
                    elapsed_time=elapsed_time,
                )
                if batch_success_count > 0:
                    try:
                        save_start = time.perf_counter()
                        self.vector_store.save()
                        self._log_stage_timing(
                            "save_vector_store_batch",
                            time.perf_counter() - save_start,
                            details={
                                "batch_start_index": start,
                                "batch_size": len(batch),
                                "batch_success_count": batch_success_count,
                            },
                        )
                    except Exception as exc:
                        return self._response_with_message("failed", f"索引保存失败: {exc}")
                self._log_stage_timing(
                    "process_batch",
                    time.perf_counter() - batch_start,
                    details={
                        "batch_start_index": start,
                        "batch_size": len(batch),
                        "batch_success_count": batch_success_count,
                        "batch_failed_count": len(batch) - batch_success_count,
                    },
                )

            try:
                final_save_start = time.perf_counter()
                self.vector_store.save()
                self._log_stage_timing(
                    "save_vector_store_final",
                    time.perf_counter() - final_save_start,
                    details={"total_items": self.vector_store.get_total_items()},
                )
            except Exception as exc:
                return self._response_with_message("failed", f"索引保存失败: {exc}")

            fallback_ratio = self._compute_fallback_ratio(success_count)
            elapsed_time = time.time() - start_time

            min_success_count = min(100, total_count)
            if success_count < min_success_count or fallback_ratio >= 0.1:
                message = "索引构建未达标（成功数量不足或降级占比过高）"
                self._update_status(
                    status="failed",
                    message=message,
                    total_count=total_count,
                    indexed_count=success_count,
                    failed_count=failed_count,
                    fallback_ratio=fallback_ratio,
                    elapsed_time=elapsed_time,
                )
                self._append_timing_log(
                    {
                        "event": "build_finished",
                        "status": "failed",
                        "elapsed_seconds": round(time.perf_counter() - build_perf_start, 4),
                        "details": {
                            "total_count": total_count,
                            "indexed_count": success_count,
                            "failed_count": failed_count,
                            "fallback_ratio": fallback_ratio,
                            "message": message,
                        },
                    }
                )
                return self._status.copy()

            self._create_ready_marker()
            self._update_status(
                status="success",
                message="索引构建成功",
                total_count=total_count,
                indexed_count=success_count,
                failed_count=failed_count,
                fallback_ratio=fallback_ratio,
                elapsed_time=elapsed_time,
            )
            self._append_timing_log(
                {
                    "event": "build_finished",
                    "status": "success",
                    "elapsed_seconds": round(time.perf_counter() - build_perf_start, 4),
                    "details": {
                        "total_count": total_count,
                        "indexed_count": success_count,
                        "failed_count": failed_count,
                        "fallback_ratio": fallback_ratio,
                    },
                }
            )
            return self._status.copy()
        finally:
            self._release_lock()
            self._current_run_id = None

    def get_status(self) -> Dict[str, Any]:
        """
        获取索引构建状态，包含锁文件与状态文件的综合判断。

        改进：添加EXIF覆盖率统计。
        """
        cleared_stale_lock = self._clear_stale_lock_if_needed()
        status = self._read_status_file()
        
        # 添加EXIF覆盖率统计
        if self.vector_store.metadata:
            exif_count = sum(1 for item in self.vector_store.metadata
                            if item.get("exif_data", {}).get("datetime"))
            if len(self.vector_store.metadata) > 0:
                status["exif_coverage"] = round(
                    exif_count / len(self.vector_store.metadata), 4
                )
            else:
                status["exif_coverage"] = 0.0
        else:
            status["exif_coverage"] = 0.0

        if os.path.exists(self._lock_path):
            status["status"] = "processing"
            status["message"] = "索引构建中"
            return status
        if cleared_stale_lock and status.get("status") == "processing":
            status["status"] = "failed"
            status["message"] = "检测到上次索引任务已中断，请重新开始"
            self._status = {**self._status, **status}
            self._write_status_file(self._status)
        if os.path.exists(self._ready_path):
            status["status"] = "ready"
            status["message"] = "索引已就绪"
        return status

    def _compute_fallback_ratio(self, success_count: int) -> float:
        if success_count <= 0:
            return 0.0
        return round(self._fallback_count / float(success_count), 4)

    def _create_lock(self) -> bool:
        self._clear_stale_lock_if_needed()
        if os.path.exists(self._lock_path):
            return False
        try:
            now = datetime.now().isoformat()
            payload = {
                "pid": os.getpid(),
                "created_at": now,
                "updated_at": now,
            }
            with open(self._lock_path, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    def _release_lock(self) -> None:
        try:
            if os.path.exists(self._lock_path):
                os.remove(self._lock_path)
        except Exception:
            pass

    def _create_ready_marker(self) -> None:
        try:
            with open(self._ready_path, "w", encoding="utf-8") as file:
                file.write("ready")
        except Exception:
            pass

    def _remove_ready_marker(self) -> None:
        try:
            if os.path.exists(self._ready_path):
                os.remove(self._ready_path)
        except Exception:
            pass

    def _update_status(
        self,
        status: str,
        message: str,
        total_count: int,
        indexed_count: int,
        failed_count: int,
        fallback_ratio: float,
        elapsed_time: float,
        ) -> None:
        self._status = {
            "status": status,
            "message": message,
            "total_count": total_count,
            "indexed_count": indexed_count,
            "failed_count": failed_count,
            "fallback_ratio": fallback_ratio,
            "index_path": self.vector_store.index_path,
            "elapsed_time": round(elapsed_time, 4),
            "timing_log_path": self._timing_log_path,
        }
        self._write_status_file(self._status)
        if status == "processing" and os.path.exists(self._lock_path):
            self._refresh_lock()

    def _write_status_file(self, payload: Dict[str, Any]) -> None:
        try:
            with open(self._status_path, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _read_status_file(self) -> Dict[str, Any]:
        if not os.path.exists(self._status_path):
            return self._status.copy()
        try:
            with open(self._status_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            return {**self._status, **data}
        except Exception:
            return self._status.copy()

    def _response_with_message(self, status: str, message: str) -> Dict[str, Any]:
        self._update_status(
            status=status,
            message=message,
            total_count=self._status.get("total_count", 0),
            indexed_count=self._status.get("indexed_count", 0),
            failed_count=self._status.get("failed_count", 0),
            fallback_ratio=self._status.get("fallback_ratio", 0.0),
            elapsed_time=self._status.get("elapsed_time", 0.0),
        )
        return self._status.copy()

    def _refresh_lock(self) -> None:
        payload = self._read_lock_payload()
        now = datetime.now().isoformat()
        next_payload = {
            "pid": os.getpid(),
            "created_at": (payload or {}).get("created_at", now),
            "updated_at": now,
        }
        try:
            with open(self._lock_path, "w", encoding="utf-8") as file:
                json.dump(next_payload, file, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _read_lock_payload(self) -> Dict[str, Any]:
        if not os.path.exists(self._lock_path):
            return {}
        try:
            with open(self._lock_path, "r", encoding="utf-8") as file:
                raw = file.read().strip()
        except Exception:
            return {}

        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {"legacy": True, "raw": raw}
        return payload if isinstance(payload, dict) else {"legacy": True, "raw": raw}

    def _clear_stale_lock_if_needed(self) -> bool:
        payload = self._read_lock_payload()
        if not payload:
            return False
        if payload.get("legacy"):
            self._release_lock()
            return True
        else:
            pid = payload.get("pid")
            if not isinstance(pid, int):
                return False
            if self._pid_exists(pid):
                return False
        self._release_lock()
        return True

    def _legacy_lock_is_stale(self) -> bool:
        lock_age = self._get_file_age(self._lock_path)
        if lock_age is None or lock_age < self._lock_stale_seconds:
            return False
        status_age = self._get_file_age(self._status_path)
        return status_age is None or status_age >= self._lock_stale_seconds

    @staticmethod
    def _get_file_age(path: str) -> Optional[float]:
        if not os.path.exists(path):
            return None
        try:
            return time.time() - os.path.getmtime(path)
        except OSError:
            return None

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True
