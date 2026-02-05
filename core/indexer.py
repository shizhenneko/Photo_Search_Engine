from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from utils.image_parser import (
    extract_exif_metadata,
    generate_fallback_description,
    get_file_time,
    is_valid_image,
)
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
        self._fallback_count = 0
        self._status: Dict[str, Any] = {
            "status": "idle",
            "message": "尚未开始索引构建",
            "total_count": 0,
            "indexed_count": 0,
            "failed_count": 0,
            "fallback_ratio": 0.0,
            "index_path": self.vector_store.index_path,
            "elapsed_time": 0.0,
        }

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

    def generate_description(self, photo_path: str) -> str:
        """
        调用Vision LLM生成描述，失败时使用文件名降级策略。
        """
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                print(f"[INFO] 尝试生成图片描述 (第{attempt+1}/{self.max_retries}次): {photo_path}")
                description = self.vision_llm_service.generate_description(photo_path)
                if not description:
                    raise ValueError("描述结果为空")
                print(f"[INFO] Vision LLM返回描述成功")
                return description
            except Exception as exc:
                last_error = exc
                print(f"[WARN] Vision LLM调用失败 (第{attempt+1}次): {exc}")
                time.sleep(0.5)

        print(f"[FALLBACK] Vision LLM失败，使用降级描述策略")
        fallback = generate_fallback_description(photo_path)
        self._fallback_count += 1
        if last_error is not None:
            _ = last_error
        return fallback

    def process_batch(self, photo_paths: List[str]) -> List[Dict[str, Any]]:
        """
        批量处理照片（描述+嵌入+元数据），单个失败不影响整批。

        改进：
        - embedding 只基于纯 description 生成
        - 新增 time_info 字段包含详细时间信息
        """
        results: List[Dict[str, Any]] = []
        for photo_path in photo_paths:
            print(f"[INFO] 开始处理图片: {photo_path}")
            try:
                description = self.generate_description(photo_path)
                print(f"[INFO] 图片描述生成成功: {description[:50]}...")
                
                exif_data = extract_exif_metadata(photo_path)
                file_time = get_file_time(photo_path)
                
                # 只使用纯 description 进行 embedding
                search_text = self._build_search_text(description, photo_path, exif_data, file_time)
                
                # 提取详细时间信息用于 ES 过滤
                time_info = self._extract_time_info(exif_data, file_time)
                
                print(f"[INFO] 开始生成embedding向量（仅基于description）...")
                embedding = self.embedding_service.generate_embedding(search_text)
                print(f"[INFO] Embedding生成成功，维度: {len(embedding)}")
                
                results.append(
                    {
                        "photo_path": photo_path,
                        "description": description,
                        "search_text": search_text,
                        "embedding": embedding,
                        "exif_data": exif_data,
                        "file_time": file_time,
                        "time_info": time_info,  # 新增：详细时间信息
                        "status": "success",
                        "error": None,
                    }
                )
                print(f"[SUCCESS] 图片处理成功: {photo_path}")
            except Exception as exc:
                import traceback
                error_trace = traceback.format_exc()
                print(f"[ERROR] 处理图片失败: {photo_path}")
                print(f"[ERROR] 错误信息: {exc}")
                print(f"[ERROR] 详细堆栈:\n{error_trace}")
                results.append(
                    {
                        "photo_path": photo_path,
                        "description": None,
                        "embedding": None,
                        "exif_data": None,
                        "file_time": None,
                        "time_info": None,
                        "status": "failed",
                        "error": f"处理照片失败: {exc}",
                    }
                )
        return results

    def _build_search_text(
        self,
        description: str,
        photo_path: str,
        exif_data: Optional[Dict[str, Any]],
        file_time: Optional[str],
    ) -> str:
        """
        构建用于向量化的搜索文本。

        改进：只使用纯 description 进行 embedding，不再混入时间/文件名等信息。
        EXIF 元数据通过 Elasticsearch 进行精确过滤，不污染向量空间。
        """
        # 只返回纯描述用于 embedding
        if description and len(description) >= 20:
            return description.strip()
        return ""

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
        if file_time:
            try:
                return datetime.fromisoformat(file_time)
            except Exception:
                return None
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

    def build_index(self) -> Dict[str, Any]:
        """
        主流程：扫描、批处理、写入索引并进行验收门槛检查。
        """
        if not self._create_lock():
            return self._response_with_message("processing", "索引构建正在进行中")

        start_time = time.time()
        self._fallback_count = 0
        success_count = 0
        failed_count = 0

        try:
            self._remove_ready_marker()
            photo_paths = self.scan_photos()
            total_count = len(photo_paths)
            self._update_status(
                status="processing",
                message="索引构建中",
                total_count=total_count,
                indexed_count=0,
                failed_count=0,
                fallback_ratio=0.0,
                elapsed_time=0.0,
            )

            if total_count == 0:
                return self._response_with_message("failed", "未找到可索引的图片文件")

            for start in range(0, total_count, self.batch_size):
                batch = photo_paths[start : start + self.batch_size]
                batch_results = self.process_batch(batch)

                for item in batch_results:
                    if item["status"] == "success":
                        try:
                            # 更新 metadata 结构，包含 time_info
                            metadata = {
                                "photo_path": item["photo_path"],
                                "description": item["description"],
                                "search_text": item.get("search_text"),
                                "exif_data": item["exif_data"],
                                "file_time": item["file_time"],
                                "time_info": item.get("time_info"),  # 新增：详细时间信息
                            }
                            self.vector_store.add_item(item["embedding"], metadata)
                            
                            # Sync to KeywordStore with complete EXIF fields
                            if self.keyword_store is not None:
                                import hashlib
                                doc_id = hashlib.md5(item["photo_path"].encode()).hexdigest()
                                
                                time_info = item.get("time_info") or {}
                                exif_data = item.get("exif_data") or {}
                                
                                # 构建包含独立 EXIF 字段的文档
                                document = {
                                    "photo_path": item["photo_path"],
                                    "description": item["description"],
                                    "file_name": os.path.basename(item["photo_path"]),
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
                                self.keyword_store.add_document(doc_id, document)

                            success_count += 1
                        except Exception as exc:
                            failed_count += 1
                            item["status"] = "failed"
                            item["error"] = f"写入索引失败: {exc}"
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

            try:
                self.vector_store.save()
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
            return self._status.copy()
        finally:
            self._release_lock()

    def get_status(self) -> Dict[str, Any]:
        """
        获取索引构建状态，包含锁文件与状态文件的综合判断。

        改进：添加EXIF覆盖率统计。
        """
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
        if os.path.exists(self._ready_path):
            status["status"] = "ready"
            status["message"] = "索引已就绪"
        return status

    def _compute_fallback_ratio(self, success_count: int) -> float:
        if success_count <= 0:
            return 0.0
        return round(self._fallback_count / float(success_count), 4)

    def _create_lock(self) -> bool:
        if os.path.exists(self._lock_path):
            return False
        try:
            with open(self._lock_path, "w", encoding="utf-8") as file:
                file.write(datetime.now().isoformat())
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
        }
        self._write_status_file(self._status)

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
