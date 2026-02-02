from __future__ import annotations

import json
import os
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
        for _ in range(self.max_retries):
            try:
                description = self.vision_llm_service.generate_description(photo_path)
                if not description:
                    raise ValueError("描述结果为空")
                return description
            except Exception as exc:
                last_error = exc
                time.sleep(0.5)

        fallback = generate_fallback_description(photo_path)
        self._fallback_count += 1
        if last_error is not None:
            _ = last_error
        return fallback

    def process_batch(self, photo_paths: List[str]) -> List[Dict[str, Any]]:
        """
        批量处理照片（描述+嵌入+元数据），单个失败不影响整批。
        """
        results: List[Dict[str, Any]] = []
        for photo_path in photo_paths:
            try:
                description = self.generate_description(photo_path)
                embedding = self.embedding_service.generate_embedding(description)
                exif_data = extract_exif_metadata(photo_path)
                file_time = get_file_time(photo_path)
                results.append(
                    {
                        "photo_path": photo_path,
                        "description": description,
                        "embedding": embedding,
                        "exif_data": exif_data,
                        "file_time": file_time,
                        "status": "success",
                        "error": None,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "photo_path": photo_path,
                        "description": None,
                        "embedding": None,
                        "exif_data": None,
                        "file_time": None,
                        "status": "failed",
                        "error": f"处理照片失败: {exc}",
                    }
                )
        return results

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
                            metadata = {
                                "photo_path": item["photo_path"],
                                "description": item["description"],
                                "exif_data": item["exif_data"],
                                "file_time": item["file_time"],
                            }
                            self.vector_store.add_item(item["embedding"], metadata)
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

            if success_count < 100 or fallback_ratio >= 0.1:
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
        """
        status = self._read_status_file()
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
