from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from utils.vector_store import VectorStore

if TYPE_CHECKING:
    from utils.embedding_service import EmbeddingService
    from utils.time_parser import TimeParser


class Searcher:
    """
    照片检索器，负责加载索引、查询向量化、相似度检索与时间过滤。
    """

    def __init__(
        self,
        embedding: "EmbeddingService",
        time_parser: "TimeParser",
        vector_store: VectorStore,
        data_dir: str = "./data",
        top_k: int = 10,
    ) -> None:
        """
        初始化检索器并记录索引路径与加载状态。
        """
        self.embedding_service = embedding
        self.time_parser = time_parser
        self.vector_store = vector_store
        self.data_dir = data_dir
        self.top_k = max(1, top_k)
        self.index_loaded = False
        self.index_path = vector_store.index_path
        self.metadata_path = vector_store.metadata_path
        self.metric = getattr(vector_store, "metric", "l2")

    def load_index(self) -> bool:
        """
        从磁盘加载FAISS索引与元数据，验证索引完整性。
        """
        loaded = self.vector_store.load()
        if not loaded:
            self.index_loaded = False
            return False

        expected_dimension = getattr(self.embedding_service, "dimension", None)
        if expected_dimension is not None and self.vector_store.dimension != expected_dimension:
            raise ValueError("向量维度不一致")

        self.index_loaded = True
        return True

    def validate_query(self, query: str) -> bool:
        """
        校验查询文本长度与有效字符。
        """
        if not isinstance(query, str):
            return False
        text = query.strip()
        if len(text) < 5 or len(text) > 500:
            return False
        if re.fullmatch(r"[\W_]+", text):
            return False
        return True

    def _extract_time_constraints(self, query: str) -> Dict[str, Any]:
        """
        调用TimeParser解析时间约束，失败时返回无约束。
        """
        try:
            constraints = self.time_parser.extract_time_constraints(query)
            if not isinstance(constraints, dict):
                return {"start_date": None, "end_date": None, "precision": "none"}
            return {
                "start_date": constraints.get("start_date"),
                "end_date": constraints.get("end_date"),
                "precision": constraints.get("precision", "none"),
            }
        except Exception:
            return {"start_date": None, "end_date": None, "precision": "none"}

    def _filter_by_time(self, results: List[Dict[str, Any]], constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        根据时间约束过滤结果，优先EXIF时间，其次文件时间。
        """
        start_date = constraints.get("start_date")
        end_date = constraints.get("end_date")
        if not start_date and not end_date:
            return results

        start = self._parse_date(start_date) if start_date else None
        end = self._parse_date(end_date) if end_date else None
        if start is None and end is None:
            return results

        filtered: List[Dict[str, Any]] = []
        for item in results:
            metadata = item.get("metadata") or {}
            exif_data = metadata.get("exif_data") or {}
            timestamp = exif_data.get("datetime") or metadata.get("file_time")
            if not timestamp:
                continue
            photo_date = self._parse_date(timestamp)
            if photo_date is None:
                continue
            if start and photo_date < start:
                continue
            if end and photo_date > end:
                continue
            filtered.append(item)
        return filtered

    def _distance_to_score(self, distance: float) -> float:
        """
        将L2距离转换为0-1的相似度分数。
        """
        if self.metric == "cosine":
            similarity = max(-1.0, min(1.0, distance))
            return round(max(0.0, min(1.0, (similarity + 1.0) / 2.0)), 6)
        if distance < 0:
            distance = 0
        return round(1.0 / (1.0 + distance), 6)

    def _calculate_candidate_k(self, normalized_top_k: int, has_time_filter: bool) -> int:
        """
        根据是否有时间过滤动态计算候选数量。

        Args:
            normalized_top_k: 用户请求的结果数量
            has_time_filter: 是否存在时间约束

        Returns:
            int: 候选数量
        """
        if has_time_filter:
            # 有时间过滤：扩大候选集，保证过滤后仍有足够结果
            multiplier = 10
            max_candidate = 100
        else:
            # 无时间过滤：保持原有策略
            multiplier = 5
            max_candidate = 50

        candidate_k = normalized_top_k * multiplier
        return min(candidate_k, max_candidate)

    def _has_time_terms(self, query: str) -> bool:
        patterns = [
            r"\d{4}年",
            r"\d{1,2}月",
            r"\d{1,2}日",
            r"\d{4}-\d{1,2}-\d{1,2}",
            r"去年|今年|前年|明年|上个月|下个月|上周|下周|本周|这个月|上个?星期|下个?星期",
            r"春天|夏天|秋天|冬天|季节|月份|年份",
        ]
        return any(re.search(pattern, query) for pattern in patterns)

    def _strip_time_terms(self, query: str) -> str:
        cleaned = query
        patterns = [
            r"\d{4}年",
            r"\d{1,2}月",
            r"\d{1,2}日",
            r"\d{4}-\d{1,2}-\d{1,2}",
            r"去年|今年|前年|明年|上个月|下个月|上周|下周|本周|这个月|上个?星期|下个?星期",
            r"春天|夏天|秋天|冬天|季节|月份|年份",
        ]
        for pattern in patterns:
            cleaned = re.sub(pattern, " ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        解析查询、执行向量检索、时间过滤并返回排序结果。
        """
        if not self.validate_query(query):
            raise ValueError("查询内容不合法，请输入5-500字符的描述")

        if not self.index_loaded and not self.load_index():
            raise ValueError("索引未加载，请先初始化索引")

        normalized_top_k = max(1, min(int(top_k), 50))
        constraints = {"start_date": None, "end_date": None, "precision": "none"}
        cleaned_query = query
        has_time_filter = self._has_time_terms(query)

        if has_time_filter:
            constraints = self._extract_time_constraints(query)
            cleaned_query = self._strip_time_terms(query) or query

        query_embedding = self.embedding_service.generate_embedding(cleaned_query)
        candidate_k = self._calculate_candidate_k(normalized_top_k, has_time_filter)
        raw_results = self.vector_store.search(query_embedding, candidate_k)
        filtered_results = self._filter_by_time(raw_results, constraints)

        enriched: List[Dict[str, Any]] = []
        for item in filtered_results:
            metadata = item.get("metadata") or {}
            score = self._distance_to_score(float(item.get("distance", 0.0)))
            enriched.append(
                {
                    "photo_path": metadata.get("photo_path"),
                    "description": metadata.get("description"),
                    "score": score,
                }
            )

        enriched.sort(key=lambda x: x["score"], reverse=True)
        for rank, item in enumerate(enriched, start=1):
            item["rank"] = rank
        return enriched

    def get_index_stats(self) -> Dict[str, Any]:
        """
        获取索引统计信息，索引未加载时返回默认值。
        """
        return {
            "total_items": self.vector_store.get_total_items() if self.index_loaded else 0,
            "vector_dimension": self.vector_store.dimension if self.index_loaded else None,
            "index_loaded": self.index_loaded,
            "index_path": self.index_path,
        }

    def _parse_date(self, value: str) -> Optional[datetime]:
        try:
            if len(value) == 10:
                return datetime.strptime(value, "%Y-%m-%d")
            return datetime.fromisoformat(value)
        except Exception:
            return None
