from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
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
        self.metric = getattr(vector_store, "metric", "cosine")

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
        # 修复边界问题：end_date包含到当天结束，所以使用<=比较
        end = self._parse_date(end_date, is_end_date=True) if end_date else None
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
        将距离转换为0-1的相似度分数。

        改进：Cosine使用Sigmoid映射增强高分区，L2使用指数衰减。
        """
        if self.metric == "cosine":
            # Cosine相似度范围 [-1, 1]，映射到 [0, 1]
            similarity = max(-1.0, min(1.0, distance))
            score = (similarity + 1.0) / 2.0
            # 高分区拉伸，低分区压缩
            if score > 0.7:
                score = 0.7 + (score - 0.7) * 1.3
            elif score < 0.3:
                score = score * 0.8
            return round(max(0.0, min(1.0, score)), 6)
        # L2距离：使用指数衰减替代线性反比，更平滑
        if distance < 0:
            distance = 0
        # alpha=0.5 在 distance=1 时给出约0.61，比 1/2=0.5 更合理
        return round(round(float(np.exp(-0.5 * distance)), 6), 6)

    def _calculate_dynamic_threshold(
        self,
        scores: List[float],
        top_k: int,
    ) -> float:
        """
        基于分数分布计算动态自适应阈值（简化版）。

        改进：使用简单有效的策略，移除复杂且不稳定的多种方法组合。
        """
        if not scores:
            return 0.1

        n = len(scores)

        # 如果候选数量很少，使用第top_k位置的分数作为基准
        if n <= top_k * 2:
            return max(scores[-1] * 0.9, 0.05)

        # 使用第一四分位数作为基础阈值
        q25 = np.percentile(scores, 25)
        q75 = np.percentile(scores, 75)
        median = np.median(scores)

        # 计算变异系数，判断分数分布的稳定性
        if median > 0:
            cv = (q75 - q25) / median
        else:
            cv = 1.0

        # 根据分布特征选择阈值
        if cv < 0.2:
            # 分数集中：使用稍严格阈值
            threshold = max(median * 0.85, q25 * 0.9)
        elif cv < 0.5:
            # 分数中等分散：使用分位数阈值
            threshold = q25
        else:
            # 分数高度分散：使用宽松阈值，保留更多结果
            threshold = max(q25 * 0.7, median * 0.7)

        # 确保top_k结果有机会返回
        if n >= top_k:
            min_threshold = scores[top_k - 1] * 0.8
            threshold = max(threshold, min_threshold)

        # 下限保护
        return round(max(threshold, 0.05), 6)

    def _get_gradient_threshold(
        self,
        scores: List[float],
        top_k: int,
    ) -> float:
        """
        基于分数梯度检测自然断点。

        寻找分数骤降的位置,即"自然边界"。
        如果最大降阶超过30%,则使用该位置分数作为阈值。

        Args:
            scores: 降序排列的分数列表
            top_k: 用户请求的结果数量

        Returns:
            float: 基于梯度的阈值
        """
        if len(scores) < 2:
            return 0.1

        import numpy as np

        # 计算相邻分数的梯度
        gradients = []
        for i in range(len(scores) - 1):
            drop_ratio = (scores[i] - scores[i + 1]) / (scores[i] + 1e-6)
            gradients.append(drop_ratio)

        # 找到最大降阶的位置（下降比例最大的地方）
        max_drop_index = np.argmax(gradients)
        max_drop = gradients[max_drop_index]

        # 如果最大降阶超过30%,使用该位置分数作为阈值
        if max_drop > 0.3:
            return scores[max_drop_index + 1]

        # 否则,使用第一四分位数作为回退
        return max(np.percentile(scores, 25), 0.05)

    def _get_statistical_threshold(
        self,
        scores: List[float],
        top_k: int,
    ) -> float:
        """
        基于分数统计特性计算阈值。

        使用均值和标准差,低于均值-1.5倍标准差视为异常。

        Args:
            scores: 降序排列的分数列表
            top_k: 用户请求的结果数量

        Returns:
            float: 基于统计特性的阈值
        """
        import numpy as np

        mean = np.mean(scores)
        std = np.std(scores)

        # 使用1.5倍标准差(比传统的2倍更严格)
        threshold = max(mean - 1.5 * std, 0.1)

        # 保护下限:不低于0.05
        return max(threshold, 0.05)

    def _calculate_candidate_k(self, normalized_top_k: int, has_time_filter: bool) -> int:
        """
        根据数据集规模和时间过滤动态计算候选数量。

        改进：考虑数据集规模的自适应策略，避免大数据集召回不足。

        Args:
            normalized_top_k: 用户请求的结果数量
            has_time_filter: 是否存在时间约束

        Returns:
            int: 候选数量
        """
        total_items = self.vector_store.get_total_items()

        # 基础乘数：有过滤时扩大候选集
        base_multiplier = 10 if has_time_filter else 5

        # 数据集规模自适应
        if total_items <= 50:
            # 微型数据集：检索全部
            candidate_k = total_items
        elif total_items <= 500:
            # 小数据集：5-10倍
            candidate_k = normalized_top_k * base_multiplier
        elif total_items <= 5000:
            # 中等数据集：3-5倍，最小100
            candidate_k = max(
                normalized_top_k * (base_multiplier - 2),
                100
            )
        else:
            # 大数据集：使用对数缩放，1%或上限500
            candidate_k = max(
                normalized_top_k * 3,
                min(int(total_items * 0.01), 500)
            )

        # 不超过实际数据量
        return min(candidate_k, total_items)

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

        scores = [item["score"] for item in enriched]

        if scores:
            dynamic_threshold = self._calculate_dynamic_threshold(scores, normalized_top_k)
            threshold_filtered = [
                item for item in enriched
                if item["score"] >= dynamic_threshold
            ]
        else:
            threshold_filtered = enriched

        final_results = threshold_filtered[:normalized_top_k]

        for rank, item in enumerate(final_results, start=1):
            item["rank"] = rank
        return final_results

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

    def _parse_date(self, value: str, is_end_date: bool = False) -> Optional[datetime]:
        """
        解析多种日期格式，支持EXIF标准格式。

        改进：支持更多日期格式，包括标准EXIF冒号分隔格式。
        """
        if not value or not isinstance(value, str):
            return None

        # 支持的日期格式列表
        formats = [
            "%Y-%m-%d",                    # 2024-01-01
            "%Y-%m-%dT%H:%M:%S",           # 2024-01-01T08:30:00
            "%Y-%m-%d %H:%M:%S",           # 2024-01-01 08:30:00
            "%Y:%m:%d %H:%M:%S",          # EXIF标准格式: 2024:01:01 08:30:00
            "%Y/%m/%d %H:%M:%S",          # 2024/01/01 08:30:00
            "%Y/%m/%d",                    # 2024/01/01
            "%Y%m%d",                      # 20240101
        ]

        # 移除常见的干扰字符（EXIF可能有空填充）
        cleaned = value.strip().rstrip("\x00")

        for fmt in formats:
            try:
                parsed = datetime.strptime(cleaned, fmt)
                # 如果是日期格式（无时间），设为当天结束23:59:59
                if fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"]:
                    if is_end_date:
                        return datetime(parsed.year, parsed.month, parsed.day, 23, 59, 59)
                return parsed
            except ValueError:
                continue

        # 尝试ISO格式作为最后的回退
        try:
            return datetime.fromisoformat(cleaned)
        except Exception:
            return None
