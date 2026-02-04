from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
from utils.vector_store import VectorStore

if TYPE_CHECKING:
    from utils.embedding_service import EmbeddingService
    from utils.time_parser import TimeParser
    from utils.keyword_store import KeywordStore
    from utils.query_formatter import QueryFormatter


class Searcher:
    """
    照片检索器，负责加载索引、查询向量化、相似度检索与时间过滤。
    """

    def __init__(
        self,
        embedding: "EmbeddingService",
        time_parser: "TimeParser",
        vector_store: VectorStore,
        keyword_store: Optional["KeywordStore"] = None,
        query_formatter: Optional["QueryFormatter"] = None,
        data_dir: str = "./data",
        top_k: int = 10,
        vector_weight: float = 0.8,
        keyword_weight: float = 0.2,
    ) -> None:
        """
        初始化检索器并记录索引路径与加载状态。

        Args:
            embedding: 嵌入服务
            time_parser: 时间解析器
            vector_store: 向量存储
            keyword_store: 关键字存储（可选，不传则禁用混合检索）
            query_formatter: 查询格式化服务（可选）
            data_dir: 数据目录
            top_k: 默认返回数量
            vector_weight: 向量检索权重（0-1）
            keyword_weight: 关键字检索权重（0-1）

        Raises:
            ValueError: 权重之和不为 1 时抛出
        """
        if abs(vector_weight + keyword_weight - 1.0) > 0.001:
            raise ValueError("vector_weight + keyword_weight 必须等于 1.0")

        self.embedding_service = embedding
        self.time_parser = time_parser
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.query_formatter = query_formatter
        self.data_dir = data_dir
        self.top_k = max(1, top_k)
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
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

    def _hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        candidate_k: int,
    ) -> List[Dict[str, Any]]:
        """
        执行混合检索（向量 + 关键字）。

        Args:
            query: 原始查询文本
            query_embedding: 查询向量
            candidate_k: 候选数量

        Returns:
            List[Dict[str, Any]]: 混合排序后的结果
        """
        # 1. 向量检索
        vector_results = self.vector_store.search(query_embedding, candidate_k)

        # 2. 构建向量分数映射
        vector_scores: Dict[str, float] = {}
        # 同时保存 vector result 的 metadata，后续使用
        vector_metadata: Dict[str, Dict[str, Any]] = {}

        for item in vector_results:
            metadata = item.get("metadata") or {}
            photo_path = metadata.get("photo_path", "")
            score = self._distance_to_score(float(item.get("distance", 0.0)))
            vector_scores[photo_path] = score
            vector_metadata[photo_path] = metadata

        # 3. 关键字检索（如果启用）
        keyword_scores: Dict[str, float] = {}
        if self.keyword_store is not None:
            keyword_results = self.keyword_store.search(query, candidate_k)
            for item in keyword_results:
                keyword_scores[item["photo_path"]] = item["score"]

        # 4. 混合评分
        all_paths = set(vector_scores.keys()) | set(keyword_scores.keys())
        combined_results: List[Dict[str, Any]] = []

        for photo_path in all_paths:
            v_score = vector_scores.get(photo_path, 0.0)
            k_score = keyword_scores.get(photo_path, 0.0)

            # 加权融合
            combined_score = (
                self.vector_weight * v_score + self.keyword_weight * k_score
            )

            # 获取元数据 (优先从 vector_metadata 获取，如果没有则构造基础 metadata)
            metadata = vector_metadata.get(photo_path)
            if not metadata:
                # 只有关键字命中的情况 (metadata不全，可能有风险，但至少返回路径)
                # 注意：如果 metadata 不全，时间过滤可能会有问题（如果没有 timestamp）
                metadata = {"photo_path": photo_path, "description": "Keyword Match Only"}

            combined_results.append({
                "photo_path": photo_path,
                "description": metadata.get("description", ""),
                "score": round(combined_score, 6),
                "vector_score": round(v_score, 6),
                "keyword_score": round(k_score, 6),
                "rank": 0, # Placeholder
                "metadata": metadata # Pass full metadata for downstream filtering
            })

        # 5. 按混合分数降序排序
        combined_results.sort(key=lambda x: x["score"], reverse=True)

        return combined_results

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        解析查询、执行混合检索、时间过滤并返回排序结果。
        """
        if not self.validate_query(query):
            raise ValueError("查询内容不合法，请输入5-500字符的描述")

        if not self.index_loaded and not self.load_index():
            raise ValueError("索引未加载，请先初始化索引")

        normalized_top_k = max(1, min(int(top_k), 50))
        
        # 1. 查询格式化
        formatted_query = query
        time_hints = {}
        
        if self.query_formatter is not None and self.query_formatter.is_enabled():
            format_result = self.query_formatter.format_query(query)
            formatted_query = format_result.get("search_text", query)
            time_hints = {
                "time_hint": format_result.get("time_hint"),
                "season": format_result.get("season"),
            }
            # 如果格式化结果有时间，可以辅助时间解析（这部分逻辑可根据需求扩展）

        # 2. 时间解析
        constraints = {"start_date": None, "end_date": None, "precision": "none"}
        cleaned_query = formatted_query  # 默认使用格式化后的查询
        has_time_filter = self._has_time_terms(query) # 仍使用原始查询检查时间词

        if has_time_filter:
            constraints = self._extract_time_constraints(query)
            # 如果没有格式化服务，或者即使有格式化也需要手动清除时间词（格式化服务通常已经处理好了，但双重保险）
            if not (self.query_formatter and self.query_formatter.is_enabled()):
                cleaned_query = self._strip_time_terms(query) or query
        
        # 3. 向量检索
        query_embedding = self.embedding_service.generate_embedding(cleaned_query)
        candidate_k = self._calculate_candidate_k(normalized_top_k, has_time_filter)

        # 执行检索 (混合 or 纯向量)
        if self.keyword_store is not None:
             # 混合检索使用原始查询(keyword matching)和向量
            combined_results = self._hybrid_search(
                query, query_embedding, candidate_k
            )
        else:
            # 降级为纯向量检索
            raw_results = self.vector_store.search(query_embedding, candidate_k)
            combined_results = []
            for item in raw_results:
                metadata = item.get("metadata") or {}
                score = self._distance_to_score(float(item.get("distance", 0.0)))
                combined_results.append({
                    "photo_path": metadata.get("photo_path"),
                    "description": metadata.get("description"),
                    "score": score,
                    "metadata": metadata # Keep consistency
                })

        # 结果过滤链：时间过滤 -> 动态阈值 -> 数量限制
        
        filtered_results = []
        for item in combined_results:
            # Time Filter
            if has_time_filter:
                meta = item.get("metadata", {})
                if not self._check_time_match(meta, constraints):
                    continue
            
            filtered_results.append(item)

        scores = [item["score"] for item in filtered_results]

        if scores:
            dynamic_threshold = self._calculate_dynamic_threshold(scores, normalized_top_k)
            threshold_filtered = [
                item for item in filtered_results
                if item["score"] >= dynamic_threshold
            ]
        else:
            threshold_filtered = filtered_results

        final_results = threshold_filtered[:normalized_top_k]

        for rank, item in enumerate(final_results, start=1):
            item["rank"] = rank
            # Remove internal metadata field before returning to keep API clean
            if "metadata" in item:
                del item["metadata"]
                
        return final_results

    def _check_time_match(self, metadata: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
        """Helper to check time constraints directly on metadata dict."""
        # 提取时间
        exif_date = self._parse_date(metadata.get("exif_date"))
        file_date = self._parse_date(metadata.get("file_date"))
        
        # 优先使用EXIF时间
        check_date = exif_date or file_date
        
        if not check_date:
            return False
            
        start_date = constraints.get("start_date")
        end_date = constraints.get("end_date")
        
        if start_date and check_date < start_date:
            return False
        if end_date and check_date > end_date:
            return False
            
        return True


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
