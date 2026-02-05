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
        调用TimeParser解析时间约束，返回结构化过滤条件。

        返回格式适配 Elasticsearch 过滤：
        - start_date/end_date: 日期范围
        - year/month/day: 精确匹配
        - season: 季节（春天/夏天/秋天/冬天）
        - time_period: 时段（凌晨/早晨/上午/中午/下午/傍晚/夜晚）
        """
        result: Dict[str, Any] = {
            "start_date": None,
            "end_date": None,
            "year": None,
            "month": None,
            "day": None,
            "season": None,
            "time_period": None,
            "precision": "none",
        }

        try:
            constraints = self.time_parser.extract_time_constraints(query)
            if not isinstance(constraints, dict):
                return result

            result["start_date"] = constraints.get("start_date")
            result["end_date"] = constraints.get("end_date")
            result["precision"] = constraints.get("precision", "none")

            # 从查询中提取季节和时段
            season_match = re.search(r"(春天|夏天|秋天|冬天|春季|夏季|秋季|冬季)", query)
            if season_match:
                season_text = season_match.group(1)
                # 统一为"春天"格式
                season_map = {
                    "春天": "春天", "春季": "春天",
                    "夏天": "夏天", "夏季": "夏天",
                    "秋天": "秋天", "秋季": "秋天",
                    "冬天": "冬天", "冬季": "冬天",
                }
                result["season"] = season_map.get(season_text)

            # 从查询中提取时段
            time_period_match = re.search(
                r"(凌晨|早晨|早上|上午|中午|下午|傍晚|晚上|夜晚|深夜)", query
            )
            if time_period_match:
                period_text = time_period_match.group(1)
                # 统一时段名称
                period_map = {
                    "凌晨": "凌晨", "深夜": "凌晨",
                    "早晨": "早晨", "早上": "早晨",
                    "上午": "上午",
                    "中午": "中午",
                    "下午": "下午",
                    "傍晚": "傍晚",
                    "晚上": "夜晚", "夜晚": "夜晚",
                }
                result["time_period"] = period_map.get(period_text)

            # 如果有精确日期，提取年月日
            if result["start_date"] and result["start_date"] == result["end_date"]:
                # 单一日期，提取年月日
                try:
                    dt = datetime.fromisoformat(result["start_date"])
                    result["year"] = dt.year
                    result["month"] = dt.month
                    result["day"] = dt.day
                except Exception:
                    pass

            return result
        except Exception:
            return result

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
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        执行混合检索（向量 + 关键字 + EXIF过滤）。

        改进：
        - 向量检索只基于 description 语义匹配
        - EXIF 条件（时间、季节、时段）通过 Elasticsearch 精确过滤
        - 分数融合时考虑 ES 过滤结果

        Args:
            query: 原始查询文本
            query_embedding: 查询向量
            candidate_k: 候选数量
            filters: EXIF 过滤条件（year, month, season, time_period, start_date, end_date）

        Returns:
            List[Dict[str, Any]]: 混合排序后的结果
        """
        # 1. 向量检索（纯语义匹配）
        vector_results = self.vector_store.search(query_embedding, candidate_k)

        # 2. 构建向量分数映射
        vector_scores: Dict[str, float] = {}
        vector_metadata: Dict[str, Dict[str, Any]] = {}

        for item in vector_results:
            metadata = item.get("metadata") or {}
            photo_path = metadata.get("photo_path", "")
            score = self._distance_to_score(float(item.get("distance", 0.0)))
            vector_scores[photo_path] = score
            vector_metadata[photo_path] = metadata

        # 3. 关键字检索 + EXIF 过滤（如果启用）
        keyword_scores: Dict[str, float] = {}
        es_filtered_paths: Optional[set] = None

        if self.keyword_store is not None:
            # 构建 ES 过滤条件
            es_filters = self._build_es_filters(filters) if filters else {}

            if es_filters:
                # 使用带过滤的搜索
                keyword_results = self.keyword_store.search_with_filters(
                    query, es_filters, candidate_k
                )
                # 记录 ES 返回的路径集合，用于后续过滤
                es_filtered_paths = set()
                for item in keyword_results:
                    keyword_scores[item["photo_path"]] = item["score"]
                    es_filtered_paths.add(item["photo_path"])
            else:
                # 无过滤条件，普通搜索
                keyword_results = self.keyword_store.search(query, candidate_k)
                for item in keyword_results:
                    keyword_scores[item["photo_path"]] = item["score"]

        # 4. 混合评分
        all_paths = set(vector_scores.keys()) | set(keyword_scores.keys())
        combined_results: List[Dict[str, Any]] = []

        for photo_path in all_paths:
            # 如果有 ES 过滤，只保留过滤后的结果
            if es_filtered_paths is not None and photo_path not in es_filtered_paths:
                # 但如果向量检索命中且 ES 没有过滤条件，则保留
                if filters and self._has_strict_filters(filters):
                    continue

            v_score = vector_scores.get(photo_path, 0.0)
            k_score = keyword_scores.get(photo_path, 0.0)

            # 加权融合
            combined_score = (
                self.vector_weight * v_score + self.keyword_weight * k_score
            )

            # 获取元数据
            metadata = vector_metadata.get(photo_path)
            if not metadata:
                metadata = {"photo_path": photo_path, "description": "Keyword Match Only"}

            combined_results.append({
                "photo_path": photo_path,
                "description": metadata.get("description", ""),
                "score": round(combined_score, 6),
                "vector_score": round(v_score, 6),
                "keyword_score": round(k_score, 6),
                "rank": 0,
                "metadata": metadata,
            })

        # 5. 按混合分数降序排序
        combined_results.sort(key=lambda x: x["score"], reverse=True)

        return combined_results

    def _build_es_filters(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        从时间约束构建 ES 过滤条件。

        Args:
            constraints: 时间约束字典

        Returns:
            Dict[str, Any]: ES 过滤条件
        """
        es_filters: Dict[str, Any] = {}

        # 精确字段
        for field in ["year", "month", "day", "season", "time_period"]:
            value = constraints.get(field)
            if value is not None:
                es_filters[field] = value

        # 日期范围
        if constraints.get("start_date"):
            es_filters["start_date"] = constraints["start_date"]
        if constraints.get("end_date"):
            es_filters["end_date"] = constraints["end_date"]

        return es_filters

    def _has_strict_filters(self, filters: Dict[str, Any]) -> bool:
        """
        判断是否有严格的过滤条件（需要排除不满足条件的结果）。
        """
        strict_fields = ["year", "month", "day", "season", "time_period", "start_date", "end_date"]
        return any(filters.get(f) is not None for f in strict_fields)

    def _get_metadata_by_path(self, photo_path: str) -> Optional[Dict[str, Any]]:
        """
        根据照片路径从 vector_store 获取元数据。
        
        Args:
            photo_path: 照片路径
        
        Returns:
            Optional[Dict[str, Any]]: 元数据，未找到返回 None
        """
        if not self.vector_store.metadata:
            return None
        
        for item in self.vector_store.metadata:
            if item.get("photo_path") == photo_path:
                return item
        return None

    def _filter_only_search(
        self, 
        constraints: Dict[str, Any], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        纯过滤查询搜索：跳过向量检索，直接使用 ES 过滤。
        
        当查询只包含时间/季节/时段等过滤条件时调用此方法。
        结果按时间倒序排列（最新的照片优先）。
        
        Args:
            constraints: 过滤条件（year, month, season, time_period, start_date, end_date）
            top_k: 返回数量
        
        Returns:
            List[Dict[str, Any]]: 搜索结果
        """
        if self.keyword_store is None:
            # 无 ES 时降级为内存过滤
            return self._memory_filter_search(constraints, top_k)
        
        # 构建 ES 过滤条件
        es_filters = self._build_es_filters(constraints)
        
        # 使用 ES 执行纯过滤搜索（无文本查询）
        results = self.keyword_store.search_with_filters(
            query=None,  # 无文本查询
            filters=es_filters,
            top_k=top_k * 2  # 多取一些用于后续处理
        )
        
        # 构建返回结果
        final_results = []
        for rank, item in enumerate(results[:top_k], start=1):
            photo_path = item["photo_path"]
            # 从 vector_store 获取完整元数据
            metadata = self._get_metadata_by_path(photo_path)
            final_results.append({
                "photo_path": photo_path,
                "description": metadata.get("description", "") if metadata else "",
                "score": 1.0,  # 纯过滤查询不计算相似度
                "rank": rank,
            })
        
        return final_results

    def _memory_filter_search(
        self,
        constraints: Dict[str, Any],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        内存过滤搜索：当无 ES 时的降级方案。
        
        遍历所有元数据，根据约束条件过滤。
        
        Args:
            constraints: 过滤条件
            top_k: 返回数量
        
        Returns:
            List[Dict[str, Any]]: 搜索结果
        """
        if not self.vector_store.metadata:
            return []
        
        filtered_results = []
        for item in self.vector_store.metadata:
            if self._check_time_match_v2(item, constraints):
                filtered_results.append({
                    "photo_path": item.get("photo_path", ""),
                    "description": item.get("description", ""),
                    "score": 1.0,
                    "rank": 0,
                })
        
        # 按照片路径排序（简单的默认排序）
        filtered_results.sort(key=lambda x: x["photo_path"], reverse=True)
        
        # 设置排名
        for rank, item in enumerate(filtered_results[:top_k], start=1):
            item["rank"] = rank
        
        return filtered_results[:top_k]

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        解析查询、执行混合检索、时间过滤并返回排序结果。

        改进：
        - embedding 只基于纯 description 语义匹配
        - EXIF 条件（时间、季节、时段）通过 Elasticsearch 精确过滤
        - 支持更细粒度的时段查询（7档细分）
        """
        if not self.validate_query(query):
            raise ValueError("查询内容不合法，请输入5-500字符的描述")

        if not self.index_loaded and not self.load_index():
            raise ValueError("索引未加载，请先初始化索引")

        normalized_top_k = max(1, min(int(top_k), 50))
        
        # 1. 查询格式化
        # 架构改进：QueryFormatter 返回纯语义的 search_text（用于 embedding）
        # 时间信息作为独立字段（用于 ES 过滤）
        formatted_query = query
        time_hints = {}
        
        if self.query_formatter is not None and self.query_formatter.is_enabled():
            format_result = self.query_formatter.format_query(query)
            # search_text 是纯视觉语义描述，不含时间信息
            formatted_query = format_result.get("search_text", query)
            # 时间信息作为独立字段保存
            time_hints = {
                "time_hint": format_result.get("time_hint"),
                "season": format_result.get("season"),
                "time_period": format_result.get("time_period"),
            }

        # 2. 时间解析（返回结构化过滤条件）
        constraints: Dict[str, Any] = {
            "start_date": None, "end_date": None,
            "year": None, "month": None, "day": None,
            "season": None, "time_period": None,
            "precision": "none",
        }
        cleaned_query = formatted_query
        # 扩展过滤检测：时间词、时段词、季节词
        has_filter = (
            self._has_time_terms(query) or 
            self._has_time_period_terms(query) or
            self._has_season_terms(query)
        )

        if has_filter:
            constraints = self._extract_time_constraints(query)
            # 清除时间词、时段词、季节词，保留纯语义部分用于 embedding
            if not (self.query_formatter and self.query_formatter.is_enabled()):
                cleaned_query = self._strip_time_terms(query)
                cleaned_query = self._strip_time_period_terms(cleaned_query)
                cleaned_query = self._strip_season_terms(cleaned_query) or query
        
        # 合并 QueryFormatter 的时间提示到 ES 过滤条件
        if time_hints.get("season") and not constraints.get("season"):
            constraints["season"] = time_hints["season"]
        if time_hints.get("time_period") and not constraints.get("time_period"):
            constraints["time_period"] = time_hints["time_period"]

        # 3. 检测是否为纯过滤查询（只有过滤条件，没有视觉内容）
        is_pure_filter = self._is_pure_filter_query(query, cleaned_query)
        
        if is_pure_filter:
            # 纯过滤查询：跳过向量检索，直接使用 ES 过滤
            return self._filter_only_search(constraints, normalized_top_k)

        # 4. 向量检索（仅基于纯语义描述生成 embedding）
        # cleaned_query 不包含时间信息，保证向量空间的纯净性
        # 空值防护：确保 cleaned_query 不为空
        if not cleaned_query or not cleaned_query.strip():
            cleaned_query = "照片 图片 场景"
        
        query_embedding = self.embedding_service.generate_embedding(cleaned_query)
        candidate_k = self._calculate_candidate_k(normalized_top_k, has_filter)

        # 5. 执行检索（混合检索 + ES 过滤）
        if self.keyword_store is not None:
            # 混合检索：向量语义 + ES 关键字 + EXIF 过滤
            combined_results = self._hybrid_search(
                query, query_embedding, candidate_k, filters=constraints
            )
        else:
            # 降级为纯向量检索 + 内存时间过滤
            raw_results = self.vector_store.search(query_embedding, candidate_k)
            combined_results = []
            for item in raw_results:
                metadata = item.get("metadata") or {}
                score = self._distance_to_score(float(item.get("distance", 0.0)))
                combined_results.append({
                    "photo_path": metadata.get("photo_path"),
                    "description": metadata.get("description"),
                    "score": score,
                    "metadata": metadata,
                })

        # 6. 结果过滤链
        filtered_results = []
        for item in combined_results:
            # 如果没有 ES（纯向量模式），需要内存过滤时间
            if self.keyword_store is None and has_filter:
                meta = item.get("metadata", {})
                if not self._check_time_match_v2(meta, constraints):
                    continue
            filtered_results.append(item)

        # 7. 动态阈值过滤
        scores = [item["score"] for item in filtered_results]
        if scores:
            dynamic_threshold = self._calculate_dynamic_threshold(scores, normalized_top_k)
            threshold_filtered = [
                item for item in filtered_results
                if item["score"] >= dynamic_threshold
            ]
        else:
            threshold_filtered = filtered_results

        # 8. 返回 Top-K 结果
        final_results = threshold_filtered[:normalized_top_k]

        for rank, item in enumerate(final_results, start=1):
            item["rank"] = rank
            if "metadata" in item:
                del item["metadata"]
                
        return final_results

    def _has_time_period_terms(self, query: str) -> bool:
        """检查查询中是否包含时段词汇。"""
        patterns = [
            r"凌晨|早晨|早上|上午|中午|下午|傍晚|晚上|夜晚|深夜",
        ]
        return any(re.search(pattern, query) for pattern in patterns)

    def _has_season_terms(self, query: str) -> bool:
        """检查查询中是否包含季节词汇。"""
        pattern = r"春天|夏天|秋天|冬天|春季|夏季|秋季|冬季"
        return bool(re.search(pattern, query))

    def _strip_season_terms(self, query: str) -> str:
        """从查询中移除季节词汇。"""
        pattern = r"春天|夏天|秋天|冬天|春季|夏季|秋季|冬季"
        cleaned = re.sub(pattern, " ", query)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _is_pure_filter_query(self, query: str, cleaned_query: str) -> bool:
        """
        检测是否为纯过滤查询（只有过滤条件，没有视觉内容）。
        
        纯过滤查询特征：
        - 包含时间词汇（年、月、日、去年、今年等）
        - 包含季节词汇（春天、夏天、秋天、冬天）
        - 包含时段词汇（早上、中午、晚上等）
        - 清理后的查询为空或只剩"的照片"、"拍的"等无意义词
        
        Args:
            query: 原始查询
            cleaned_query: 清理时间/季节/时段词汇后的查询
        
        Returns:
            bool: 是否为纯过滤查询
        """
        # 检查是否包含过滤条件
        has_filter = (
            self._has_time_terms(query) or 
            self._has_time_period_terms(query) or
            self._has_season_terms(query)
        )
        
        if not has_filter:
            return False
        
        # 检查清理后是否还有有意义的内容
        # 移除常见的无意义词汇
        noise_patterns = [
            r"的?照片", r"的?图片", r"的?相片", 
            r"拍的", r"拍摄的", r"的$", r"^的"
        ]
        meaningful_text = cleaned_query
        for pattern in noise_patterns:
            meaningful_text = re.sub(pattern, "", meaningful_text)
        meaningful_text = meaningful_text.strip()
        
        # 如果清理后为空或长度过短，认为是纯过滤查询
        return len(meaningful_text) < 2

    def _strip_time_period_terms(self, query: str) -> str:
        """从查询中移除时段词汇。"""
        patterns = [
            r"凌晨|早晨|早上|上午|中午|下午|傍晚|晚上|夜晚|深夜",
        ]
        cleaned = query
        for pattern in patterns:
            cleaned = re.sub(pattern, " ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _check_time_match_v2(self, metadata: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
        """
        检查元数据是否满足时间约束（改进版，支持 time_info 字段）。

        Args:
            metadata: 照片元数据
            constraints: 时间约束

        Returns:
            bool: 是否满足约束
        """
        time_info = metadata.get("time_info") or {}
        exif_data = metadata.get("exif_data") or {}

        # 检查季节
        if constraints.get("season"):
            if time_info.get("season") != constraints["season"]:
                return False

        # 检查时段
        if constraints.get("time_period"):
            if time_info.get("time_period") != constraints["time_period"]:
                return False

        # 检查年份
        if constraints.get("year"):
            if time_info.get("year") != constraints["year"]:
                return False

        # 检查月份
        if constraints.get("month"):
            if time_info.get("month") != constraints["month"]:
                return False

        # 检查日期范围
        start_date = constraints.get("start_date")
        end_date = constraints.get("end_date")
        if start_date or end_date:
            # 获取照片日期
            photo_datetime_str = time_info.get("datetime_str") or exif_data.get("datetime") or metadata.get("file_time")
            if not photo_datetime_str:
                return False

            photo_date = self._parse_date(photo_datetime_str)
            if not photo_date:
                return False

            if start_date:
                start = self._parse_date(start_date)
                if start and photo_date < start:
                    return False

            if end_date:
                end = self._parse_date(end_date, is_end_date=True)
                if end and photo_date > end:
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
