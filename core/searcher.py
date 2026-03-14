from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
from utils.path_utils import normalize_local_path, same_file_path
from utils.structured_analysis import build_match_summary
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
        query_expansion_enabled: bool = True,
        query_expansion_max_alternatives: int = 2,
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
        self.query_expansion_enabled = bool(query_expansion_enabled)
        self.query_expansion_max_alternatives = max(0, int(query_expansion_max_alternatives))
        self.index_loaded = False
        self.index_path = vector_store.index_path
        self.metadata_path = vector_store.metadata_path
        self.metric = getattr(vector_store, "metric", "cosine")
        self._metadata_by_path: Dict[str, Dict[str, Any]] = {}
        self._last_search_debug: Dict[str, Any] = self._empty_search_debug()
        self._refresh_metadata_cache()

    @staticmethod
    def _empty_search_debug() -> Dict[str, Any]:
        return {
            "mode": "text",
            "base_intent": {},
            "expansion_triggered": False,
            "expansion_reason": "",
            "alternatives": [],
            "reflection_triggered": False,
            "reflection_reason": "",
            "reflection": {},
            "rounds": [],
        }

    @staticmethod
    def _path_key(photo_path: str) -> str:
        normalized = normalize_local_path(photo_path) if photo_path else ""
        if not normalized and photo_path:
            normalized = str(photo_path).strip()
        return os.path.normcase(normalized)

    @staticmethod
    def _round_summary(
        *,
        round_name: str,
        intent: Dict[str, Any],
        results: List[Dict[str, Any]],
        reason: str = "",
    ) -> Dict[str, Any]:
        top_score = float(results[0].get("score", 0.0)) if results else 0.0
        return {
            "round": round_name,
            "reason": reason,
            "intent": {
                "search_text": str(intent.get("search_text") or "").strip(),
                "media_terms": list(intent.get("media_terms") or []),
                "identity_terms": list(intent.get("identity_terms") or []),
                "strict_identity_filter": bool(intent.get("strict_identity_filter", False)),
                "time_hint": intent.get("time_hint"),
                "season": intent.get("season"),
                "time_period": intent.get("time_period"),
            },
            "result_count": len(results),
            "top_score": round(top_score, 6),
        }

    def get_last_search_debug(self) -> Dict[str, Any]:
        return dict(self._last_search_debug)

    def _set_last_search_debug(self, debug: Dict[str, Any]) -> None:
        self._last_search_debug = debug

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: Dict[str, Dict[str, Any]] = {}
        ordered_keys: List[str] = []

        for item in results:
            photo_path = item.get("photo_path")
            key = self._path_key(photo_path)
            if not key:
                continue

            existing = deduped.get(key)
            if existing is None:
                deduped[key] = item
                ordered_keys.append(key)
                continue

            if float(item.get("score", 0.0)) > float(existing.get("score", 0.0)):
                deduped[key] = item

        return [deduped[key] for key in ordered_keys]

    def _refresh_metadata_cache(self) -> None:
        cache: Dict[str, Dict[str, Any]] = {}
        for item in self.vector_store.metadata or []:
            photo_path = item.get("photo_path")
            if photo_path:
                cache[photo_path] = item
        self._metadata_by_path = cache

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
        self._refresh_metadata_cache()
        return True

    def validate_query(self, query: str) -> bool:
        """
        校验查询文本长度与有效字符。
        """
        if not isinstance(query, str):
            return False
        text = query.strip()
        if len(text) < 1 or len(text) > 500:
            return False
        if all(not char.isalnum() and not ("\u4e00" <= char <= "\u9fff") for char in text):
            return False
        if len(text) == 1 and text.isascii() and text.isalpha():
            return False
        return True

    @staticmethod
    def _build_query_text(
        search_text: str,
        media_terms: List[str],
        identity_terms: List[str],
        original_query: str,
    ) -> str:
        parts: List[str] = []
        if search_text.strip():
            parts.append(search_text.strip())
        if media_terms:
            parts.append(" ".join(media_terms))
        if identity_terms:
            parts.append(" ".join(identity_terms))
        query_text = " ".join(parts).strip()
        return query_text or original_query.strip()

    @staticmethod
    def _compute_metadata_boost(
        metadata: Dict[str, Any],
        media_terms: List[str],
        identity_terms: List[str],
    ) -> float:
        boost = 1.0
        metadata_media = set(metadata.get("media_types") or [])
        metadata_identities = set(metadata.get("identity_names") or [])
        if media_terms and metadata_media.intersection(media_terms):
            boost += 0.18
        if identity_terms and metadata_identities.intersection(identity_terms):
            boost += 0.28
        return boost

    @staticmethod
    def _candidate_matches_identity_terms(
        metadata: Dict[str, Any],
        identity_terms: List[str],
    ) -> bool:
        if not identity_terms:
            return True

        normalized_terms = {term.strip().lower() for term in identity_terms if term and term.strip()}
        if not normalized_terms:
            return True

        identity_names = {
            str(name).strip().lower()
            for name in (metadata.get("identity_names") or [])
            if str(name).strip()
        }
        if identity_names.intersection(normalized_terms):
            return True

        for candidate in metadata.get("identity_candidates") or []:
            if not isinstance(candidate, dict):
                continue
            names = [candidate.get("name")] + list(candidate.get("aliases") or [])
            normalized_names = {
                str(name).strip().lower()
                for name in names
                if str(name).strip()
            }
            if normalized_names.intersection(normalized_terms):
                return True
        return False

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
            timestamp = exif_data.get("datetime")
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

    @staticmethod
    def _should_expand_results(results: List[Dict[str, Any]], top_k: int) -> bool:
        if not results:
            return True
        top_score = float(results[0].get("score", 0.0))
        if top_score < 0.55:
            return True
        if len(results) < min(top_k, 3) and top_score < 0.72:
            return True
        return False

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

    def _hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        candidate_k: int,
        filters: Optional[Dict[str, Any]] = None,
        allow_keyword_only_results: bool = False,
        media_terms: Optional[List[str]] = None,
        identity_terms: Optional[List[str]] = None,
        strict_identity_filter: bool = False,
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
        media_terms = media_terms or []
        identity_terms = identity_terms or []

        # 1. 向量检索（基于 retrieval_text 的语义匹配）
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
            keyword_candidate_k = max(1, min(candidate_k, max(self.top_k * 3, 15)))
            # 构建 ES 过滤条件
            es_filters = self._build_es_filters(filters) if filters else {}

            if es_filters:
                # 使用带过滤的搜索
                keyword_results = self.keyword_store.search_with_filters(
                    query, es_filters, keyword_candidate_k
                )
                # 记录 ES 返回的路径集合，用于后续过滤
                es_filtered_paths = set()
                for item in keyword_results:
                    keyword_scores[item["photo_path"]] = item["score"]
                    es_filtered_paths.add(item["photo_path"])
            else:
                # 无过滤条件，普通搜索
                keyword_results = self.keyword_store.search(query, keyword_candidate_k)
                for item in keyword_results:
                    keyword_scores[item["photo_path"]] = item["score"]

        # 4. 混合评分
        all_paths = set(vector_scores.keys())
        if allow_keyword_only_results:
            all_paths |= set(keyword_scores.keys())
        combined_results: List[Dict[str, Any]] = []

        for photo_path in all_paths:
            # 如果有 ES 过滤，只保留过滤后的结果
            if es_filtered_paths is not None and photo_path not in es_filtered_paths:
                # 但如果向量检索命中且 ES 没有过滤条件，则保留
                if filters and self._has_strict_filters(filters):
                    continue

            v_score = vector_scores.get(photo_path, 0.0)
            k_score = keyword_scores.get(photo_path, 0.0)
            has_vector = photo_path in vector_scores
            has_keyword = photo_path in keyword_scores

            # 搜索结果必须来自当前本地向量索引，避免 ES 中历史脏文档或失效路径进入结果。
            metadata = self._get_metadata_by_path(photo_path)
            if metadata is None:
                continue
            normalized_path = normalize_local_path(photo_path)
            if normalized_path and not os.path.exists(normalized_path):
                continue
            if strict_identity_filter and identity_terms and not self._candidate_matches_identity_terms(metadata, identity_terms):
                continue

            # 只按命中的检索通道做归一融合。
            # 这样当图片没有 BM25 命中时，不会因为 keyword_score=0 被无端压分。
            available_weight = 0.0
            weighted_score = 0.0
            if has_vector:
                available_weight += self.vector_weight
                weighted_score += self.vector_weight * v_score
            if has_keyword:
                available_weight += self.keyword_weight
                weighted_score += self.keyword_weight * k_score
            if available_weight <= 0:
                continue
            combined_score = weighted_score / available_weight
            combined_score *= self._compute_metadata_boost(metadata, media_terms, identity_terms)

            # 若仅命中关键词且向量完全缺失，只作为弱候选保留，避免 BM25 单独把无关结果顶到前面。
            if has_keyword and not has_vector:
                combined_score *= 0.65

            # 无时间过滤时，纯关键词候选必须达到更高阈值才允许进入结果，
            # 避免大量“文件名/BM25 命中但视觉无关”的项目污染结果集。
            if has_keyword and not has_vector and es_filtered_paths is None and k_score < 0.45:
                continue

            combined_results.append({
                "photo_path": photo_path,
                "description": metadata.get("description", ""),
                "score": round(combined_score, 6),
                "vector_score": round(v_score, 6),
                "keyword_score": round(k_score, 6),
                "rank": 0,
                "metadata": metadata,
                "match_summary": build_match_summary(metadata),
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
        if not self._metadata_by_path:
            self._refresh_metadata_cache()
        return self._metadata_by_path.get(photo_path)

    def _filter_only_search(
        self, 
        query: Optional[str],
        constraints: Dict[str, Any], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        纯过滤查询搜索：利用 ES 关键字检索与过滤条件。
        
        当查询只包含时间/季节/时段等过滤条件时调用此方法。
        
        Args:
            query: 纯文本查询，可为空；纯 EXIF 过滤时应传 None
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
        
        # 使用 ES 执行带关键字的过滤搜索
        results = self.keyword_store.search_with_filters(
            query=query,
            filters=es_filters,
            top_k=top_k * 2  # 多取一些用于后续处理
        )
        
        # 降级机制：如果ES检索结果为空，且有过滤条件，尝试使用内存元数据进行兜底
        if not results and self.vector_store.metadata:
            print(f"[WARN] ES检索结果为空，尝试降级到内存元数据检索。query: {query}, filters: {constraints}")
            return self._memory_filter_search(constraints, top_k)
        
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
                "match_summary": build_match_summary(metadata or {}),
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
                    "match_summary": build_match_summary(item),
                })
        
        # 按照片路径排序（简单的默认排序）
        filtered_results.sort(key=lambda x: x["photo_path"], reverse=True)
        
        # 设置排名
        for rank, item in enumerate(filtered_results[:top_k], start=1):
            item["rank"] = rank

        return filtered_results[:top_k]

    def _vector_results_to_combined(
        self,
        raw_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        combined_results = []
        for item in raw_results:
            metadata = item.get("metadata") or {}
            photo_path = metadata.get("photo_path")
            normalized_path = normalize_local_path(photo_path) if photo_path else ""
            if not photo_path or not normalized_path or not os.path.exists(normalized_path):
                continue
            score = self._distance_to_score(float(item.get("distance", 0.0)))
            combined_results.append(
                {
                    "photo_path": photo_path,
                    "description": metadata.get("description"),
                    "retrieval_text": metadata.get("retrieval_text"),
                    "score": score,
                    "metadata": metadata,
                    "match_summary": build_match_summary(metadata),
                }
            )
        return self._deduplicate_results(combined_results)

    def _run_single_search_round(
        self,
        *,
        query: str,
        intent: Dict[str, Any],
        embedding_query: str,
        media_terms: List[str],
        identity_terms: List[str],
        strict_identity_filter: bool,
        constraints: Dict[str, Any],
        normalized_top_k: int,
        has_filter: bool,
    ) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_service.generate_embedding(embedding_query)
        candidate_k = self._calculate_candidate_k(normalized_top_k, has_filter)

        if self.keyword_store is not None:
            combined_results = self._hybrid_search(
                embedding_query,
                query_embedding,
                candidate_k,
                filters=constraints,
                allow_keyword_only_results=False,
                media_terms=media_terms,
                identity_terms=identity_terms,
                strict_identity_filter=strict_identity_filter,
            )
        else:
            raw_results = self.vector_store.search(query_embedding, candidate_k)
            combined_results = self._vector_results_to_combined(raw_results)

        return self._finalize_results(
            combined_results=combined_results,
            normalized_top_k=normalized_top_k,
            has_filter=has_filter,
            constraints=constraints,
            identity_terms=identity_terms,
            strict_identity_filter=strict_identity_filter,
        )

    def _maybe_reflect_query_results(
        self,
        *,
        query: str,
        base_intent: Dict[str, Any],
        current_results: List[Dict[str, Any]],
        normalized_top_k: int,
        constraints: Dict[str, Any],
        has_filter: bool,
        debug: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not self.query_formatter or not self.query_formatter.is_enabled():
            return current_results
        if not self._should_expand_results(current_results, normalized_top_k):
            return current_results

        reflection = self.query_formatter.reflect_on_weak_results(
            user_query=query,
            base_intent=base_intent,
            weak_results=current_results,
        )
        if not reflection:
            return current_results

        embedding_query = self._build_query_text(
            search_text=str(reflection.get("search_text") or ""),
            media_terms=list(reflection.get("media_terms") or []),
            identity_terms=list(reflection.get("identity_terms") or []),
            original_query=query,
        )
        reflected_results = self._run_single_search_round(
            query=query,
            intent=reflection,
            embedding_query=embedding_query,
            media_terms=list(reflection.get("media_terms") or []),
            identity_terms=list(reflection.get("identity_terms") or []),
            strict_identity_filter=bool(reflection.get("strict_identity_filter", False)),
            constraints=constraints,
            normalized_top_k=normalized_top_k,
            has_filter=has_filter,
        )
        if not reflected_results:
            return current_results

        debug["reflection_triggered"] = True
        debug["reflection_reason"] = str(reflection.get("reason") or "").strip()
        debug["reflection"] = dict(reflection)
        debug["rounds"].append(
            self._round_summary(
                round_name="reflection",
                intent=reflection,
                results=reflected_results,
                reason=str(reflection.get("reason") or "").strip(),
            )
        )

        if self._should_expand_results(reflected_results, normalized_top_k):
            return current_results
        return reflected_results

    def _maybe_expand_query_results(
        self,
        *,
        query: str,
        base_intent: Dict[str, Any],
        base_results: List[Dict[str, Any]],
        normalized_top_k: int,
        constraints: Dict[str, Any],
        has_filter: bool,
        debug: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not self.query_formatter or not self.query_formatter.is_enabled():
            return base_results
        if not self.query_expansion_enabled or self.query_expansion_max_alternatives <= 0:
            return base_results
        if not self._should_expand_results(base_results, normalized_top_k):
            return base_results

        alternatives = self.query_formatter.expand_query_intents(
            user_query=query,
            base_intent=base_intent,
            max_alternatives=self.query_expansion_max_alternatives,
        )
        if not alternatives:
            return base_results

        debug["expansion_triggered"] = True
        merged: List[Dict[str, Any]] = [dict(item) for item in base_results]
        best_results: List[Dict[str, Any]] = base_results
        for alt in alternatives:
            embedding_query = self._build_query_text(
                search_text=str(alt.get("search_text") or ""),
                media_terms=list(alt.get("media_terms") or []),
                identity_terms=list(alt.get("identity_terms") or []),
                original_query=query,
            )
            alt_results = self._run_single_search_round(
                query=query,
                intent=alt,
                embedding_query=embedding_query,
                media_terms=list(alt.get("media_terms") or []),
                identity_terms=list(alt.get("identity_terms") or []),
                strict_identity_filter=bool(alt.get("strict_identity_filter", False)),
                constraints=constraints,
                normalized_top_k=normalized_top_k,
                has_filter=has_filter,
            )
            debug["alternatives"].append(dict(alt))
            debug["rounds"].append(
                self._round_summary(
                    round_name="expansion",
                    intent=alt,
                    results=alt_results,
                    reason=str(alt.get("reason") or "").strip(),
                )
            )
            if alt_results:
                current_best_score = float(best_results[0].get("score", 0.0)) if best_results else 0.0
                alt_best_score = float(alt_results[0].get("score", 0.0))
                if alt_best_score > current_best_score:
                    best_results = alt_results
            merged.extend(dict(item) for item in alt_results)

        merged = self._deduplicate_results(merged)
        merged.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        final_results = merged[:normalized_top_k]
        for rank, item in enumerate(final_results, start=1):
            item["rank"] = rank

        expansion_reason = ""
        if debug["alternatives"]:
            expansion_reason = "第一轮结果偏弱，尝试保守扩写查询意图"
        debug["expansion_reason"] = expansion_reason

        if self._should_expand_results(final_results, normalized_top_k):
            reflected = self._maybe_reflect_query_results(
                query=query,
                base_intent=base_intent,
                current_results=best_results,
                normalized_top_k=normalized_top_k,
                constraints=constraints,
                has_filter=has_filter,
                debug=debug,
            )
            if reflected is not best_results:
                return reflected
        return final_results

    def _finalize_results(
        self,
        combined_results: List[Dict[str, Any]],
        normalized_top_k: int,
        has_filter: bool,
        constraints: Dict[str, Any],
        identity_terms: Optional[List[str]] = None,
        strict_identity_filter: bool = False,
    ) -> List[Dict[str, Any]]:
        filtered_results = []
        identity_terms = identity_terms or []
        for item in combined_results:
            if self.keyword_store is None and has_filter:
                meta = item.get("metadata", {})
                if not self._check_time_match_v2(meta, constraints):
                    continue
            if strict_identity_filter and identity_terms:
                meta = item.get("metadata", {})
                if not self._candidate_matches_identity_terms(meta, identity_terms):
                    continue
            filtered_results.append(item)

        filtered_results = self._deduplicate_results(filtered_results)

        scores = [item["score"] for item in filtered_results]
        if scores:
            dynamic_threshold = self._calculate_dynamic_threshold(scores, normalized_top_k)
            threshold_filtered = [
                item for item in filtered_results if item["score"] >= dynamic_threshold
            ]
        else:
            threshold_filtered = filtered_results

        final_results = threshold_filtered[:normalized_top_k]
        for rank, item in enumerate(final_results, start=1):
            item["rank"] = rank
            if "metadata" in item:
                del item["metadata"]
        return final_results

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        解析查询、执行混合检索、时间过滤并返回排序结果。

        改进：
        - embedding 改为基于 retrieval_text 风格的组合查询文本
        - EXIF 条件（时间、季节、时段）通过 Elasticsearch 精确过滤
        - 支持更细粒度的时段查询（7档细分）
        """
        if not self.validate_query(query):
            raise ValueError("查询内容不合法，请输入1-500字符的描述")

        if not self.index_loaded and not self.load_index():
            raise ValueError("索引未加载，请先初始化索引")

        normalized_top_k = max(1, min(int(top_k), 50))
        debug = self._empty_search_debug()
        debug["mode"] = "text"

        # 1. 查询格式化
        # 优先信任 QueryFormatter 的 LLM 输出：
        # - search_text: 可向量化的视觉语义
        # - season/time_period/time_hint: 仅用于 EXIF/时间过滤
        query_formatter_enabled = bool(
            self.query_formatter is not None and self.query_formatter.is_enabled()
        )
        formatted_query = query.strip()
        media_terms: List[str] = []
        identity_terms: List[str] = []
        strict_identity_filter = False
        time_hints = {}

        if query_formatter_enabled:
            format_result = self.query_formatter.format_query(query)
            formatted_query = (format_result.get("search_text") or "").strip()
            media_terms = list(format_result.get("media_terms") or [])
            identity_terms = list(format_result.get("identity_terms") or [])
            strict_identity_filter = bool(format_result.get("strict_identity_filter", False))
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
        has_filter = bool(
            time_hints.get("time_hint")
            or time_hints.get("season")
            or time_hints.get("time_period")
        )

        if has_filter:
            constraints = self._extract_time_constraints(query)

        # 合并 QueryFormatter 的时间提示到 ES 过滤条件
        if time_hints.get("season") and not constraints.get("season"):
            constraints["season"] = time_hints["season"]
        if time_hints.get("time_period") and not constraints.get("time_period"):
            constraints["time_period"] = time_hints["time_period"]

        # 3. 只有“纯 EXIF/时间过滤查询”才降级为纯关键字检索。
        # 其判定依据是：LLM 没有抽出任何视觉语义，且确实存在过滤条件。
        is_filter_only_query = query_formatter_enabled and not cleaned_query and has_filter
        if is_filter_only_query:
            # 纯过滤查询：跳过向量检索，直接使用 ES 关键字与过滤检索
            print(f"[DEBUG] 纯过滤查询模式，constraints: {constraints}")
            filter_only_intent = {
                "search_text": cleaned_query,
                "media_terms": list(media_terms),
                "identity_terms": list(identity_terms),
                "strict_identity_filter": strict_identity_filter,
                "time_hint": time_hints.get("time_hint"),
                "season": time_hints.get("season"),
                "time_period": time_hints.get("time_period"),
            }
            results = self._filter_only_search(None, constraints, normalized_top_k)
            debug["base_intent"] = dict(filter_only_intent)
            debug["rounds"].append(
                self._round_summary(
                    round_name="base",
                    intent=filter_only_intent,
                    results=results,
                    reason="纯时间过滤查询",
                )
            )
            self._set_last_search_debug(debug)
            return results

        # 4. 向量检索（仅基于纯语义描述生成 embedding）
        # 除纯过滤查询外，其余一律走混合检索。
        # search_text 不做发散改写，直接使用格式化器从原 query 中抽出的视觉片段。
        # 若格式化器异常未产出，则仅回退到原始 query，避免把普通查询误判成纯过滤。
        embedding_query = self._build_query_text(
            search_text=cleaned_query,
            media_terms=media_terms,
            identity_terms=identity_terms,
            original_query=query,
        )
        base_intent = {
            "search_text": cleaned_query,
            "media_terms": list(media_terms),
            "identity_terms": list(identity_terms),
            "strict_identity_filter": strict_identity_filter,
            "time_hint": time_hints.get("time_hint"),
            "season": time_hints.get("season"),
            "time_period": time_hints.get("time_period"),
            "original_query": query,
        }
        debug["base_intent"] = dict(base_intent)
        first_round_results = self._run_single_search_round(
            query=query,
            intent=base_intent,
            embedding_query=embedding_query,
            media_terms=media_terms,
            identity_terms=identity_terms,
            strict_identity_filter=strict_identity_filter,
            constraints=constraints,
            normalized_top_k=normalized_top_k,
            has_filter=has_filter,
        )
        debug["rounds"].append(
            self._round_summary(
                round_name="base",
                intent=base_intent,
                results=first_round_results,
            )
        )
        final_results = self._maybe_expand_query_results(
            query=query,
            base_intent=base_intent,
            base_results=first_round_results,
            normalized_top_k=normalized_top_k,
            constraints=constraints,
            has_filter=has_filter,
            debug=debug,
        )
        self._set_last_search_debug(debug)
        return final_results

    def search_by_image_path(self, image_path: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.index_loaded and not self.load_index():
            raise ValueError("索引未加载，请先初始化索引")

        normalized_path = normalize_local_path(image_path)
        if not normalized_path or not os.path.isabs(normalized_path):
            raise ValueError("图片路径必须为绝对路径")

        query_embedding = self.vector_store.get_embedding_by_photo_path(normalized_path)
        if query_embedding is None:
            for metadata in self.vector_store.metadata:
                candidate_path = metadata.get("photo_path")
                if candidate_path and same_file_path(candidate_path, normalized_path):
                    query_embedding = self.vector_store.get_embedding_by_photo_path(candidate_path)
                    normalized_path = candidate_path
                    break

        if query_embedding is None:
            raise ValueError("图片路径未建立索引，请先重建索引或确认路径存在于数据库中")

        normalized_top_k = max(1, min(int(top_k), 50))
        debug = self._empty_search_debug()
        debug["mode"] = "text"
        candidate_k = min(
            self.vector_store.get_total_items(),
            max(normalized_top_k + 1, normalized_top_k * 5),
        )
        raw_results = self.vector_store.search(query_embedding, candidate_k)
        combined_results = self._vector_results_to_combined(raw_results)

        filtered = [
            item
            for item in combined_results
            if item.get("photo_path") and not same_file_path(item["photo_path"], normalized_path)
        ]
        filtered = self._deduplicate_results(filtered)

        for rank, item in enumerate(filtered[:normalized_top_k], start=1):
            item["rank"] = rank
            if "metadata" in item:
                del item["metadata"]
        results = filtered[:normalized_top_k]
        self._set_last_search_debug(
            {
                "mode": "image_path",
                "base_intent": {"image_path": normalized_path},
                "expansion_triggered": False,
                "expansion_reason": "",
                "alternatives": [],
                "reflection_triggered": False,
                "reflection_reason": "",
                "reflection": {},
                "rounds": [
                    {
                        "round": "base",
                        "reason": "按参考图 embedding 检索相似图片",
                        "intent": {"image_path": normalized_path},
                        "result_count": len(results),
                        "top_score": round(float(results[0].get("score", 0.0)), 6) if results else 0.0,
                    }
                ],
            }
        )
        return results

    def search_by_uploaded_image(
        self,
        image_path: str,
        analysis: Dict[str, Any],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        if not self.index_loaded and not self.load_index():
            raise ValueError("索引未加载，请先初始化索引")

        normalized_path = normalize_local_path(image_path)
        if not normalized_path or not os.path.isabs(normalized_path):
            raise ValueError("上传图片路径必须为绝对路径")
        if not os.path.exists(normalized_path):
            raise ValueError("上传图片不存在")

        retrieval_text = str((analysis or {}).get("retrieval_text") or "").strip()
        if not retrieval_text:
            retrieval_text = str((analysis or {}).get("description") or "").strip()
        if not retrieval_text:
            raise ValueError("上传图片分析结果为空，无法进行相似图检索")

        query_embedding = self.embedding_service.generate_embedding(retrieval_text)
        normalized_top_k = max(1, min(int(top_k), 50))
        candidate_k = min(
            self.vector_store.get_total_items(),
            max(normalized_top_k * 5, normalized_top_k + 5),
        )
        raw_results = self.vector_store.search(query_embedding, candidate_k)
        combined_results = self._vector_results_to_combined(raw_results)

        filtered = [
            item
            for item in combined_results
            if item.get("photo_path") and not same_file_path(item["photo_path"], normalized_path)
        ]
        filtered = self._deduplicate_results(filtered)

        for rank, item in enumerate(filtered[:normalized_top_k], start=1):
            item["rank"] = rank
            if "metadata" in item:
                del item["metadata"]
        results = filtered[:normalized_top_k]
        self._set_last_search_debug(
            {
                "mode": "uploaded_image",
                "base_intent": {
                    "image_path": normalized_path,
                    "retrieval_text": retrieval_text,
                },
                "expansion_triggered": False,
                "expansion_reason": "",
                "alternatives": [],
                "reflection_triggered": False,
                "reflection_reason": "",
                "reflection": {},
                "rounds": [
                    {
                        "round": "base",
                        "reason": "按上传图片分析结果生成 embedding 检索相似图片",
                        "intent": {"retrieval_text": retrieval_text},
                        "result_count": len(results),
                        "top_score": round(float(results[0].get("score", 0.0)), 6) if results else 0.0,
                    }
                ],
            }
        )
        return results

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
        exif_datetime = exif_data.get("datetime")

        # 检查季节
        if constraints.get("season"):
            if not exif_datetime:
                return False
            if time_info.get("season") != constraints["season"]:
                return False

        # 检查时段
        if constraints.get("time_period"):
            if not exif_datetime:
                return False
            if time_info.get("time_period") != constraints["time_period"]:
                return False

        # 检查年份
        if constraints.get("year"):
            if not exif_datetime:
                return False
            if time_info.get("year") != constraints["year"]:
                return False

        # 检查月份
        if constraints.get("month"):
            if not exif_datetime:
                return False
            if time_info.get("month") != constraints["month"]:
                return False

        # 检查日期范围
        start_date = constraints.get("start_date")
        end_date = constraints.get("end_date")
        if start_date or end_date:
            # 获取照片日期
            photo_datetime_str = time_info.get("datetime_str") or exif_datetime
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
