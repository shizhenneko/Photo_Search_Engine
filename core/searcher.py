from __future__ import annotations

import os
import time
from math import ceil
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


MIN_RESULT_SCORE = 0.4


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
        query_multi_round_enabled: bool = False,
        query_reflection_enabled: bool = False,
        query_max_reflection_rounds: int = 2,
        query_dynamic_threshold_floor: float = 0.05,
        query_strict_floor_min: float = 0.22,
        query_broad_floor_min: float = 0.12,
        time_parse_strategy: str = "local_first",
        validate_file_exists: bool = False,
        query_cache_enabled: bool = True,
        query_cache_size: int = 2000,
        embedding_cache_enabled: bool = True,
        embedding_cache_size: int = 5000,
        default_search_mode: str = "balanced",
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
        self.query_multi_round_enabled = bool(query_multi_round_enabled)
        self.query_reflection_enabled = bool(query_reflection_enabled)
        self.query_max_reflection_rounds = max(0, int(query_max_reflection_rounds))
        self.query_dynamic_threshold_floor = max(0.0, min(1.0, float(query_dynamic_threshold_floor)))
        self.query_strict_floor_min = max(0.0, min(1.0, float(query_strict_floor_min)))
        self.query_broad_floor_min = max(0.0, min(1.0, float(query_broad_floor_min)))
        if self.query_broad_floor_min > self.query_strict_floor_min:
            self.query_broad_floor_min = self.query_strict_floor_min
        self.time_parse_strategy = str(time_parse_strategy or "local_first").strip().lower()
        self.validate_file_exists = bool(validate_file_exists)
        self.query_cache_enabled = bool(query_cache_enabled)
        self.query_cache_size = max(1, int(query_cache_size))
        self.embedding_cache_enabled = bool(embedding_cache_enabled)
        self.embedding_cache_size = max(1, int(embedding_cache_size))
        self.default_search_mode = self._normalize_search_mode(default_search_mode)
        self.index_loaded = False
        self.index_path = vector_store.index_path
        self.metadata_path = vector_store.metadata_path
        self.metric = getattr(vector_store, "metric", "cosine")
        self._metadata_by_path: Dict[str, Dict[str, Any]] = {}
        self._last_search_debug: Dict[str, Any] = self._empty_search_debug()
        self._last_round_quality: Dict[str, Any] = {}
        self._query_cache: Dict[tuple[str, str], Dict[str, Any]] = {}
        self._embedding_cache: Dict[str, List[float]] = {}
        self._refresh_metadata_cache()

    @staticmethod
    def _empty_search_debug() -> Dict[str, Any]:
        return {
            "mode": "text",
            "search_mode": "balanced",
            "base_intent": {},
            "expansion_triggered": False,
            "expansion_reason": "",
            "alternatives": [],
            "reflection_triggered": False,
            "reflection_reason": "",
            "reflection": {},
            "rounds": [],
            "timing": {},
        }

    @staticmethod
    def _normalize_search_mode(search_mode: Any) -> str:
        normalized = str(search_mode or "balanced").strip().lower()
        if normalized in {"fast", "balanced", "high_recall"}:
            return normalized
        return "balanced"

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
                "retrieval_mode": str(intent.get("retrieval_mode") or "hybrid"),
                "media_terms": list(intent.get("media_terms") or []),
                "identity_terms": list(intent.get("identity_terms") or []),
                "strict_identity_filter": bool(intent.get("strict_identity_filter", False)),
                "intent_mode": str(intent.get("intent_mode") or "open"),
                "intent_contract": dict(intent.get("intent_contract") or {}),
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

    @staticmethod
    def _record_timing(debug: Dict[str, Any], key: str, started_at: float) -> None:
        timing = debug.setdefault("timing", {})
        timing[key] = round((time.perf_counter() - started_at) * 1000, 3)

    def _cache_get(self, cache: Dict[Any, Any], key: Any) -> Any:
        value = cache.get(key)
        if value is None:
            return None
        cache.pop(key, None)
        cache[key] = value
        return value

    @staticmethod
    def _cache_put(cache: Dict[Any, Any], key: Any, value: Any, capacity: int) -> None:
        cache.pop(key, None)
        cache[key] = value
        while len(cache) > capacity:
            cache.pop(next(iter(cache)))

    def _format_query(self, query: str) -> Dict[str, Any]:
        default = {
            "search_text": query,
            "retrieval_mode": "hybrid",
            "media_terms": [],
            "identity_terms": [],
            "strict_identity_filter": False,
            "intent_mode": "open",
            "intent_contract": {},
            "time_hint": None,
            "season": None,
            "time_period": None,
            "original_query": query,
        }
        if not self.query_formatter or not self.query_formatter.is_enabled():
            return default
        cache_key = ("format_query", query)
        if self.query_cache_enabled:
            cached = self._cache_get(self._query_cache, cache_key)
            if cached is not None:
                return dict(cached)
        result = self.query_formatter.format_query(query)
        if self.query_cache_enabled:
            self._cache_put(self._query_cache, cache_key, dict(result), self.query_cache_size)
        return result

    def _generate_embedding(self, embedding_query: str) -> List[float]:
        normalized = str(embedding_query or "").strip()
        if not normalized:
            return self.embedding_service.generate_embedding(embedding_query)
        if self.embedding_cache_enabled:
            cached = self._cache_get(self._embedding_cache, normalized)
            if cached is not None:
                return list(cached)
        embedding = self.embedding_service.generate_embedding(embedding_query)
        if self.embedding_cache_enabled:
            self._cache_put(self._embedding_cache, normalized, list(embedding), self.embedding_cache_size)
        return embedding

    def _should_validate_path(self, normalized_path: str) -> bool:
        return bool(self.validate_file_exists and normalized_path)

    def _get_last_round_quality(self) -> Dict[str, Any]:
        return dict(self._last_round_quality)

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

    def _fill_results_to_top_k(
        self,
        primary_results: List[Dict[str, Any]],
        fallback_results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        filled: List[Dict[str, Any]] = []
        seen: set[str] = set()

        for source in (primary_results, fallback_results):
            for item in source:
                photo_path = item.get("photo_path")
                key = self._path_key(photo_path)
                if not key or key in seen:
                    continue
                filled.append(item)
                seen.add(key)
                if len(filled) >= top_k:
                    return filled
        return filled

    @staticmethod
    def _sort_results_for_merge(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            results,
            key=lambda item: (
                int(item.get("_confidence_bucket", 1)),
                float(item.get("score", 0.0)),
                -int(item.get("_relaxation_level", 0)),
            ),
            reverse=True,
        )

    @staticmethod
    def _sanitize_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sanitized: List[Dict[str, Any]] = []
        for item in results:
            clean = dict(item)
            clean.pop("metadata", None)
            for key in list(clean.keys()):
                if key.startswith("_"):
                    clean.pop(key, None)
            sanitized.append(clean)
        return sanitized

    @staticmethod
    def _intent_signature(intent: Dict[str, Any]) -> tuple[Any, ...]:
        return (
            str(intent.get("retrieval_mode") or "hybrid").strip().lower(),
            str(intent.get("search_text") or "").strip().lower(),
            tuple(
                sorted(
                    str(item).strip().lower()
                    for item in (intent.get("media_terms") or [])
                    if str(item).strip()
                )
            ),
            tuple(
                sorted(
                    str(item).strip().lower()
                    for item in (intent.get("identity_terms") or [])
                    if str(item).strip()
                )
            ),
            bool(intent.get("strict_identity_filter", False)),
        )

    def _results_signature(self, results: List[Dict[str, Any]]) -> tuple[tuple[str, float], ...]:
        signature: List[tuple[str, float]] = []
        for item in results:
            signature.append(
                (
                    self._path_key(item.get("photo_path", "")),
                    round(float(item.get("score", 0.0)), 6),
                )
            )
        return tuple(signature)

    def _should_continue_multi_round_search(
        self,
        results: List[Dict[str, Any]],
        top_k: int,
    ) -> bool:
        return self._should_expand_to_fill_results(results, top_k) or self._should_expand_results(results, top_k)

    def _max_relaxation_rounds_until_floor(self, start_level: int = 1) -> int:
        level = max(0, int(start_level))
        rounds = 1
        while self._get_round_score_floors(level + 1) != self._get_round_score_floors(level):
            rounds += 1
            level += 1
        return rounds

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
        normalized_search_text = search_text.strip()
        normalized_media_terms = [term.strip() for term in media_terms if term and term.strip()]
        normalized_identity_terms = [term.strip() for term in identity_terms if term and term.strip()]

        if normalized_search_text:
            parts.append(normalized_search_text)
        if normalized_media_terms:
            parts.append(" ".join(normalized_media_terms))

        # 人物名/公众人物名称容易把一阶段召回拉向 OCR、海报或截图。
        # 当 query 已具备视觉或媒介语义时，embedding 优先依赖这些可见语义；
        # identity_terms 仍保留给 metadata boost 与后续 rerank 使用。
        if normalized_identity_terms and not parts:
            parts.append(" ".join(normalized_identity_terms))
        query_text = " ".join(parts).strip()
        return query_text or original_query.strip()

    @staticmethod
    def _intent_contract_is_satisfied(
        base_intent: Dict[str, Any],
        candidate_intent: Dict[str, Any],
    ) -> bool:
        base_mode = str(base_intent.get("intent_mode") or "open").strip().lower()
        if base_mode != "strict":
            return bool(candidate_intent.get("contract_satisfied", True))

        if candidate_intent.get("contract_satisfied") is False:
            return False
        return True

    @staticmethod
    def _compute_metadata_boost(
        metadata: Dict[str, Any],
        media_terms: List[str],
        identity_terms: List[str],
    ) -> float:
        boost = 1.0
        metadata_media = {str(value).strip().lower() for value in (metadata.get("media_types") or []) if str(value).strip()}
        metadata_identities = {str(value).strip().lower() for value in (metadata.get("identity_names") or []) if str(value).strip()}
        query_media = {str(value).strip().lower() for value in media_terms if str(value).strip()}
        query_identities = {str(value).strip().lower() for value in identity_terms if str(value).strip()}
        if query_media and metadata_media.intersection(query_media):
            boost += 0.18
        if query_identities and metadata_identities.intersection(query_identities):
            boost += 0.12
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

    @staticmethod
    def _candidate_matches_media_terms(
        metadata: Dict[str, Any],
        media_terms: List[str],
    ) -> bool:
        if not media_terms:
            return True

        normalized_terms = [
            term.strip().lower()
            for term in media_terms
            if term and term.strip()
        ]
        normalized_media = [
            str(value).strip().lower()
            for value in (metadata.get("media_types") or [])
            if str(value).strip()
        ]
        if not normalized_terms:
            return True
        if not normalized_media:
            return False

        for term in normalized_terms:
            for candidate in normalized_media:
                if term == candidate or term in candidate or candidate in term:
                    return True
        return False

    def _split_identity_matches(
        self,
        results: List[Dict[str, Any]],
        identity_terms: List[str],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        matched: List[Dict[str, Any]] = []
        unmatched: List[Dict[str, Any]] = []
        for item in results:
            metadata = item.get("metadata", {})
            if self._candidate_matches_identity_terms(metadata, identity_terms):
                matched.append(item)
            else:
                unmatched.append(item)
        return matched, unmatched

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
            return max(scores[-1] * 0.9, self.query_dynamic_threshold_floor)

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
        return round(max(threshold, self.query_dynamic_threshold_floor), 6)

    @staticmethod
    def _should_expand_results(
        results: List[Dict[str, Any]],
        top_k: int,
        round_quality: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not results:
            return True
        top_score = float(results[0].get("score", 0.0))
        if top_score < 0.55:
            return True
        if round_quality:
            if int(round_quality.get("fallback_used_count", 0)) > 0:
                return True
            if int(round_quality.get("reliable_count", len(results))) < len(results):
                return True
        elif any(float(item.get("score", 0.0)) < MIN_RESULT_SCORE for item in results):
            return True
        if len(results) < min(top_k, 3) and top_score < 0.72:
            return True
        return False

    @staticmethod
    def _should_expand_to_fill_results(results: List[Dict[str, Any]], top_k: int) -> bool:
        target = max(1, int(top_k))
        return len(results) < target

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

    def _calculate_candidate_k(
        self,
        normalized_top_k: int,
        has_time_filter: bool,
        relaxation_level: int = 0,
    ) -> int:
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

        if relaxation_level > 0:
            expansion_floor = normalized_top_k * (base_multiplier + relaxation_level)
            candidate_k = max(candidate_k, expansion_floor)
            candidate_k = ceil(candidate_k * (1 + min(relaxation_level, 3) * 0.35))

        # 不超过实际数据量
        return min(candidate_k, total_items)

    def _get_round_score_floors(self, relaxation_level: int) -> tuple[float, float]:
        normalized_level = max(0, int(relaxation_level))
        strict_floor = max(self.query_strict_floor_min, MIN_RESULT_SCORE - 0.08 * normalized_level)
        broad_floor = max(self.query_broad_floor_min, strict_floor - 0.12)
        return round(strict_floor, 6), round(broad_floor, 6)

    def _assign_confidence_bucket(
        self,
        *,
        item: Dict[str, Any],
        strict_threshold: float,
        broad_threshold: float,
        media_terms: List[str],
        identity_terms: List[str],
        strict_identity_filter: bool,
    ) -> int:
        score = float(item.get("score", 0.0))
        if score >= strict_threshold:
            bucket = 3
        elif score >= broad_threshold:
            bucket = 2
        else:
            bucket = 1

        metadata = item.get("metadata") or {}
        if media_terms and not self._candidate_matches_media_terms(metadata, media_terms):
            bucket = max(1, bucket - 1)

        if identity_terms and not self._candidate_matches_identity_terms(metadata, identity_terms):
            bucket = max(1, bucket - (1 if strict_identity_filter else 0))

        return bucket

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
        - 向量检索基于 embedding_text 对应的视觉语义匹配
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

        # 1. 向量检索（基于 embedding_text 的语义匹配）
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
            if self._should_validate_path(normalized_path) and not os.path.exists(normalized_path):
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
            if not photo_path or not normalized_path:
                continue
            if self._should_validate_path(normalized_path) and not os.path.exists(normalized_path):
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
        relaxation_level: int = 0,
        debug: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        embedding_started_at = time.perf_counter()
        query_embedding = self._generate_embedding(embedding_query)
        if debug is not None and "embedding_ms" not in debug.get("timing", {}):
            self._record_timing(debug, "embedding_ms", embedding_started_at)
        candidate_k = self._calculate_candidate_k(
            normalized_top_k,
            has_filter,
            relaxation_level=relaxation_level,
        )

        recall_started_at = time.perf_counter()
        if self.keyword_store is not None:
            combined_results = self._hybrid_search(
                query,
                query_embedding,
                candidate_k,
                filters=constraints,
                allow_keyword_only_results=True,
                media_terms=media_terms,
                identity_terms=identity_terms,
                strict_identity_filter=strict_identity_filter,
            )
        else:
            raw_results = self.vector_store.search(query_embedding, candidate_k)
            combined_results = self._vector_results_to_combined(raw_results)
        if debug is not None:
            timing_key = "hybrid_search_ms" if self.keyword_store is not None else "vector_search_ms"
            if timing_key not in debug.get("timing", {}):
                self._record_timing(debug, timing_key, recall_started_at)

        finalize_started_at = time.perf_counter()
        results = self._finalize_results(
            combined_results=combined_results,
            normalized_top_k=normalized_top_k,
            has_filter=has_filter,
            constraints=constraints,
            search_text=str(intent.get("search_text") or ""),
            media_terms=media_terms,
            identity_terms=identity_terms,
            strict_identity_filter=strict_identity_filter,
            relaxation_level=relaxation_level,
            strip_internal=False,
        )
        if debug is not None and "merge_ms" not in debug.get("timing", {}):
            self._record_timing(debug, "merge_ms", finalize_started_at)
        return results

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
        relaxation_level: int = 2,
        seen_intent_signatures: Optional[set[tuple[Any, ...]]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.query_formatter or not self.query_formatter.is_enabled():
            return current_results
        needs_reflection_for_quality = self._should_expand_results(current_results, normalized_top_k)
        needs_reflection_for_count = self._should_expand_to_fill_results(current_results, normalized_top_k)
        if not needs_reflection_for_quality and not needs_reflection_for_count:
            return current_results

        reflection = self.query_formatter.reflect_on_weak_results(
            user_query=query,
            base_intent=base_intent,
            weak_results=current_results,
        )
        if not reflection:
            return current_results
        if not self._intent_contract_is_satisfied(base_intent, reflection):
            return current_results
        reflection_signature = self._intent_signature(reflection)
        if seen_intent_signatures is not None:
            if reflection_signature in seen_intent_signatures:
                return current_results
            seen_intent_signatures.add(reflection_signature)

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
            relaxation_level=relaxation_level,
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

        merged_results = [dict(item) for item in reflected_results]
        merged_results.extend(dict(item) for item in current_results)
        merged_results = self._deduplicate_results(merged_results)
        merged_results = self._sort_results_for_merge(merged_results)
        final_results = self._fill_results_to_top_k(
            merged_results,
            current_results,
            normalized_top_k,
        )
        for rank, item in enumerate(final_results, start=1):
            item["rank"] = rank
        return final_results

    def _continue_reflection_rounds(
        self,
        *,
        query: str,
        base_intent: Dict[str, Any],
        current_results: List[Dict[str, Any]],
        normalized_top_k: int,
        constraints: Dict[str, Any],
        has_filter: bool,
        debug: Dict[str, Any],
        start_relaxation_level: int = 2,
    ) -> List[Dict[str, Any]]:
        if not self.query_formatter or not self.query_formatter.is_enabled():
            return current_results
        if not self.query_reflection_enabled:
            return current_results
        if self.query_max_reflection_rounds < 0:
            return current_results

        reflection_round = max(2, int(start_relaxation_level))
        results = current_results
        seen_intent_signatures: set[tuple[Any, ...]] = set()
        reflection_attempts = 0
        max_reflection_rounds = self.query_max_reflection_rounds
        if max_reflection_rounds == 0:
            max_reflection_rounds = self._max_relaxation_rounds_until_floor(reflection_round)

        while (
            reflection_attempts < max_reflection_rounds
            and self._should_continue_multi_round_search(results, normalized_top_k)
        ):
            before_signature = self._results_signature(results)
            updated_results = self._maybe_reflect_query_results(
                query=query,
                base_intent=base_intent,
                current_results=results,
                normalized_top_k=normalized_top_k,
                constraints=constraints,
                has_filter=has_filter,
                debug=debug,
                relaxation_level=reflection_round,
                seen_intent_signatures=seen_intent_signatures,
            )
            after_signature = self._results_signature(updated_results)
            if after_signature == before_signature:
                break
            results = updated_results
            reflection_round += 1
            reflection_attempts += 1

        return results

    def _maybe_expand_query_results(
        self,
        *,
        query: str,
        base_intent: Dict[str, Any],
        base_results: List[Dict[str, Any]],
        base_round_quality: Optional[Dict[str, Any]],
        normalized_top_k: int,
        constraints: Dict[str, Any],
        has_filter: bool,
        debug: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not self.query_formatter or not self.query_formatter.is_enabled():
            return base_results
        if not self.query_expansion_enabled:
            return base_results
        max_expansion_rounds = self.query_expansion_max_alternatives
        if max_expansion_rounds == 0:
            max_expansion_rounds = self._max_relaxation_rounds_until_floor(1)
        if max_expansion_rounds < 0:
            return base_results
        should_expand_for_quality = self._should_expand_results(
            base_results,
            normalized_top_k,
            round_quality=base_round_quality,
        )
        should_expand_for_count = self._should_expand_to_fill_results(base_results, normalized_top_k)
        if not should_expand_for_quality and not should_expand_for_count:
            return base_results

        alternatives = self.query_formatter.expand_query_intents(
            user_query=query,
            base_intent=base_intent,
            max_alternatives=max_expansion_rounds,
        )
        merged: List[Dict[str, Any]] = [dict(item) for item in base_results]
        best_results: List[Dict[str, Any]] = base_results
        final_results = base_results
        if alternatives:
            debug["expansion_triggered"] = True
            for alt_index, alt in enumerate(alternatives[:max_expansion_rounds], start=1):
                if not self._intent_contract_is_satisfied(base_intent, alt):
                    continue
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
                    relaxation_level=alt_index,
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
            merged = self._sort_results_for_merge(merged)
            final_results = self._fill_results_to_top_k(
                merged,
                base_results,
                normalized_top_k,
            )
            for rank, item in enumerate(final_results, start=1):
                item["rank"] = rank

        expansion_reason = ""
        if debug["alternatives"]:
            if should_expand_for_quality and should_expand_for_count:
                expansion_reason = "第一轮结果偏弱且数量不足，尝试保守扩写查询意图"
            elif should_expand_for_quality:
                expansion_reason = "第一轮结果偏弱，尝试保守扩写查询意图"
            else:
                expansion_reason = "第一轮结果数量不足，尝试保守扩写查询意图"
        debug["expansion_reason"] = expansion_reason

        return self._continue_reflection_rounds(
            query=query,
            base_intent=base_intent,
            current_results=final_results,
            normalized_top_k=normalized_top_k,
            constraints=constraints,
            has_filter=has_filter,
            debug=debug,
            start_relaxation_level=max(2, len(debug["alternatives"]) + 1),
        )

    def _finalize_results(
        self,
        combined_results: List[Dict[str, Any]],
        normalized_top_k: int,
        has_filter: bool,
        constraints: Dict[str, Any],
        search_text: str = "",
        media_terms: Optional[List[str]] = None,
        identity_terms: Optional[List[str]] = None,
        strict_identity_filter: bool = False,
        relaxation_level: int = 0,
        strip_internal: bool = True,
    ) -> List[Dict[str, Any]]:
        filtered_results = []
        media_terms = media_terms or []
        identity_terms = identity_terms or []
        for item in combined_results:
            if self.keyword_store is None and has_filter:
                meta = item.get("metadata", {})
                if not self._check_time_match_v2(meta, constraints):
                    continue
            filtered_results.append(dict(item))

        filtered_results = self._deduplicate_results(filtered_results)
        fallback_results = filtered_results

        has_visual_grounding = bool(str(search_text or "").strip()) or bool(media_terms)
        should_promote_labeled_identity = strict_identity_filter and identity_terms and not has_visual_grounding

        if should_promote_labeled_identity:
            matched_results, unmatched_results = self._split_identity_matches(filtered_results, identity_terms)
            if matched_results:
                filtered_results = matched_results + unmatched_results
                fallback_results = filtered_results

        strict_floor, broad_floor = self._get_round_score_floors(relaxation_level)
        scores = [item["score"] for item in filtered_results]
        if scores:
            dynamic_threshold = self._calculate_dynamic_threshold(scores, normalized_top_k)
            strict_threshold = max(dynamic_threshold, strict_floor)
            broad_threshold = min(
                strict_threshold - 0.05,
                max(broad_floor, strict_threshold * 0.84),
            )
            broad_threshold = round(max(broad_floor, broad_threshold), 6)
        else:
            strict_threshold = strict_floor
            broad_threshold = broad_floor

        reliable_results: List[Dict[str, Any]] = []
        generalized_results: List[Dict[str, Any]] = []
        for item in filtered_results:
            bucket = self._assign_confidence_bucket(
                item=item,
                strict_threshold=strict_threshold,
                broad_threshold=broad_threshold,
                media_terms=media_terms,
                identity_terms=identity_terms,
                strict_identity_filter=strict_identity_filter,
            )
            item["_confidence_bucket"] = bucket
            item["_relaxation_level"] = max(0, int(relaxation_level))
            if bucket >= 3:
                reliable_results.append(item)
            elif bucket >= 2:
                generalized_results.append(item)

        prioritized_results = reliable_results + generalized_results

        final_results = self._fill_results_to_top_k(
            prioritized_results,
            fallback_results,
            normalized_top_k,
        )
        prioritized_keys = {
            self._path_key(item.get("photo_path", ""))
            for item in prioritized_results
            if item.get("photo_path")
        }
        reliable_keys = {
            self._path_key(item.get("photo_path", ""))
            for item in reliable_results
            if item.get("photo_path")
        }
        fallback_used_count = 0
        for item in final_results:
            key = self._path_key(item.get("photo_path", ""))
            if key and key not in prioritized_keys:
                fallback_used_count += 1

        self._last_round_quality = {
            "raw_count": len(filtered_results),
            "returned_count": len(final_results),
            "reliable_count": len(reliable_results),
            "generalized_count": len(prioritized_results),
            "fallback_used_count": fallback_used_count,
            "strict_threshold": round(strict_threshold, 6),
            "broad_threshold": round(broad_threshold, 6),
            "relaxation_level": max(0, int(relaxation_level)),
            "top_score": round(float(filtered_results[0].get("score", 0.0)), 6) if filtered_results else 0.0,
        }
        for rank, item in enumerate(final_results, start=1):
            item["rank"] = rank
        if strip_internal:
            return self._sanitize_results(final_results)
        return final_results

    def search(self, query: str, top_k: int = 10, search_mode: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        解析查询、执行混合检索、时间过滤并返回排序结果。

        改进：
        - embedding 基于更偏视觉语义的组合查询文本
        - EXIF 条件（时间、季节、时段）通过 Elasticsearch 精确过滤
        - 支持更细粒度的时段查询（7档细分）
        """
        if not self.validate_query(query):
            raise ValueError("查询内容不合法，请输入1-500字符的描述")

        if not self.index_loaded and not self.load_index():
            raise ValueError("索引未加载，请先初始化索引")

        search_mode = self._normalize_search_mode(search_mode or self.default_search_mode)
        normalized_top_k = max(1, min(int(top_k), 50))
        debug = self._empty_search_debug()
        debug["mode"] = "text"
        debug["search_mode"] = search_mode

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
        retrieval_mode = "hybrid"
        time_hints = {}

        format_result: Dict[str, Any] = {
            "intent_mode": "open",
            "intent_contract": {},
        }
        if query_formatter_enabled:
            formatter_started_at = time.perf_counter()
            format_result = self._format_query(query)
            self._record_timing(debug, "query_formatter_ms", formatter_started_at)
            formatted_query = (format_result.get("search_text") or "").strip()
            media_terms = list(format_result.get("media_terms") or [])
            identity_terms = list(format_result.get("identity_terms") or [])
            strict_identity_filter = bool(format_result.get("strict_identity_filter", False))
            time_hints = {
                "time_hint": format_result.get("time_hint"),
                "season": format_result.get("season"),
                "time_period": format_result.get("time_period"),
            }
            retrieval_mode = str(format_result.get("retrieval_mode") or "").strip().lower()
            if retrieval_mode not in {"hybrid", "filter_only"}:
                retrieval_mode = "filter_only" if (not formatted_query and any(time_hints.values())) else "hybrid"

        # 2. 时间解析（返回结构化过滤条件）
        constraints: Dict[str, Any] = {
            "start_date": None, "end_date": None,
            "year": None, "month": None, "day": None,
            "season": None, "time_period": None,
            "precision": "none",
        }
        cleaned_query = formatted_query
        explicit_time_filter_requested = self.time_parser.detect_time_terms(
            query,
            strategy=self.time_parse_strategy,
        )

        if explicit_time_filter_requested:
            time_parse_started_at = time.perf_counter()
            constraints = self._extract_time_constraints(query)
            self._record_timing(debug, "time_parse_ms", time_parse_started_at)

            # 只有用户查询中真的包含时间语义时，才把 QueryFormatter 的 season/time_period
            # 作为 EXIF 过滤条件；否则它们只是视觉语义，不应把普通雪景/夜景查询误伤为 0 结果。
            if time_hints.get("season") and not constraints.get("season"):
                constraints["season"] = time_hints["season"]
            if time_hints.get("time_period") and not constraints.get("time_period"):
                constraints["time_period"] = time_hints["time_period"]

        if retrieval_mode == "filter_only" and not explicit_time_filter_requested:
            retrieval_mode = "hybrid"

        has_filter = bool(
            constraints.get("start_date")
            or constraints.get("end_date")
            or constraints.get("year")
            or constraints.get("month")
            or constraints.get("day")
            or constraints.get("season")
            or constraints.get("time_period")
        )

        # 3. 只有“纯 EXIF/时间过滤查询”才降级为纯关键字检索。
        # 其判定依据是：LLM 没有抽出任何视觉语义，且确实存在过滤条件。
        is_filter_only_query = query_formatter_enabled and retrieval_mode == "filter_only" and has_filter
        if is_filter_only_query:
            # 纯过滤查询：跳过向量检索，直接使用 ES 关键字与过滤检索
            print(f"[DEBUG] 纯过滤查询模式，constraints: {constraints}")
            filter_only_intent = {
                "search_text": cleaned_query,
                "retrieval_mode": retrieval_mode,
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
            "retrieval_mode": retrieval_mode,
            "media_terms": list(media_terms),
            "identity_terms": list(identity_terms),
            "strict_identity_filter": strict_identity_filter,
            "intent_mode": str(format_result.get("intent_mode") or "open") if query_formatter_enabled else "open",
            "intent_contract": dict(format_result.get("intent_contract") or {}) if query_formatter_enabled else {},
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
            relaxation_level=0,
            debug=debug,
        )
        base_round_quality = self._get_last_round_quality()
        debug["rounds"].append(
            self._round_summary(
                round_name="base",
                intent=base_intent,
                results=first_round_results,
            )
        )
        final_results = first_round_results
        if search_mode == "high_recall" and self.query_multi_round_enabled:
            final_results = self._maybe_expand_query_results(
                query=query,
                base_intent=base_intent,
                base_results=first_round_results,
                base_round_quality=base_round_quality,
                normalized_top_k=normalized_top_k,
                constraints=constraints,
                has_filter=has_filter,
                debug=debug,
            )
        final_results = self._sanitize_results(final_results)
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
