from __future__ import annotations

from typing import Any, Dict, List, Optional


class FakeEmbeddingService:
    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension

    def generate_embedding(self, text: str) -> List[float]:
        seed = float(sum(ord(char) for char in (text or "")) % 13)
        return [seed + float(index) for index in range(self.dimension)]

    def generate_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.generate_embedding(text) for text in texts]


class FakeTimeParser:
    def has_time_terms(self, query: str) -> bool:
        return "去年" in query or "2024" in query

    def extract_time_constraints(self, query: str) -> Dict[str, Any]:
        if "去年" in query:
            return {"start_date": "2025-01-01", "end_date": "2025-12-31", "precision": "year"}
        return {"start_date": None, "end_date": None, "precision": "none"}


class FakeQueryFormatter:
    def __init__(self, mapping: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self.mapping = mapping or {}
        self.expansion_mapping: Dict[str, List[Dict[str, Any]]] = {}
        self.reflection_mapping: Dict[str, Dict[str, Any]] = {}

    def format_query(self, user_query: str) -> Dict[str, Any]:
        return self.mapping.get(
            user_query,
            {
                "search_text": user_query,
                "media_terms": [],
                "identity_terms": [],
                "strict_identity_filter": False,
                "time_hint": None,
                "season": None,
                "time_period": None,
                "original_query": user_query,
            },
        )

    def expand_query_intents(
        self,
        user_query: str,
        base_intent: Dict[str, Any],
        max_alternatives: int = 2,
    ) -> List[Dict[str, Any]]:
        alternatives = self.expansion_mapping.get(user_query, [])
        return alternatives[:max_alternatives]

    def reflect_on_weak_results(
        self,
        user_query: str,
        base_intent: Dict[str, Any],
        weak_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return self.reflection_mapping.get(user_query, {})

    def is_enabled(self) -> bool:
        return True


class FakeTextRerankService:
    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        reordered = list(reversed(candidates))
        for rank, item in enumerate(reordered, start=1):
            item["text_rerank_score"] = float(top_k - rank + 1)
        return reordered[:top_k]

    def is_enabled(self) -> bool:
        return True


class FakeVisualRerankService:
    def rerank(self, query: str, candidates: List[Dict[str, Any]], rerank_top_k: int) -> List[Dict[str, Any]]:
        return candidates[:rerank_top_k]

    def rerank_by_reference_image(
        self,
        reference_image_path: str,
        candidates: List[Dict[str, Any]],
        rerank_top_k: int,
    ) -> List[Dict[str, Any]]:
        reordered = sorted(candidates, key=lambda item: item.get("photo_path", ""), reverse=True)
        return reordered[:rerank_top_k]

    def is_enabled(self) -> bool:
        return True


class FakeStructuredVisionService:
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        return {
            "description": "测试图片",
            "outer_scene_summary": "一张测试图片",
            "inner_content_summary": "",
            "media_types": ["photo"],
            "tags": ["测试图片"],
            "ocr_text": "",
            "person_roles": [],
            "identity_candidates": [],
            "identity_names": [],
            "identity_evidence": [],
            "analysis_flags": {},
            "embedding_text": "photo 测试图片",
            "retrieval_text": "photo 测试图片",
        }

    def analyze_image_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        return [self.analyze_image(path) for path in image_paths]

    def generate_description(self, image_path: str) -> str:
        return self.analyze_image(image_path)["description"]

    def generate_description_batch(self, image_paths: List[str]) -> List[str]:
        return [self.generate_description(path) for path in image_paths]
