from __future__ import annotations

import time
from abc import ABC, abstractmethod
import json
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

from utils.llm_compat import (
    create_chat_completion,
    extract_response_text,
    is_ollama_base_url,
    normalize_openai_base_url,
    requires_api_key,
    resolve_api_key,
)


class EmbeddingService(ABC):
    """Embedding 服务抽象接口。"""

    dimension: Optional[int] = None

    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """生成单条文本向量。"""

    @abstractmethod
    def generate_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本向量。"""


class OpenAICompatibleEmbeddingService(EmbeddingService):
    """基于 OpenAI 兼容 embeddings 接口的文本向量服务。"""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str,
        timeout: int = 30,
        max_retries: int = 3,
        client: Optional[OpenAI] = None,
        dimension: Optional[int] = None,
    ) -> None:
        if requires_api_key(base_url) and not api_key:
            raise ValueError("EMBEDDING_API_KEY 未设置")
        resolved_api_key = resolve_api_key(api_key, base_url)
        self.api_key = resolved_api_key
        self.model_name = model_name
        self.base_url = normalize_openai_base_url(base_url)
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.client = client or OpenAI(api_key=resolved_api_key, base_url=self.base_url)
        self.dimension = dimension

    def generate_embedding(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("待向量化文本不能为空")

        for attempt in range(self.max_retries):
            try:
                request_payload: Dict[str, Any] = {
                    "model": self.model_name,
                    "input": text,
                    "timeout": self.timeout,
                }
                if self.dimension:
                    request_payload["dimensions"] = self.dimension
                response = self.client.embeddings.create(
                    **request_payload,
                )
                embedding = response.data[0].embedding
                if self.dimension is None:
                    self.dimension = len(embedding)
                return embedding
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise ValueError(f"向量生成失败: {exc}") from exc
                time.sleep(1)
        raise ValueError("向量生成失败")

    def generate_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        cleaned_texts = [text for text in texts if text and text.strip()]
        if not cleaned_texts:
            raise ValueError("待向量化文本不能为空")

        for attempt in range(self.max_retries):
            try:
                request_payload: Dict[str, Any] = {
                    "model": self.model_name,
                    "input": cleaned_texts,
                    "timeout": self.timeout,
                }
                if self.dimension:
                    request_payload["dimensions"] = self.dimension
                response = self.client.embeddings.create(
                    **request_payload,
                )
                embeddings = [item.embedding for item in response.data]
                if embeddings and self.dimension is None:
                    self.dimension = len(embeddings[0])
                return embeddings
            except Exception:
                if attempt == self.max_retries - 1:
                    break
                time.sleep(1)

        return [self.generate_embedding(text) for text in cleaned_texts]


class TumuerEmbeddingService(OpenAICompatibleEmbeddingService):
    """Tumuer Router 上的 embedding 服务。"""


class TextRerankService:
    """兼容专有 rerank API 与聊天模型排序的文本 rerank 服务。"""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str,
        timeout: int = 30,
        max_retries: int = 3,
        session: Optional[requests.Session] = None,
        client: Optional[OpenAI] = None,
        backend: str = "auto",
    ) -> None:
        if requires_api_key(base_url) and not api_key:
            raise ValueError("TEXT_RERANK_API_KEY 未设置")
        resolved_api_key = resolve_api_key(api_key, base_url)
        self.api_key = resolved_api_key
        self.model_name = model_name
        self.base_url = normalize_openai_base_url(base_url)
        self.http_base_url = (base_url or "").rstrip("/")
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.session = session or requests.Session()
        self.client = client or OpenAI(api_key=resolved_api_key, base_url=self.base_url)
        self.backend = (backend or "auto").strip().lower()

    def _resolve_backend(self) -> str:
        if self.backend in {"api", "chat"}:
            return self.backend
        if is_ollama_base_url(self.http_base_url):
            return "chat"
        return "api"

    @staticmethod
    def _build_documents(candidates: List[Dict[str, Any]]) -> List[str]:
        documents = []
        for item in candidates:
            documents.append(
                item.get("retrieval_text")
                or item.get("description")
                or item.get("match_summary", {}).get("ocr_excerpt")
                or item.get("photo_path")
                or ""
            )
        return documents

    def _rerank_with_api(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        documents = self._build_documents(candidates)

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": min(max(1, top_k), len(documents)),
            "return_documents": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = self.session.post(
            f"{self.http_base_url}/rerank",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results") or data.get("data") or []
        if not isinstance(results, list):
            raise ValueError("rerank 返回格式不正确")

        reranked: List[Dict[str, Any]] = []
        for rank, item in enumerate(results, start=1):
            index = item.get("index")
            if index is None or index < 0 or index >= len(candidates):
                continue
            candidate = dict(candidates[index])
            score = item.get("relevance_score")
            if score is not None:
                candidate["text_rerank_score"] = round(float(score), 6)
            candidate["rank"] = rank
            reranked.append(candidate)

        if reranked:
            return reranked[:top_k]
        raise ValueError("rerank 未返回有效结果")

    def _rerank_with_chat(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        documents = self._build_documents(candidates)
        prompt_documents = [
            {"index": index + 1, "text": document}
            for index, document in enumerate(documents)
        ]
        prompt = (
            "你是照片搜索结果的文本重排器。"
            "请根据 query 和候选文档内容，将最相关的候选按从高到低排序。"
            "只返回 JSON，格式固定为 {\"ranking\":[{\"index\":1,\"score\":0.98}]}。"
            "index 从 1 开始，score 为 0 到 1 之间的小数。"
            f"只返回前 {min(max(1, top_k), len(documents))} 个结果。\n"
            f"query: {query}\n"
            f"documents: {documents_to_json(prompt_documents)}"
        )
        response = create_chat_completion(
            self.client,
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            timeout=self.timeout,
            temperature=0,
            response_format={"type": "json_object"},
        )
        payload = json.loads(extract_response_text(response))
        ranking = payload.get("ranking") or []
        if not isinstance(ranking, list):
            raise ValueError("聊天 rerank 返回格式不正确")

        reranked: List[Dict[str, Any]] = []
        for rank, item in enumerate(ranking, start=1):
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            if index is None:
                continue
            candidate_index = int(index) - 1
            if candidate_index < 0 or candidate_index >= len(candidates):
                continue
            candidate = dict(candidates[candidate_index])
            score = item.get("score")
            if score is not None:
                candidate["text_rerank_score"] = round(float(score), 6)
            candidate["rank"] = rank
            reranked.append(candidate)

        if reranked:
            return reranked[:top_k]
        raise ValueError("聊天 rerank 未返回有效结果")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        if not query or not query.strip():
            return candidates[:top_k]

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                if self._resolve_backend() == "api":
                    return self._rerank_with_api(query, candidates, top_k)
                return self._rerank_with_chat(query, candidates, top_k)
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(1)

        if last_error is not None:
            raise ValueError(f"文本 rerank 失败: {last_error}") from last_error
        raise ValueError("文本 rerank 失败")

    def is_enabled(self) -> bool:
        return bool(self.api_key and self.model_name and self.base_url)


def documents_to_json(documents: List[Dict[str, Any]]) -> str:
    return json.dumps(documents, ensure_ascii=False)
