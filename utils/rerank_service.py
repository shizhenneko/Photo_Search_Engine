from __future__ import annotations

import base64
import json
import re
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from utils.image_parser import resize_and_optimize_image
from utils.llm_compat import (
    build_image_url_content,
    create_chat_completion,
    extract_response_text,
    normalize_openai_base_url,
    requires_api_key,
    resolve_api_key,
)
from utils.path_utils import normalize_local_path


class VisualRerankService:
    """使用视觉模型对候选图片进行二次精排。"""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str,
        reasoning_effort: str = "medium",
        timeout: int = 60,
        max_retries: int = 3,
        image_max_size: int = 512,
        image_quality: int = 75,
        image_format: str = "WEBP",
        max_images: int = 10,
        client: Optional[OpenAI] = None,
    ) -> None:
        if requires_api_key(base_url) and not api_key:
            raise ValueError("SU8_API_KEY 未设置")
        if not model_name:
            raise ValueError("VISUAL_RERANK_MODEL 未设置")
        resolved_api_key = resolve_api_key(api_key, base_url)
        self.api_key = resolved_api_key
        self.model_name = model_name
        self.base_url = normalize_openai_base_url(base_url)
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.image_max_size = max(256, min(2048, image_max_size))
        self.image_quality = max(1, min(100, image_quality))
        self.image_format = image_format.upper() if image_format.upper() in {"JPEG", "PNG", "WEBP"} else "WEBP"
        self.max_images = max(1, min(20, max_images))
        self.client = client or OpenAI(api_key=resolved_api_key, base_url=self.base_url)

    def _get_image_base64(self, image_path: str) -> str:
        image_bytes = resize_and_optimize_image(
            image_path,
            max_size=self.image_max_size,
            quality=self.image_quality,
            format=self.image_format,
        )
        mime_type = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
        }.get(self.image_format, "image/webp")
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def _build_prompt(self, query: str, num_images: int) -> str:
        return f"""用户在检索与查询最相关的照片："{query}"

你将看到 {num_images} 张候选图片，编号从 1 到 {num_images}。
请按与查询的视觉相关性从高到低排序。
优先判断图片是否直接呈现了用户要找的主体、场景、动作、构图或载体组合。
如果一张图只是通过文字、界面、转述内容、嵌入式屏幕或二次载体间接相关，而另一张图能更直接满足查询目标，应把后者排在前面。
不要因为图片里出现相关文字或名称，就把间接相关的图片排到真正更符合目标的图片前面。

只返回 JSON：
{{"ranking": [1, 3, 2]}}"""

    def _build_reference_prompt(self, num_images: int) -> str:
        return f"""第一张图片是查询图。

后面会依次给出 {num_images} 张候选图片，编号从 1 到 {num_images}。
请按与查询图在主体、场景、构图和视觉风格上的相似度从高到低排序。

只返回 JSON：
{{"ranking": [2, 1, 3]}}"""

    @staticmethod
    def _build_rank_score_map(ranking: List[int], num_images: int) -> Dict[int, float]:
        if num_images <= 0:
            return {}
        score_map: Dict[int, float] = {}
        max_score = float(num_images)
        for rank, candidate_index in enumerate(ranking, start=1):
            if candidate_index < 0 or candidate_index >= num_images or candidate_index in score_map:
                continue
            score_map[candidate_index] = (max_score - rank + 1.0) / max_score
        return score_map

    def _rerank_chunk(
        self,
        *,
        content: List[Dict[str, Any]],
        candidates_to_process: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        num_images = len(candidates_to_process)
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self._create_completion(content)
                ranking = self._parse_ranking_response(
                    extract_response_text(response),
                    num_images,
                )
                score_map = self._build_rank_score_map(ranking, num_images)
                reranked: List[Dict[str, Any]] = []
                for index, candidate in enumerate(candidates_to_process):
                    item = dict(candidate)
                    item["visual_rerank_score"] = round(score_map.get(index, 0.0), 6)
                    reranked.append(item)
                reranked.sort(
                    key=lambda item: (
                        float(item.get("visual_rerank_score", 0.0)),
                        float(item.get("score", 0.0)),
                    ),
                    reverse=True,
                )
                return reranked
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(1)

        if last_error is not None:
            raise ValueError(f"视觉 rerank 失败: {last_error}") from last_error
        raise ValueError("视觉 rerank 失败")

    def _rerank_in_batches(
        self,
        *,
        candidates: List[Dict[str, Any]],
        build_content,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        if len(candidates) <= self.max_images:
            content = build_content(candidates)
            return self._rerank_chunk(content=content, candidates_to_process=candidates)

        chunk_results: List[Dict[str, Any]] = []
        for start in range(0, len(candidates), self.max_images):
            chunk = candidates[start:start + self.max_images]
            if len(chunk) <= 1:
                reranked_chunk = [dict(chunk[0])] if chunk else []
                for item in reranked_chunk:
                    item["visual_rerank_score"] = round(float(item.get("score", 0.0)), 6)
            else:
                content = build_content(chunk)
                reranked_chunk = self._rerank_chunk(content=content, candidates_to_process=chunk)
            for item in reranked_chunk:
                item["visual_rerank_batch"] = start // self.max_images + 1
            chunk_results.extend(reranked_chunk)

        if len(chunk_results) <= 1:
            return chunk_results

        merge_content = build_content(chunk_results)
        return self._rerank_chunk(content=merge_content, candidates_to_process=chunk_results)

    def _parse_ranking_response(self, response_text: str, num_images: int) -> List[int]:
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).rstrip("`").strip()
        try:
            data = json.loads(cleaned)
            ranking = data.get("ranking", [])
            indexes = [int(item) - 1 for item in ranking if 1 <= int(item) <= num_images]
            if indexes:
                return indexes
        except Exception:
            pass

        numbers = re.findall(r"\d+", cleaned)
        indexes = [int(item) - 1 for item in numbers if 1 <= int(item) <= num_images]
        if indexes:
            return indexes
        raise ValueError("无法解析视觉 rerank 响应")

    def _filter_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        for candidate in candidates:
            photo_path = candidate.get("photo_path")
            normalized_path = normalize_local_path(photo_path) if photo_path else ""
            if not normalized_path or not normalized_path.strip():
                continue
            if not normalized_path.startswith("/"):
                normalized_path = normalize_local_path(normalized_path)
            try:
                with open(normalized_path, "rb"):
                    pass
            except Exception:
                continue
            normalized_candidate = dict(candidate)
            normalized_candidate["photo_path"] = normalized_path
            filtered.append(normalized_candidate)
        return filtered

    @staticmethod
    def _merge_with_unprocessed_candidates(
        reranked_candidates: List[Dict[str, Any]],
        original_candidates: List[Dict[str, Any]],
        rerank_top_k: int,
    ) -> List[Dict[str, Any]]:
        if rerank_top_k <= 0:
            return []

        merged: List[Dict[str, Any]] = []
        seen_paths: set[str] = set()

        def append_item(item: Dict[str, Any]) -> None:
            photo_path = str(item.get("photo_path") or "")
            if photo_path and photo_path in seen_paths:
                return
            merged.append(dict(item))
            if photo_path:
                seen_paths.add(photo_path)

        for candidate in reranked_candidates:
            append_item(candidate)
            if len(merged) >= rerank_top_k:
                break

        if len(merged) < rerank_top_k:
            for candidate in original_candidates:
                append_item(candidate)
                if len(merged) >= rerank_top_k:
                    break

        for rank, item in enumerate(merged, start=1):
            item["rank"] = rank
        return merged

    def _create_completion(self, content: List[Dict[str, Any]]):
        message_payload = [{"role": "user", "content": content}]
        try:
            return create_chat_completion(
                self.client,
                model=self.model_name,
                messages=message_payload,
                timeout=self.timeout,
                reasoning_effort=self.reasoning_effort,
            )
        except Exception:
            # 某些 OpenAI 兼容网关不接受多模态 content list，
            # 需要退化为标准 JSON 字符串消息体。
            fallback_content = json.dumps(content, ensure_ascii=False)
            return create_chat_completion(
                self.client,
                model=self.model_name,
                messages=[{"role": "user", "content": fallback_content}],
                timeout=self.timeout,
                reasoning_effort=self.reasoning_effort,
            )

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        rerank_top_k: int,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        if not query or not query.strip():
            return candidates[:rerank_top_k]

        candidates_to_process = self._filter_candidates(candidates)
        num_images = len(candidates_to_process)
        if num_images <= 1:
            return candidates[:rerank_top_k]

        def build_content(chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            content: List[Dict[str, Any]] = [{"type": "text", "text": self._build_prompt(query, len(chunk))}]
            for index, candidate in enumerate(chunk, start=1):
                photo_path = candidate.get("photo_path")
                if not photo_path:
                    continue
                content.append({"type": "text", "text": f"候选图片 {index}"})
                content.append(build_image_url_content(self._get_image_base64(photo_path), self.base_url))
            return content

        reranked = self._rerank_in_batches(
            candidates=candidates_to_process,
            build_content=build_content,
        )
        return self._merge_with_unprocessed_candidates(reranked, candidates, rerank_top_k)

    def rerank_by_reference_image(
        self,
        reference_image_path: str,
        candidates: List[Dict[str, Any]],
        rerank_top_k: int,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        reference_image_path = normalize_local_path(reference_image_path)
        candidates_to_process = self._filter_candidates(candidates)
        num_images = len(candidates_to_process)
        if num_images <= 1:
            return candidates[:rerank_top_k]

        def build_content(chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            content: List[Dict[str, Any]] = [
                {"type": "text", "text": self._build_reference_prompt(len(chunk))},
                {"type": "text", "text": "查询图片"},
                build_image_url_content(self._get_image_base64(reference_image_path), self.base_url),
            ]
            for index, candidate in enumerate(chunk, start=1):
                photo_path = candidate.get("photo_path")
                if not photo_path:
                    continue
                content.append({"type": "text", "text": f"候选图片 {index}"})
                content.append(build_image_url_content(self._get_image_base64(photo_path), self.base_url))
            return content

        reranked = self._rerank_in_batches(
            candidates=candidates_to_process,
            build_content=build_content,
        )
        return self._merge_with_unprocessed_candidates(reranked, candidates, rerank_top_k)

    def is_enabled(self) -> bool:
        return bool(self.api_key and self.model_name)
