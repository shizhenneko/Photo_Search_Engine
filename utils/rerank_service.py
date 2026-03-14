from __future__ import annotations

import base64
import json
import re
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from utils.image_parser import resize_and_optimize_image
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
        if not api_key:
            raise ValueError("SU8_API_KEY 未设置")
        if not model_name:
            raise ValueError("VISUAL_RERANK_MODEL 未设置")
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.image_max_size = max(256, min(2048, image_max_size))
        self.image_quality = max(1, min(100, image_quality))
        self.image_format = image_format.upper() if image_format.upper() in {"JPEG", "PNG", "WEBP"} else "WEBP"
        self.max_images = max(1, min(20, max_images))
        self.client = client or OpenAI(api_key=api_key, base_url=base_url)

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

只返回 JSON：
{{"ranking": [1, 3, 2]}}"""

    def _build_reference_prompt(self, num_images: int) -> str:
        return f"""第一张图片是查询图。

后面会依次给出 {num_images} 张候选图片，编号从 1 到 {num_images}。
请按与查询图在主体、场景、构图和视觉风格上的相似度从高到低排序。

只返回 JSON：
{{"ranking": [2, 1, 3]}}"""

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

    def _create_completion(self, content: List[Dict[str, Any]]):
        message_payload = [{"role": "user", "content": content}]
        try:
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=message_payload,
                timeout=self.timeout,
                extra_body={"reasoning_effort": self.reasoning_effort},
            )
        except Exception:
            # 某些 OpenAI 兼容网关不接受多模态 content list，
            # 需要退化为标准 JSON 字符串消息体。
            fallback_content = json.dumps(content, ensure_ascii=False)
            try:
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": fallback_content}],
                    timeout=self.timeout,
                )
            except Exception:
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=message_payload,
                    timeout=self.timeout,
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

        candidates_to_process = self._filter_candidates(candidates)[: self.max_images]
        num_images = len(candidates_to_process)
        if num_images <= 1:
            return candidates[:rerank_top_k]

        content: List[Dict[str, Any]] = [{"type": "text", "text": self._build_prompt(query, num_images)}]
        for index, candidate in enumerate(candidates_to_process, start=1):
            photo_path = candidate.get("photo_path")
            if not photo_path:
                continue
            content.append({"type": "text", "text": f"候选图片 {index}"})
            content.append({"type": "image_url", "image_url": {"url": self._get_image_base64(photo_path)}})

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self._create_completion(content)
                ranking = self._parse_ranking_response(
                    response.choices[0].message.content or "",
                    num_images,
                )
                reranked = []
                used = set()
                for rank, candidate_index in enumerate(ranking, start=1):
                    if candidate_index in used or candidate_index >= len(candidates_to_process):
                        continue
                    used.add(candidate_index)
                    item = dict(candidates_to_process[candidate_index])
                    item["rank"] = rank
                    reranked.append(item)
                for item in candidates_to_process:
                    if len(reranked) >= rerank_top_k:
                        break
                    if item not in reranked:
                        reranked.append(dict(item))
                return reranked[:rerank_top_k]
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(1)

        if last_error is not None:
            raise ValueError(f"视觉 rerank 失败: {last_error}") from last_error
        raise ValueError("视觉 rerank 失败")

    def rerank_by_reference_image(
        self,
        reference_image_path: str,
        candidates: List[Dict[str, Any]],
        rerank_top_k: int,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        reference_image_path = normalize_local_path(reference_image_path)
        candidates_to_process = self._filter_candidates(candidates)[: self.max_images]
        num_images = len(candidates_to_process)
        if num_images <= 1:
            return candidates[:rerank_top_k]

        content: List[Dict[str, Any]] = [
            {"type": "text", "text": self._build_reference_prompt(num_images)},
            {"type": "text", "text": "查询图片"},
            {"type": "image_url", "image_url": {"url": self._get_image_base64(reference_image_path)}},
        ]
        for index, candidate in enumerate(candidates_to_process, start=1):
            photo_path = candidate.get("photo_path")
            if not photo_path:
                continue
            content.append({"type": "text", "text": f"候选图片 {index}"})
            content.append({"type": "image_url", "image_url": {"url": self._get_image_base64(photo_path)}})

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self._create_completion(content)
                ranking = self._parse_ranking_response(
                    response.choices[0].message.content or "",
                    num_images,
                )
                reranked = []
                used = set()
                for rank, candidate_index in enumerate(ranking, start=1):
                    if candidate_index in used or candidate_index >= len(candidates_to_process):
                        continue
                    used.add(candidate_index)
                    item = dict(candidates_to_process[candidate_index])
                    item["rank"] = rank
                    reranked.append(item)
                for item in candidates_to_process:
                    if len(reranked) >= rerank_top_k:
                        break
                    if item not in reranked:
                        reranked.append(dict(item))
                return reranked[:rerank_top_k]
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(1)

        if last_error is not None:
            raise ValueError(f"视觉 rerank 失败: {last_error}") from last_error
        raise ValueError("视觉 rerank 失败")

    def is_enabled(self) -> bool:
        return bool(self.api_key and self.model_name)
