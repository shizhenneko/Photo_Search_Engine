from __future__ import annotations

import base64
import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from openai import OpenAI

from utils.image_parser import get_image_dimensions, resize_and_optimize_image
from utils.llm_compat import (
    build_image_url_content,
    create_chat_completion,
    extract_response_text,
    normalize_openai_base_url,
    requires_api_key,
    resolve_api_key,
)
from utils.structured_analysis import (
    get_enhanced_analysis_reason,
    normalize_analysis_payload,
)


class VisionLLMService(ABC):
    """视觉模型抽象接口。"""

    def get_last_analysis_metrics(self) -> Optional[Dict[str, Any]]:
        """返回最近一次 analyze_image 的诊断指标。"""
        return None

    @abstractmethod
    def generate_description(self, image_path: str) -> str:
        """生成图片中文描述。"""

    @abstractmethod
    def generate_description_batch(self, image_paths: List[str]) -> List[str]:
        """批量生成图片描述。"""

    @abstractmethod
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """生成结构化图片分析结果。"""

    @abstractmethod
    def analyze_image_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """批量生成结构化图片分析结果。"""


class SU8VisionLLMService(VisionLLMService):
    """通过 OpenAI 兼容接口调用视觉模型。"""

    EXPECTED_ANALYSIS_KEYS = (
        "description",
        "outer_scene_summary",
        "inner_content_summary",
        "media_types",
        "tags",
        "ocr_text",
        "person_roles",
        "identity_candidates",
        "analysis_flags",
    )

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str,
        reasoning_effort: str = "medium",
        timeout: int = 30,
        max_retries: int = 3,
        use_base64: bool = True,
        image_max_size: int = 1024,
        image_quality: int = 85,
        image_format: str = "WEBP",
        enhanced_reasoning_effort: str = "low",
        base_max_output_tokens: int = 700,
        enhanced_max_output_tokens: int = 420,
        repair_max_output_tokens: int = 420,
        client: Optional[OpenAI] = None,
    ) -> None:
        if requires_api_key(base_url) and not api_key:
            raise ValueError("SU8_API_KEY 未设置")
        resolved_api_key = resolve_api_key(api_key, base_url)
        self.api_key = resolved_api_key
        self.model_name = model_name
        self.base_url = normalize_openai_base_url(base_url)
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.use_base64 = use_base64
        self.image_max_size = max(256, min(4096, image_max_size))
        self.image_quality = max(1, min(100, image_quality))
        self.image_format = image_format.upper() if image_format.upper() in {"JPEG", "PNG", "WEBP"} else "WEBP"
        self.enhanced_reasoning_effort = enhanced_reasoning_effort
        self.base_max_output_tokens = max(128, int(base_max_output_tokens))
        self.enhanced_max_output_tokens = max(128, int(enhanced_max_output_tokens))
        self.repair_max_output_tokens = max(128, int(repair_max_output_tokens))
        self.client = client or OpenAI(api_key=resolved_api_key, base_url=self.base_url)
        self._last_analysis_metrics: Optional[Dict[str, Any]] = None
        self.enhanced_analysis_enabled = True

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

    def _build_description_prompt(self) -> str:
        return (
            "观察图片并只返回 JSON。字段固定为："
            "{\"description\":\"\",\"outer_scene_summary\":\"\",\"inner_content_summary\":\"\","
            "\"media_types\":[],\"tags\":[],\"ocr_text\":\"\",\"person_roles\":[],"
            "\"identity_candidates\":[],\"analysis_flags\":{}}。\n"
            "要求：description 用一句话简洁总结；outer_scene_summary 只写相机实际拍到的外层场景；"
            "inner_content_summary 只写被拍对象内部最有检索价值的内容。"
            "media_types 使用自由文本短语描述图片载体、媒介或内容类型，不使用固定词表。"
            "tags 用高价值短标签，最多 8 个，优先描述可直接看到的主体、场景、动作、构图或媒介特征；"
            "不要把 OCR 片段或名字机械重复成标签，除非它们在视觉上可确认且确实提升检索价值。"
            "如返回对象，格式为 {\"tag\":\"\",\"confidence\":0-1}。"
            "ocr_text 只保留最有检索价值的关键文字，尽量控制在 200 字内。"
            "如果涉及可命名主体，必须区分画面中真正出现的主体、被拍载体里出现的主体、以及仅被文字提及但画面里无法确认的主体。"
            "identity_candidates 仅在视觉或可读文字足以支持身份时返回，格式为 "
            "{\"name\":\"\",\"aliases\":[],\"confidence\":0-1,\"evidence_sources\":[],\"evidence_types\":[],\"scope\":\"\"}。"
            "evidence_types 由模型自行判断，可用 text、visual 或 mixed；scope 用 depicted、embedded 或 mentioned 表示身份出现范围。"
            "analysis_flags 只保留值为 true 的键，可用 text_heavy, has_stage, has_screen, has_packaging, has_public_figure_likelihood, classification_uncertain。"
            "不要猜测具体身份，不要解释，不要输出 JSON 以外内容。"
        )

    def _default_analysis(self, image_path: str) -> Dict[str, Any]:
        return normalize_analysis_payload(
            {
                "description": "一张照片",
                "outer_scene_summary": "",
                "inner_content_summary": "",
                "media_types": ["photo"],
                "tags": [],
                "ocr_text": "",
                "person_roles": [],
                "identity_candidates": [],
                "analysis_flags": {},
            },
            tag_min_confidence=0.65,
            identity_text_threshold=0.7,
            identity_visual_threshold=0.92,
        )

    def _create_completion(
        self,
        content: Sequence[Dict[str, Any]] | str,
        *,
        reasoning_effort: Optional[str] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ):
        return create_chat_completion(
            self.client,
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            timeout=self.timeout,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        for candidate in self._build_json_candidates(response_text):
            try:
                data = json.loads(candidate)
            except Exception:
                continue
            if isinstance(data, dict):
                return data
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        return item
        snippet = self._truncate_text(response_text, 240)
        raise ValueError(f"视觉模型返回的分析结果无法解析为对象: {snippet}")

    def _extract_response_text(self, response: Any) -> str:
        return extract_response_text(response)

    @classmethod
    def _clean_response_text(cls, response_text: str) -> str:
        cleaned = (response_text or "").strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()

    @classmethod
    def _iter_json_object_candidates(cls, text: str) -> List[str]:
        candidates: List[str] = []
        start_indexes = [index for index, char in enumerate(text) if char == "{"]
        for start in start_indexes:
            depth = 0
            in_string = False
            escape = False
            for index in range(start, len(text)):
                char = text[index]
                if in_string:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                    continue
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start : index + 1])
                        break
        return candidates

    def _build_json_candidates(self, response_text: str) -> List[str]:
        cleaned = self._clean_response_text(response_text)
        candidates: List[str] = []
        if cleaned:
            candidates.append(cleaned)
            candidates.extend(self._iter_json_object_candidates(cleaned))
        deduped: List[str] = []
        seen = set()
        for item in candidates:
            normalized = item.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    @classmethod
    def _is_expected_analysis_payload(cls, value: Dict[str, Any]) -> bool:
        return any(key in value for key in cls.EXPECTED_ANALYSIS_KEYS)

    def _repair_json_response(self, raw_text: str, *, stage: str) -> Dict[str, Any]:
        repaired_prompt = (
            "请把下面的模型原始输出整理成一个严格 JSON 对象，只返回 JSON，不要解释。\n"
            f"阶段：{stage}。\n"
            "保留原始字段语义，去掉 Markdown、前后缀和多余说明；"
            "如果原文中已经含有 JSON，请修正为可解析的 JSON 对象。\n"
            f"原始输出：{self._truncate_text(raw_text, 4000)}"
        )
        repair_start = time.perf_counter()
        repair_response = self._create_completion(
            repaired_prompt,
            reasoning_effort="low",
            max_tokens=self.repair_max_output_tokens,
            response_format={"type": "json_object"},
        )
        repair_elapsed = time.perf_counter() - repair_start
        repair_text = self._extract_response_text(repair_response)
        repaired_payload = self._parse_json_response(repair_text)
        if not self._is_expected_analysis_payload(repaired_payload):
            raise ValueError("修复后的响应不包含有效分析字段")
        return {
            "payload": repaired_payload,
            "elapsed_seconds": round(repair_elapsed, 4),
            "raw_text_length": len(raw_text or ""),
        }

    @staticmethod
    def _truncate_text(value: Any, limit: int) -> str:
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)].rstrip() + "…"

    def _build_enhancement_context(self, base_analysis: Dict[str, Any]) -> str:
        compact_candidates: List[Dict[str, Any]] = []
        for candidate in list(base_analysis.get("identity_candidates") or [])[:2]:
            if not isinstance(candidate, dict):
                continue
            compact_candidates.append(
                {
                    "name": self._truncate_text(candidate.get("name"), 32),
                    "confidence": round(float(candidate.get("confidence", 0.0)), 4),
                    "evidence_sources": list(candidate.get("evidence_sources") or [])[:3],
                    "evidence_types": list(candidate.get("evidence_types") or [])[:2],
                    "scope": self._truncate_text(candidate.get("scope"), 16),
                }
            )

        compact_flags = {
            str(key): bool(value)
            for key, value in (base_analysis.get("analysis_flags") or {}).items()
            if value
        }
        context = {
            "description": self._truncate_text(base_analysis.get("description"), 80),
            "outer_scene_summary": self._truncate_text(base_analysis.get("outer_scene_summary"), 80),
            "inner_content_summary": self._truncate_text(base_analysis.get("inner_content_summary"), 120),
            "media_types": list(base_analysis.get("media_types") or [])[:4],
            "tags": list(base_analysis.get("tags") or [])[:8],
            "ocr_text_excerpt": self._truncate_text(base_analysis.get("ocr_text"), 200),
            "person_roles": list(base_analysis.get("person_roles") or [])[:4],
            "identity_names": list(base_analysis.get("identity_names") or [])[:4],
            "identity_candidates": compact_candidates,
            "analysis_flags": compact_flags,
        }
        return json.dumps(context, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _get_enhancement_focus(reason: Optional[str]) -> str:
        focus_map = {
            "model_marked_uncertain": "优先重新确认内容类型、关键文字和主体身份。",
            "missing_media_type": "优先修正 media_types，并明确拍到的是载体还是实际场景。",
            "public_figure_needs_review": "优先复核 identity_candidates，区分画面主体与文字提及，只有证据足够时才返回姓名。",
            "person_identity_missing": "优先复核主体身份，并说明 evidence_types 与 scope。",
            "ocr_signal_weak": "优先补强 ocr_text 与 inner_content_summary，只保留关键文字。",
            "retrieval_signal_sparse": "优先补强 inner_content_summary、media_types、tags 和关键 OCR。",
        }
        return focus_map.get(reason or "", "优先修正最影响检索的字段。")

    def _build_enhanced_prompt(self, base_analysis: Dict[str, Any], enhanced_reason: Optional[str] = None) -> str:
        compact_context = self._build_enhancement_context(base_analysis)
        return (
            "同一张图片做第二轮复核，只返回 JSON。\n"
            "目标：不是重写第一次结果，而是针对弱项做更准的修正。\n"
            "输出规则："
            "1. 只返回需要修改或补充的字段，未修改字段省略；"
            "2. 可返回字段仅限 description, outer_scene_summary, inner_content_summary, media_types, tags, ocr_text, identity_candidates, analysis_flags；"
            "3. OCR 只保留最有检索价值的关键文字，尽量控制在 200 字内；"
            "4. analysis_flags 只保留值为 true 的键；"
            "5. 若身份仍不稳，不返回具体姓名；"
            "6. 如返回 identity_candidates，必须区分画面中真正出现的人物、被拍载体里出现的人物、以及仅被文字提及的人物，并给出 evidence_types 与 scope。"
            f"触发原因：{enhanced_reason or 'unknown'}。"
            f"{self._get_enhancement_focus(enhanced_reason)}"
            "不要把第一次结果整份重写回来。"
            f"第一次结果摘要：{compact_context}"
        )

    def get_last_analysis_metrics(self) -> Optional[Dict[str, Any]]:
        return dict(self._last_analysis_metrics) if self._last_analysis_metrics else None

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        encode_start = time.perf_counter()
        image_url = self._get_image_base64(image_path)
        encode_elapsed = time.perf_counter() - encode_start
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": self._build_description_prompt()},
            build_image_url_content(image_url, self.base_url),
        ]

        last_error: Optional[Exception] = None
        self._last_analysis_metrics = {
            "image_encode_seconds": round(encode_elapsed, 4),
            "attempts": [],
            "base_analysis_seconds": 0.0,
            "base_parse_seconds": 0.0,
            "base_repair_seconds": 0.0,
            "base_normalize_seconds": 0.0,
            "enhanced_prompt_seconds": 0.0,
            "enhanced_analysis_seconds": 0.0,
            "enhanced_parse_seconds": 0.0,
            "enhanced_repair_seconds": 0.0,
            "enhanced_normalize_seconds": 0.0,
            "enhanced_triggered": False,
            "enhanced_succeeded": False,
            "used_fallback": False,
        }
        for attempt in range(self.max_retries):
            attempt_metrics: Dict[str, Any] = {"attempt": attempt + 1}
            try:
                base_request_start = time.perf_counter()
                response = self._create_completion(
                    content,
                    reasoning_effort=self.reasoning_effort,
                    max_tokens=self.base_max_output_tokens,
                    response_format={"type": "json_object"},
                )
                base_request_elapsed = time.perf_counter() - base_request_start
                attempt_metrics["base_request_seconds"] = round(base_request_elapsed, 4)
                self._last_analysis_metrics["base_analysis_seconds"] = round(
                    self._last_analysis_metrics["base_analysis_seconds"] + base_request_elapsed, 4
                )

                parse_start = time.perf_counter()
                response_text = self._extract_response_text(response)
                try:
                    parsed = self._parse_json_response(response_text)
                except Exception:
                    repaired = self._repair_json_response(response_text, stage="base")
                    parsed = repaired["payload"]
                    attempt_metrics["base_repair_seconds"] = repaired["elapsed_seconds"]
                    self._last_analysis_metrics["base_repair_seconds"] = round(
                        self._last_analysis_metrics["base_repair_seconds"] + repaired["elapsed_seconds"], 4
                    )
                parse_elapsed = time.perf_counter() - parse_start
                attempt_metrics["base_parse_seconds"] = round(parse_elapsed, 4)
                self._last_analysis_metrics["base_parse_seconds"] = round(
                    self._last_analysis_metrics["base_parse_seconds"] + parse_elapsed, 4
                )

                normalize_start = time.perf_counter()
                normalized = normalize_analysis_payload(
                    parsed,
                    tag_min_confidence=0.65,
                    identity_text_threshold=0.7,
                    identity_visual_threshold=0.92,
                )
                normalize_elapsed = time.perf_counter() - normalize_start
                attempt_metrics["base_normalize_seconds"] = round(normalize_elapsed, 4)
                self._last_analysis_metrics["base_normalize_seconds"] = round(
                    self._last_analysis_metrics["base_normalize_seconds"] + normalize_elapsed, 4
                )

                enhanced_reason = get_enhanced_analysis_reason(normalized)
                enhanced_needed = self.enhanced_analysis_enabled and enhanced_reason is not None
                attempt_metrics["enhanced_triggered"] = enhanced_needed
                attempt_metrics["enhanced_reason"] = enhanced_reason
                self._last_analysis_metrics["enhanced_triggered"] = enhanced_needed
                self._last_analysis_metrics["enhanced_reason"] = enhanced_reason
                if enhanced_needed:
                    try:
                        enhanced_prompt_start = time.perf_counter()
                        enhanced_prompt = self._build_enhanced_prompt(normalized, enhanced_reason)
                        enhanced_prompt_elapsed = time.perf_counter() - enhanced_prompt_start
                        attempt_metrics["enhanced_prompt_seconds"] = round(enhanced_prompt_elapsed, 4)
                        self._last_analysis_metrics["enhanced_prompt_seconds"] = round(
                            self._last_analysis_metrics["enhanced_prompt_seconds"] + enhanced_prompt_elapsed, 4
                        )
                        enhanced_content: List[Dict[str, Any]] = [
                            {"type": "text", "text": enhanced_prompt},
                            build_image_url_content(image_url, self.base_url),
                        ]
                        enhanced_request_start = time.perf_counter()
                        enhanced_response = self._create_completion(
                            enhanced_content,
                            reasoning_effort=self.enhanced_reasoning_effort,
                            max_tokens=self.enhanced_max_output_tokens,
                            response_format={"type": "json_object"},
                        )
                        enhanced_request_elapsed = time.perf_counter() - enhanced_request_start
                        attempt_metrics["enhanced_request_seconds"] = round(enhanced_request_elapsed, 4)
                        self._last_analysis_metrics["enhanced_analysis_seconds"] = round(
                            self._last_analysis_metrics["enhanced_analysis_seconds"] + enhanced_request_elapsed, 4
                        )

                        enhanced_parse_start = time.perf_counter()
                        enhanced_response_text = self._extract_response_text(enhanced_response)
                        try:
                            enhanced_parsed = self._parse_json_response(enhanced_response_text)
                        except Exception:
                            repaired = self._repair_json_response(enhanced_response_text, stage="enhanced")
                            enhanced_parsed = repaired["payload"]
                            attempt_metrics["enhanced_repair_seconds"] = repaired["elapsed_seconds"]
                            self._last_analysis_metrics["enhanced_repair_seconds"] = round(
                                self._last_analysis_metrics["enhanced_repair_seconds"] + repaired["elapsed_seconds"], 4
                            )
                        enhanced_parse_elapsed = time.perf_counter() - enhanced_parse_start
                        attempt_metrics["enhanced_parse_seconds"] = round(enhanced_parse_elapsed, 4)
                        self._last_analysis_metrics["enhanced_parse_seconds"] = round(
                            self._last_analysis_metrics["enhanced_parse_seconds"] + enhanced_parse_elapsed, 4
                        )

                        merged = dict(normalized)
                        merged.update(enhanced_parsed)
                        enhanced_normalize_start = time.perf_counter()
                        normalized = normalize_analysis_payload(
                            merged,
                            tag_min_confidence=0.65,
                            identity_text_threshold=0.7,
                            identity_visual_threshold=0.92,
                        )
                        enhanced_normalize_elapsed = time.perf_counter() - enhanced_normalize_start
                        attempt_metrics["enhanced_normalize_seconds"] = round(enhanced_normalize_elapsed, 4)
                        self._last_analysis_metrics["enhanced_normalize_seconds"] = round(
                            self._last_analysis_metrics["enhanced_normalize_seconds"] + enhanced_normalize_elapsed, 4
                        )
                        attempt_metrics["enhanced_succeeded"] = True
                        self._last_analysis_metrics["enhanced_succeeded"] = True
                    except Exception as exc:
                        attempt_metrics["enhanced_error"] = str(exc)
                        attempt_metrics["enhanced_succeeded"] = False

                attempt_metrics["status"] = "success"
                self._last_analysis_metrics["attempts"].append(attempt_metrics)
                return normalized
            except Exception as exc:
                last_error = exc
                attempt_metrics["status"] = "failed"
                attempt_metrics["error"] = str(exc)
                self._last_analysis_metrics["attempts"].append(attempt_metrics)
                if attempt == self.max_retries - 1:
                    break
                time.sleep(1)

        if last_error is not None:
            raise ValueError(f"生成结构化分析失败: {last_error}") from last_error
        raise ValueError("生成结构化分析失败")

    def generate_description(self, image_path: str) -> str:
        analysis = self.analyze_image(image_path)
        description = str(analysis.get("description") or "").strip()
        if not description:
            raise ValueError("视觉模型返回空描述")
        return description

    def generate_description_batch(self, image_paths: List[str]) -> List[str]:
        return [self.generate_description(path) for path in image_paths]

    def analyze_image_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        return [self.analyze_image(path) for path in image_paths]


class LocalVisionLLMService(VisionLLMService):
    """测试用本地视觉服务。"""

    def __init__(self) -> None:
        self._last_analysis_metrics: Optional[Dict[str, Any]] = None

    def get_last_analysis_metrics(self) -> Optional[Dict[str, Any]]:
        return dict(self._last_analysis_metrics) if self._last_analysis_metrics else None

    def generate_description(self, image_path: str) -> str:
        return self.analyze_image(image_path)["description"]

    def generate_description_batch(self, image_paths: List[str]) -> List[str]:
        return [self.generate_description(path) for path in image_paths]

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        width, height = get_image_dimensions(image_path)
        self._last_analysis_metrics = {
            "image_encode_seconds": 0.0,
            "attempts": [{"attempt": 1, "status": "success", "base_request_seconds": 0.0}],
            "base_analysis_seconds": 0.0,
            "base_parse_seconds": 0.0,
            "base_normalize_seconds": 0.0,
            "enhanced_prompt_seconds": 0.0,
            "enhanced_analysis_seconds": 0.0,
            "enhanced_parse_seconds": 0.0,
            "enhanced_normalize_seconds": 0.0,
            "enhanced_triggered": False,
            "enhanced_succeeded": False,
            "used_fallback": False,
        }
        if width <= 0 or height <= 0:
            return {
                "description": "一张本地生成的图片描述",
                "outer_scene_summary": "一张图片",
                "inner_content_summary": "",
                "media_types": ["photo"],
                "tags": ["图片"],
                "ocr_text": "",
                "person_roles": [],
                "identity_candidates": [],
                "identity_names": [],
                "identity_evidence": [],
                "analysis_flags": {},
                "embedding_text": "photo 图片 一张本地生成的图片描述",
                "retrieval_text": "photo 图片 一张本地生成的图片描述",
            }
        return {
            "description": f"一张本地生成的图片描述，分辨率为{width}x{height}",
            "outer_scene_summary": f"一张分辨率为{width}x{height}的图片",
            "inner_content_summary": "",
            "media_types": ["photo"],
            "tags": ["图片", f"{width}x{height}"],
            "ocr_text": "",
            "person_roles": [],
            "identity_candidates": [],
            "identity_names": [],
            "identity_evidence": [],
            "analysis_flags": {},
            "embedding_text": f"photo 图片 {width}x{height}",
            "retrieval_text": f"photo 图片 {width}x{height}",
        }

    def analyze_image_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        return [self.analyze_image(path) for path in image_paths]
