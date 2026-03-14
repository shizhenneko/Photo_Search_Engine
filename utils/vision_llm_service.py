from __future__ import annotations

import base64
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from openai import OpenAI

from utils.image_parser import get_image_dimensions, resize_and_optimize_image
from utils.structured_analysis import (
    normalize_analysis_payload,
    should_run_enhanced_analysis,
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
    """通过 SU8 中转调用视觉模型。"""

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
        client: Optional[OpenAI] = None,
    ) -> None:
        if not api_key:
            raise ValueError("SU8_API_KEY 未设置")
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.use_base64 = use_base64
        self.image_max_size = max(256, min(4096, image_max_size))
        self.image_quality = max(1, min(100, image_quality))
        self.image_format = image_format.upper() if image_format.upper() in {"JPEG", "PNG", "WEBP"} else "WEBP"
        self.client = client or OpenAI(api_key=api_key, base_url=base_url)
        self._last_analysis_metrics: Optional[Dict[str, Any]] = None

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
            "你是图片结构化理解器。请严格观察图片并返回 JSON。\n"
            "要求：\n"
            "1. 只描述可见内容，但允许识别载体类型、可读文字、公众人物候选和媒介属性。\n"
            "2. description 用于前端展示，应自然、准确、简洁。\n"
            "3. outer_scene_summary 描述相机实际拍到的外层场景。\n"
            "4. inner_content_summary 描述被拍对象内部承载的内容，例如专辑封面、海报、屏幕内容。\n"
            "5. media_types 必须从这些值中选择：album_cover, poster, stage_performance, screen, artwork_print, merch, document, graffiti, photo, other。\n"
            "6. tags 返回数组，每项格式为 {\"tag\": \"...\", \"confidence\": 0-1}。\n"
            "7. identity_candidates 返回数组，每项格式为 {\"name\": \"...\", \"aliases\": [], \"confidence\": 0-1, \"evidence_sources\": []}。\n"
            "8. analysis_flags 返回布尔字典，可用键包括 text_heavy, has_stage, has_screen, has_packaging, has_public_figure_likelihood。\n"
            "9. 对公众人物识别必须保守：只有当视觉特征或可读文字足以支持具体身份时，才返回具体姓名；否则返回空数组，不要用“像某某”硬猜。\n"
            "10. evidence_sources 必须明确区分文字证据和视觉证据，例如 readable_text、visible_text、ocr_text、face_similarity、signature_stage_pose、hairstyle、tattoo。\n"
            "11. 只返回 JSON，不要输出解释。"
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

    def _create_completion(self, content: List[Dict[str, Any]]):
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            timeout=self.timeout,
            response_format={"type": "json_object"},
            extra_body={"reasoning_effort": self.reasoning_effort},
        )

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        cleaned = (response_text or "").strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json", "", 1).strip()
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("视觉模型返回的分析结果不是对象")
        return data

    def _build_enhanced_prompt(self, base_analysis: Dict[str, Any]) -> str:
        return (
            "你将看到同一张图片，请基于已有初步分析做增强识别，只返回 JSON。\n"
            "重点补强：\n"
            "1. 更准确的 inner_content_summary。\n"
            "2. 更准确的 OCR 文本。\n"
            "3. 若图片中可能出现公众人物，请返回更高质量的 identity_candidates。\n"
            "4. 若图片实际是专辑封面、海报、屏幕、舞台或周边，请修正 media_types。\n"
            "5. 若公众人物无法凭视觉稳定唯一识别，请不要返回具体姓名；宁可留空，也不要误认。\n"
            f"初步分析：{json.dumps(base_analysis, ensure_ascii=False)}"
        )

    def get_last_analysis_metrics(self) -> Optional[Dict[str, Any]]:
        return dict(self._last_analysis_metrics) if self._last_analysis_metrics else None

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        encode_start = time.perf_counter()
        image_url = self._get_image_base64(image_path)
        encode_elapsed = time.perf_counter() - encode_start
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": self._build_description_prompt()},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]

        last_error: Optional[Exception] = None
        self._last_analysis_metrics = {
            "image_encode_seconds": round(encode_elapsed, 4),
            "attempts": [],
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
        for attempt in range(self.max_retries):
            attempt_metrics: Dict[str, Any] = {"attempt": attempt + 1}
            try:
                base_request_start = time.perf_counter()
                response = self._create_completion(content)
                base_request_elapsed = time.perf_counter() - base_request_start
                attempt_metrics["base_request_seconds"] = round(base_request_elapsed, 4)
                self._last_analysis_metrics["base_analysis_seconds"] = round(
                    self._last_analysis_metrics["base_analysis_seconds"] + base_request_elapsed, 4
                )

                parse_start = time.perf_counter()
                parsed = self._parse_json_response(response.choices[0].message.content or "")
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

                enhanced_needed = should_run_enhanced_analysis(normalized)
                attempt_metrics["enhanced_triggered"] = enhanced_needed
                self._last_analysis_metrics["enhanced_triggered"] = enhanced_needed
                if enhanced_needed:
                    try:
                        enhanced_prompt_start = time.perf_counter()
                        enhanced_prompt = self._build_enhanced_prompt(normalized)
                        enhanced_prompt_elapsed = time.perf_counter() - enhanced_prompt_start
                        attempt_metrics["enhanced_prompt_seconds"] = round(enhanced_prompt_elapsed, 4)
                        self._last_analysis_metrics["enhanced_prompt_seconds"] = round(
                            self._last_analysis_metrics["enhanced_prompt_seconds"] + enhanced_prompt_elapsed, 4
                        )
                        enhanced_content: List[Dict[str, Any]] = [
                            {"type": "text", "text": enhanced_prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ]
                        enhanced_request_start = time.perf_counter()
                        enhanced_response = self._create_completion(enhanced_content)
                        enhanced_request_elapsed = time.perf_counter() - enhanced_request_start
                        attempt_metrics["enhanced_request_seconds"] = round(enhanced_request_elapsed, 4)
                        self._last_analysis_metrics["enhanced_analysis_seconds"] = round(
                            self._last_analysis_metrics["enhanced_analysis_seconds"] + enhanced_request_elapsed, 4
                        )

                        enhanced_parse_start = time.perf_counter()
                        enhanced_parsed = self._parse_json_response(
                            enhanced_response.choices[0].message.content or ""
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
            "retrieval_text": f"photo 图片 {width}x{height}",
        }

    def analyze_image_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        return [self.analyze_image(path) for path in image_paths]
