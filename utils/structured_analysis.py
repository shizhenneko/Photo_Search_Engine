from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple


OCR_HEAVY_THRESHOLD = 36
OCR_STRONG_THRESHOLD = 48
RICH_DESCRIPTION_THRESHOLD = 24
RICH_INNER_SUMMARY_THRESHOLD = 18
MIN_SIGNAL_SCORE_FOR_SKIP = 3


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _dedupe_keep_order(values: Sequence[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for value in values:
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        ordered.append(value)
        seen.add(key)
    return ordered


def normalize_media_types(values: Sequence[Any]) -> List[str]:
    normalized = [_normalize_text(value) for value in values or []]
    return _dedupe_keep_order([value for value in normalized if value])


def normalize_tags(values: Sequence[Any], min_confidence: float) -> List[str]:
    tags: List[str] = []
    for item in values or []:
        if isinstance(item, dict):
            tag = _normalize_text(item.get("tag") or item.get("name") or item.get("value"))
            confidence = item.get("confidence")
            try:
                score = float(confidence) if confidence is not None else 1.0
            except (TypeError, ValueError):
                score = 0.0
        else:
            tag = _normalize_text(item)
            score = 1.0
        if not tag or score < min_confidence:
            continue
        tags.append(tag)
    return _dedupe_keep_order(tags)


def normalize_ocr_text(value: Any) -> str:
    text = _normalize_text(value)
    return text[:400] if text else ""


def normalize_person_roles(values: Sequence[Any]) -> List[str]:
    roles = [_normalize_text(value) for value in values or []]
    return _dedupe_keep_order([role for role in roles if role])


def normalize_analysis_flags(value: Any) -> Dict[str, bool]:
    if not isinstance(value, dict):
        return {}
    flags: Dict[str, bool] = {}
    for key, flag in value.items():
        normalized_key = _normalize_text(key)
        if not normalized_key:
            continue
        flags[normalized_key] = bool(flag)
    return flags


def _normalize_identity_candidate(raw: Any) -> Dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    name = _normalize_text(raw.get("name"))
    if not name:
        return None
    aliases = [
        alias
        for alias in (_normalize_text(alias) for alias in raw.get("aliases") or [])
        if alias
    ]
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    evidence_sources = [
        source
        for source in (_normalize_text(source) for source in raw.get("evidence_sources") or [])
        if source
    ]
    evidence_types = [
        evidence_type
        for evidence_type in (_normalize_text(value) for value in raw.get("evidence_types") or [])
        if evidence_type
    ]
    return {
        "name": name,
        "aliases": _dedupe_keep_order(aliases),
        "confidence": round(confidence, 4),
        "evidence_sources": _dedupe_keep_order(evidence_sources),
        "evidence_types": _dedupe_keep_order(evidence_types),
    }


def _candidate_threshold(candidate: Dict[str, Any], text_threshold: float, visual_threshold: float) -> float:
    evidence_types = {value.lower() for value in candidate.get("evidence_types") or []}
    if "text" in evidence_types:
        return text_threshold
    if "visual" in evidence_types:
        return visual_threshold
    return max(text_threshold, visual_threshold)


def select_identity_names(
    candidates: Sequence[Any],
    text_threshold: float,
    visual_threshold: float,
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    selected_names: List[str] = []
    selected_evidence: List[str] = []
    normalized_candidates: List[Dict[str, Any]] = []

    for raw in candidates or []:
        normalized = _normalize_identity_candidate(raw)
        if normalized is None:
            continue
        normalized_candidates.append(normalized)
        if normalized["confidence"] < _candidate_threshold(normalized, text_threshold, visual_threshold):
            continue
        selected_names.append(normalized["name"])
        selected_names.extend(normalized["aliases"])
        selected_evidence.extend(normalized["evidence_sources"])

    return (
        _dedupe_keep_order(selected_names),
        _dedupe_keep_order(selected_evidence),
        normalized_candidates,
    )


def should_run_enhanced_analysis(analysis: Dict[str, Any]) -> bool:
    return get_enhanced_analysis_reason(analysis) is not None


def _has_confident_identity_candidate(candidates: Sequence[Any], threshold: float = 0.7) -> bool:
    for candidate in candidates or []:
        if not isinstance(candidate, dict):
            continue
        try:
            confidence = float(candidate.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence >= threshold:
            return True
    return False


def get_enhanced_analysis_reason(analysis: Dict[str, Any]) -> str | None:
    media_types = normalize_media_types(analysis.get("media_types") or [])
    person_roles = normalize_person_roles(analysis.get("person_roles") or [])
    flags = normalize_analysis_flags(analysis.get("analysis_flags"))
    ocr_text = normalize_ocr_text(analysis.get("ocr_text"))
    description = _normalize_text(analysis.get("description"))
    inner_summary = _normalize_text(analysis.get("inner_content_summary"))
    tags = normalize_tags(analysis.get("tags") or [], min_confidence=0.0)
    identity_names = [_normalize_text(name) for name in analysis.get("identity_names") or [] if _normalize_text(name)]
    identity_candidates = analysis.get("identity_candidates") or []

    text_heavy = bool(flags.get("text_heavy")) or len(ocr_text) >= OCR_HEAVY_THRESHOLD
    classification_uncertain = bool(flags.get("classification_uncertain"))
    missing_media = not bool(media_types)
    rich_description = len(description) >= RICH_DESCRIPTION_THRESHOLD
    rich_inner_summary = len(inner_summary) >= RICH_INNER_SUMMARY_THRESHOLD
    strong_ocr = len(ocr_text) >= (OCR_STRONG_THRESHOLD if text_heavy else 16)
    enough_tags = len(tags) >= 2
    confident_identity_candidate = _has_confident_identity_candidate(identity_candidates)

    retrieval_signal_score = 0
    if not missing_media:
        retrieval_signal_score += 1
    if rich_description:
        retrieval_signal_score += 1
    if rich_inner_summary:
        retrieval_signal_score += 1
    if strong_ocr:
        retrieval_signal_score += 1
    if enough_tags:
        retrieval_signal_score += 1
    if identity_names or confident_identity_candidate:
        retrieval_signal_score += 1

    if classification_uncertain:
        return "model_marked_uncertain"
    if person_roles and not identity_names and not confident_identity_candidate and retrieval_signal_score < MIN_SIGNAL_SCORE_FOR_SKIP:
        return "person_identity_missing"
    if text_heavy and not strong_ocr and retrieval_signal_score < MIN_SIGNAL_SCORE_FOR_SKIP:
        return "ocr_signal_weak"
    if retrieval_signal_score < MIN_SIGNAL_SCORE_FOR_SKIP and (
        missing_media or not rich_inner_summary or (text_heavy and not strong_ocr)
    ):
        return "retrieval_signal_sparse"
    return None


def build_retrieval_text(
    analysis: Dict[str, Any],
    identity_names: Sequence[str],
    ocr_text: str,
) -> str:
    parts: List[str] = []
    media_types = normalize_media_types(analysis.get("media_types") or [])
    if media_types:
        parts.append(" ".join(media_types))
    tags = normalize_tags(analysis.get("tags") or [], min_confidence=0.0)
    if tags:
        parts.append(" ".join(tags))
    outer = _normalize_text(analysis.get("outer_scene_summary"))
    if outer:
        parts.append(outer)
    inner = _normalize_text(analysis.get("inner_content_summary"))
    if inner:
        parts.append(inner)
    if ocr_text:
        parts.append(ocr_text)
    identity_text = " ".join(_dedupe_keep_order([_normalize_text(name) for name in identity_names]))
    if identity_text:
        parts.append(identity_text)
    if not parts:
        parts.append(_normalize_text(analysis.get("description")) or "一张照片")
    return " ".join(part for part in parts if part).strip()


def normalize_analysis_payload(
    payload: Dict[str, Any],
    tag_min_confidence: float,
    identity_text_threshold: float,
    identity_visual_threshold: float,
) -> Dict[str, Any]:
    description = _normalize_text(payload.get("description")) or "一张照片"
    outer = _normalize_text(payload.get("outer_scene_summary"))
    inner = _normalize_text(payload.get("inner_content_summary"))
    media_types = normalize_media_types(payload.get("media_types") or [])
    tags = normalize_tags(payload.get("tags") or [], min_confidence=tag_min_confidence)
    ocr_text = normalize_ocr_text(payload.get("ocr_text"))
    person_roles = normalize_person_roles(payload.get("person_roles") or [])
    analysis_flags = normalize_analysis_flags(payload.get("analysis_flags"))
    identity_names, identity_evidence, identity_candidates = select_identity_names(
        payload.get("identity_candidates") or [],
        text_threshold=identity_text_threshold,
        visual_threshold=identity_visual_threshold,
    )
    normalized = {
        "description": description,
        "outer_scene_summary": outer,
        "inner_content_summary": inner,
        "media_types": media_types,
        "tags": tags,
        "ocr_text": ocr_text,
        "person_roles": person_roles,
        "identity_candidates": identity_candidates,
        "identity_names": identity_names,
        "identity_evidence": identity_evidence,
        "analysis_flags": analysis_flags,
    }
    normalized["retrieval_text"] = build_retrieval_text(normalized, identity_names, ocr_text)
    return normalized


def build_match_summary(metadata: Dict[str, Any]) -> Dict[str, Any]:
    identities = metadata.get("identity_names") or []
    evidence = metadata.get("identity_evidence") or []
    ocr_text = normalize_ocr_text(metadata.get("ocr_text"))
    return {
        "media_types": list(metadata.get("media_types") or []),
        "top_tags": list(metadata.get("top_tags") or metadata.get("tags") or [])[:8],
        "identities": list(identities),
        "identity_evidence": list(evidence),
        "ocr_excerpt": ocr_text[:120],
    }
