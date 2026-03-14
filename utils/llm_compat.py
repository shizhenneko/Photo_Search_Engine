from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from unittest.mock import Mock


_LOCAL_OLLAMA_HOSTS = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "host.docker.internal",
}


def is_local_ollama_base_url(base_url: str) -> bool:
    parsed = urlparse((base_url or "").strip())
    host = (parsed.hostname or "").lower()
    return host in _LOCAL_OLLAMA_HOSTS and (parsed.port in {None, 11434})


def is_ollama_base_url(base_url: str) -> bool:
    parsed = urlparse((base_url or "").strip())
    host = (parsed.hostname or "").lower()
    return is_local_ollama_base_url(base_url) or host.endswith("ollama.com")


def requires_api_key(base_url: str) -> bool:
    return not is_local_ollama_base_url(base_url)


def resolve_api_key(api_key: str, base_url: str) -> str:
    if api_key:
        return api_key
    if is_local_ollama_base_url(base_url):
        return "ollama"
    return ""


def normalize_openai_base_url(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        return normalized
    if not is_ollama_base_url(normalized):
        return normalized
    if normalized.endswith("/v1"):
        return normalized
    if normalized.endswith("/api"):
        return normalized[:-4] + "/v1"
    parsed = urlparse(normalized)
    if parsed.path in {"", "/"}:
        return normalized + "/v1"
    return normalized


def build_image_url_content(image_url: str, base_url: str) -> Dict[str, Any]:
    if is_ollama_base_url(base_url):
        return {"type": "image_url", "image_url": image_url}
    return {"type": "image_url", "image_url": {"url": image_url}}


def safe_get_attr(value: Any, name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    if isinstance(value, Mock):
        return vars(value).get(name)
    return getattr(value, name, None)


def extract_text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="ignore")
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        parsed = content.get("parsed")
        if isinstance(parsed, dict):
            return str(parsed)
        json_value = content.get("json")
        if isinstance(json_value, dict):
            return str(json_value)
        nested = content.get("content")
        if isinstance(nested, str):
            return nested
        if nested is not None:
            return extract_text_from_content(nested)
        return str(content)
    if isinstance(content, list):
        parts = [extract_text_from_content(item) for item in content]
        return "\n".join(part for part in parts if part).strip()

    text_attr = safe_get_attr(content, "text")
    if isinstance(text_attr, str):
        return text_attr
    nested_content = safe_get_attr(content, "content")
    if nested_content is not None:
        return extract_text_from_content(nested_content)
    return str(content)


def collect_response_text_candidates(value: Any, *, depth: int = 0, seen: Optional[set[int]] = None) -> List[str]:
    if value is None or depth > 6:
        return []
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return []
    seen.add(value_id)

    candidates: List[str] = []
    if isinstance(value, str):
        text = value.strip()
        if text:
            candidates.append(text)
        return candidates
    if isinstance(value, bytes):
        return collect_response_text_candidates(value.decode("utf-8", errors="ignore"), depth=depth + 1, seen=seen)
    if isinstance(value, dict):
        for key in ("output_text", "parsed", "json", "text", "content", "message", "choices", "output"):
            if key in value:
                candidates.extend(collect_response_text_candidates(value[key], depth=depth + 1, seen=seen))
        for key, item in value.items():
            if key in {"output_text", "parsed", "json", "text", "content", "message", "choices", "output"}:
                continue
            candidates.extend(collect_response_text_candidates(item, depth=depth + 1, seen=seen))
        return candidates
    if isinstance(value, list):
        for item in value:
            candidates.extend(collect_response_text_candidates(item, depth=depth + 1, seen=seen))
        return candidates

    model_dump = safe_get_attr(value, "model_dump")
    if callable(model_dump) and not isinstance(value, Mock):
        try:
            candidates.extend(collect_response_text_candidates(model_dump(), depth=depth + 1, seen=seen))
        except Exception:
            pass

    for attr in ("output_text", "parsed", "json", "text", "content", "message", "choices", "output"):
        attr_value = safe_get_attr(value, attr)
        if attr_value is not None:
            candidates.extend(collect_response_text_candidates(attr_value, depth=depth + 1, seen=seen))
    return candidates


def extract_response_text(response: Any) -> str:
    if response is None:
        raise ValueError("模型返回为空")
    for candidate in collect_response_text_candidates(response):
        extracted = extract_text_from_content(candidate).strip()
        if extracted:
            return extracted
    raise ValueError(f"无法提取模型响应文本: {type(response).__name__}")


def create_chat_completion(
    client: Any,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    timeout: int,
    temperature: Optional[float] = None,
    response_format: Optional[Dict[str, Any]] = None,
    reasoning_effort: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> Any:
    base_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "timeout": timeout,
    }
    if temperature is not None:
        base_kwargs["temperature"] = temperature
    if max_tokens is not None:
        base_kwargs["max_tokens"] = max_tokens

    attempts: List[Dict[str, Any]] = []
    if response_format is not None or reasoning_effort:
        with_all = dict(base_kwargs)
        if response_format is not None:
            with_all["response_format"] = response_format
        if reasoning_effort:
            with_all["extra_body"] = {"reasoning_effort": reasoning_effort}
        attempts.append(with_all)

    if response_format is not None:
        without_extra = dict(base_kwargs)
        without_extra["response_format"] = response_format
        attempts.append(without_extra)

    if reasoning_effort:
        without_response_format = dict(base_kwargs)
        without_response_format["extra_body"] = {"reasoning_effort": reasoning_effort}
        attempts.append(without_response_format)

    attempts.append(dict(base_kwargs))

    last_error: Optional[Exception] = None
    seen_signatures = set()
    for kwargs in attempts:
        signature = tuple(sorted(kwargs.keys()))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # pragma: no cover - error path depends on vendor behavior
            last_error = exc

    if last_error is not None:
        raise last_error
    raise ValueError("模型调用失败")
