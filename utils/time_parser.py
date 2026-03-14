from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Dict, Optional

from openai import OpenAI


class TimeParser:
    """使用 SU8 上的文本模型解析时间约束。"""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str,
        reasoning_effort: str = "low",
        timeout: int = 10,
        max_retries: int = 3,
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
        self.client = client or OpenAI(api_key=api_key, base_url=base_url)

    def has_time_terms(self, query: str) -> bool:
        if not query or not query.strip():
            return False
        current_date = datetime.now().strftime("%Y-%m-%d")
        prompt = f"""当前日期：{current_date}

用户查询：{query}

请判断这个查询是否包含时间约束，只返回 JSON：
{{
  "has_time_constraint": true 或 false
}}

要求：
- 只根据用户表达判断。
- 相对时间、绝对时间、季节、时段都算时间约束。
- 没有时间语义就返回 false。"""

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    response_format={"type": "json_object"},
                    timeout=self.timeout,
                    extra_body={"reasoning_effort": self.reasoning_effort},
                )
                payload = json.loads(response.choices[0].message.content)
                return bool(payload.get("has_time_constraint"))
            except Exception:
                if attempt == self.max_retries - 1:
                    break
                time.sleep(1)
        return False

    def extract_time_constraints(self, query: str) -> Dict[str, Any]:
        if not self.has_time_terms(query):
            return {"start_date": None, "end_date": None, "precision": "none"}

        current_date = datetime.now().strftime("%Y-%m-%d")
        prompt = f"""当前日期：{current_date}

用户查询：{query}

请只返回 JSON：
{{
  "has_time_constraint": true,
  "start_date": "YYYY-MM-DD" 或 null,
  "end_date": "YYYY-MM-DD" 或 null
}}

规则：
1. 只有明确年份、月份、日期或相对时间时才返回日期范围。
2. 仅出现季节词或时段词但没有年份限定时，不生成日期范围。
3. 返回内容必须是合法 JSON。"""

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    response_format={"type": "json_object"},
                    timeout=self.timeout,
                    extra_body={"reasoning_effort": self.reasoning_effort},
                )
                payload = json.loads(response.choices[0].message.content)
                if not payload.get("has_time_constraint"):
                    return {"start_date": None, "end_date": None, "precision": "none"}
                start_date = payload.get("start_date")
                end_date = payload.get("end_date")
                return {
                    "start_date": start_date,
                    "end_date": end_date,
                    "precision": self._infer_precision(start_date, end_date),
                }
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(1)

        if last_error is not None:
            _ = last_error
        return {"start_date": None, "end_date": None, "precision": "none"}

    def _infer_precision(self, start_date: Optional[str], end_date: Optional[str]) -> str:
        if not start_date or not end_date:
            return "none"
        try:
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
        except Exception:
            return "none"

        delta = end - start
        if end.year != start.year:
            return "season" if delta.days <= 95 else "range"
        if delta.days <= 31:
            return "month"
        if delta.days <= 95:
            return "season"
        return "year"
