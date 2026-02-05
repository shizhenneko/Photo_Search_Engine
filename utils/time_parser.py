from __future__ import annotations

import json
import re
import time
from datetime import datetime
from typing import Any, Dict, Optional

from openai import OpenAI


class TimeParser:
    """
    使用LLM语义理解解析时间约束。

    Attributes:
        api_key (str): OpenRouter API密钥
        model_name (str): 模型名称
        base_url (str): OpenRouter API地址
        timeout (int): 超时时间
        max_retries (int): 最大重试次数
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "openai/gpt-3.5-turbo",
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: int = 10,
        max_retries: int = 3,
        client: Optional[OpenAI] = None,
    ) -> None:
        """
        初始化时间解析器。

        Args:
            api_key (str): OpenRouter API密钥
            model_name (str): 模型名称，默认openai/gpt-3.5-turbo
            base_url (str): OpenRouter API地址
            timeout (int): API超时时间（秒）
            max_retries (int): 最大重试次数
            client (Optional[OpenAI]): OpenAI客户端实例
        """
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY 未设置")
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = client or OpenAI(api_key=api_key, base_url=base_url)

    def has_time_terms(self, query: str) -> bool:
        """
        快速预检测查询中是否包含明显的时间词。

        Args:
            query (str): 用户查询文本

        Returns:
            bool: True表示包含时间词，应调用LLM解析；False表示无需解析
        """
        # 相对时间词
        relative_patterns = [
            r"去年|今年|前年|明年",
            r"上个月|下个月|这个月|上上个月",
            r"上周|下周|本周|上周|下周|这周",
            r"上个?星期|下个?星期|这个星期",
            r"前几天|最近|之前|之后"
        ]

        # 季节词
        season_patterns = [
            r"春天|夏天|秋天|冬天",
            r"春季|夏季|秋季|冬季",
            r"春|夏|秋|冬"
        ]

        # 绝对日期模式
        date_patterns = [
            r"\d{4}年",
            r"\d{4}-\d{1,2}(-\d{1,2})?",
            r"\d{1,2}月(\d{1,2}日?)?",
            r"\d{1,2}日"
        ]

        all_patterns = relative_patterns + season_patterns + date_patterns

        return any(re.search(pattern, query) for pattern in all_patterns) if isinstance(query, str) else False

    def extract_time_constraints(self, query: str) -> Dict[str, Any]:
        """
        使用LLM语义理解解析时间约束。

        优化：先进行预检测，无时间词直接返回None，节省API成本。

        Args:
            query (str): 用户查询文本

        Returns:
            Dict[str, Any]: 时间约束字典
        """
        # 预检测：快速过滤无时间词的查询
        if not self.has_time_terms(query):
            return {"start_date": None, "end_date": None, "precision": "none"}

        current_date = datetime.now().strftime("%Y-%m-%d")
        prompt = f"""当前日期：{current_date}（格式：YYYY-MM-DD）

用户查询：{query}

请分析用户查询中的时间约束，返回JSON格式：
{{
  "has_time_constraint": true/false,
  "start_date": "YYYY-MM-DD" 或 null,
  "end_date": "YYYY-MM-DD" 或 null,
  "reasoning": "简要说明解析逻辑"
}}

规则：
1. 如果没有明确的时间词，has_time_constraint=false，其他字段为null
2. **特殊规则：如果仅包含季节词（如"夏天"）或时段词（如"早上"）但没有具体年份限定（如"2023年"、"去年"、"今年"），请视为泛指，has_time_constraint=false，不要生成日期范围。**
3. 相对时间基于当前日期计算：
   - "去年" -> 去年全年
   - "今年" -> 今年全年
   - "上个月" -> 上个月
   - "冬天" -> 当年12月到次年2月
   - "去年冬天" -> 去年12月到今年2月
3. 季节定义月：
   - 春：3月1日-5月31日
   - 夏：6月1日-8月31日
   - 秋：9月1日-11月30日
   - 冬：12月1日-次年2月28/29日
4. 日期范围包含边界"""

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    response_format={"type": "json_object"},
                    timeout=self.timeout,
                )
                result = json.loads(response.choices[0].message.content)
                if not result.get("has_time_constraint"):
                    return {"start_date": None, "end_date": None, "precision": "none"}

                start_date = result.get("start_date")
                end_date = result.get("end_date")
                precision = self._infer_precision(start_date, end_date)
                return {"start_date": start_date, "end_date": end_date, "precision": precision}
            except Exception:
                if attempt == self.max_retries - 1:
                    break
                time.sleep(1)

        return {"start_date": None, "end_date": None, "precision": "none"}

    def _infer_precision(self, start_date: Optional[str], end_date: Optional[str]) -> str:
        """
        根据日期范围推断精度级别。

        Args:
            start_date (Optional[str]): 起始日期
            end_date (Optional[str]): 结束日期

        Returns:
            str: 精度级别
        """
        if not start_date or not end_date:
            return "none"

        try:
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
        except Exception:
            return "none"

        delta = end - start
        if end.year != start.year:
            return "season" if delta.days <= 90 else "range"
        if delta.days <= 31:
            return "month"
        if delta.days <= 90:
            return "season"
        return "year"
