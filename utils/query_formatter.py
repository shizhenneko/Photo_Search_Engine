from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Dict, Optional

from openai import OpenAI


class QueryFormatter:
    """
    使用 LLM 对用户查询进行格式化，提取核心检索需求。
    
    架构说明：
    - search_text：纯视觉语义描述，用于生成 embedding 向量
    - time_hint/season/time_period：独立时间字段，用于 ES 过滤
    
    输出示例：
    {
        "search_text": "海滩沙滩边的家庭合影 温馨的户外场景",
        "scene": "海滩",
        "season": "夏天",
        "time_period": null
    }
    
    Attributes:
        api_key (str): LLM API 密钥
        model_name (str): 模型名称
        base_url (str): API 基础地址
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str,
        reasoning_effort: str = "low",
        timeout: int = 15,
        max_retries: int = 3,
        client: Optional[OpenAI] = None,
    ) -> None:
        """
        初始化查询格式化服务。
        
        Args:
            api_key (str): LLM API 密钥（必填）
            model_name (str): 模型名称
            base_url (str): API 基础地址
            timeout (int): 超时时间（秒）
            max_retries (int): 最大重试次数
            client (Optional[OpenAI]): 预配置客户端（用于测试）
        
        Raises:
            ValueError: API 密钥未设置时抛出
        """
        if not api_key:
            raise ValueError("QUERY_FORMAT_API_KEY 未设置")
        
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = client or OpenAI(api_key=api_key, base_url=base_url)
    
    def format_query(self, user_query: str) -> Dict[str, Any]:
        """
        格式化用户查询，提取核心检索需求。
        
        Args:
            user_query (str): 原始用户查询
        
        Returns:
            Dict[str, Any]: 格式化结果
                - search_text: str（格式化后的检索文本）
                - scene: str（场景描述）
                - time_hint: str（时间提示，如"2020年11月"）
                - season: str（季节，如"秋天"）
                - time_period: str（时段，如"白天"）
                - original_query: str（原始查询）
        
        Example:
            >>> formatter.format_query("请展示一张公园的照片")
            {
                "search_text": "公园草地树木 户外休闲场景 绿意盎然",
                "scene": "公园",
                "time_hint": None,
                "season": None,
                "time_period": None,
                "original_query": "请展示一张公园的照片"
            }
        """
        current_time = datetime.now().strftime("%Y-%m-%d")
        system_message = f"""当前时间是 {current_time}。

你是照片搜索查询理解器。你的任务是把用户 query 解析成结构化检索意图 JSON。

输出字段固定为：
1. search_text
2. media_terms
3. identity_terms
4. strict_identity_filter
5. time_hint
6. season
7. time_period

原则：
- 以理解用户真正想找的图像内容为目标，而不是机械抽词。
- search_text 可以做保守的语义归纳、压缩或措辞标准化，但不能编造用户没有表达的实体、动作、场景或属性。
- media_terms 只在你对媒介类型判断清楚时才返回，使用内部规范值：album_cover, poster, stage_performance, screen。
- identity_terms 用于提取明确的人名、艺名、公众人物称呼或稳定别称；不确定时宁可不填。
- strict_identity_filter 表示该查询是否应把身份匹配视为硬约束。只有当用户明确要找某个特定人物本人的照片时才设为 true。
- time_hint 保留原始时间表达；season 和 time_period 做结构化归纳。
- 只返回 JSON，不要解释。
"""

        prompt = f"""输出 JSON，字段固定如下：
{{
  "search_text": "",
  "media_terms": [],
  "identity_terms": [],
  "strict_identity_filter": false,
  "time_hint": null,
  "season": null,
  "time_period": null
}}

抽取规则：
- search_text 用于后续语义检索；可以更像“检索意图表达”，但必须保守。
- 删除礼貌词、任务词和空泛检索词，例如：帮我找、给我看、搜索、检索、照片、图片、相片、截图。
- media_terms 只做内部规范化：
  - 专辑/唱片/封套/专辑封面 -> album_cover
  - 海报/宣传海报 -> poster
  - 演出/演唱会/现场/live -> stage_performance
  - 屏幕/截图 -> screen
- strict_identity_filter:
  - “请帮我找陶喆的照片” -> true
  - “像陶喆风格的舞台照” -> false
  - “陶喆演唱会现场” -> 通常 true
- 不要把“海边”发散成“沙滩海浪蓝天”，也不要把“频谱分析仪 屏幕”发散成“仪器特写 波形曲线 参数读数”。
- 如果 query 同时包含时间和画面内容，search_text 主要保留画面内容；时间相关信息填入其余字段。

示例：
- 用户 query: "去年的照片"
  输出: {{"search_text": "", "media_terms": [], "identity_terms": [], "strict_identity_filter": false, "time_hint": "去年", "season": null, "time_period": null}}
- 用户 query: "夏天海边的照片"
  输出: {{"search_text": "海边", "media_terms": [], "identity_terms": [], "strict_identity_filter": false, "time_hint": null, "season": "夏天", "time_period": null}}
- 用户 query: "频谱分析仪 屏幕"
  输出: {{"search_text": "频谱分析仪 屏幕", "media_terms": ["screen"], "identity_terms": [], "strict_identity_filter": false, "time_hint": null, "season": null, "time_period": null}}
- 用户 query: "去年傍晚的篮球场"
  输出: {{"search_text": "篮球场", "media_terms": [], "identity_terms": [], "strict_identity_filter": false, "time_hint": "去年", "season": null, "time_period": "傍晚"}}
- 用户 query: "周杰伦演唱会"
  输出: {{"search_text": "演唱会现场", "media_terms": ["stage_performance"], "identity_terms": ["周杰伦"], "strict_identity_filter": true, "time_hint": null, "season": null, "time_period": null}}

用户 query: {user_query}"""

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                    timeout=self.timeout,
                    extra_body={"reasoning_effort": self.reasoning_effort},
                )
                
                payload = json.loads(response.choices[0].message.content)
                result = {
                    "search_text": str(payload.get("search_text") or "").strip(),
                    "media_terms": payload.get("media_terms") or [],
                    "identity_terms": payload.get("identity_terms") or [],
                    "strict_identity_filter": bool(payload.get("strict_identity_filter", False)),
                    "time_hint": payload.get("time_hint") or None,
                    "season": payload.get("season") or None,
                    "time_period": payload.get("time_period") or None,
                    "original_query": user_query,
                }

                allowed_media_terms = {"album_cover", "poster", "stage_performance", "screen"}
                result["media_terms"] = [
                    str(value).strip()
                    for value in result["media_terms"]
                    if str(value).strip() in allowed_media_terms
                ]
                result["identity_terms"] = [
                    str(value).strip()
                    for value in result["identity_terms"]
                    if str(value).strip()
                ]

                allowed_seasons = {"春天", "夏天", "秋天", "冬天"}
                if result["season"] not in allowed_seasons:
                    result["season"] = None

                allowed_periods = {"凌晨", "早晨", "上午", "中午", "下午", "傍晚", "夜晚"}
                if result["time_period"] not in allowed_periods:
                    result["time_period"] = None

                if result["time_hint"] is not None:
                    result["time_hint"] = str(result["time_hint"]).strip() or None

                return result
                
            except Exception:
                if attempt == self.max_retries - 1:
                    # 降级：返回原始查询
                    return {
                        "search_text": user_query,
                        "media_terms": [],
                        "identity_terms": [],
                        "strict_identity_filter": False,
                        "time_hint": None,
                        "season": None,
                        "time_period": None,
                        "original_query": user_query,
                    }
                time.sleep(1)
        
        return {
            "search_text": user_query,
            "media_terms": [],
            "identity_terms": [],
            "strict_identity_filter": False,
            "original_query": user_query,
        }

    @staticmethod
    def _normalize_intent_payload(
        payload: Dict[str, Any],
        *,
        user_query: str,
        time_hint: Any = None,
        season: Any = None,
        time_period: Any = None,
    ) -> Dict[str, Any]:
        result = {
            "search_text": str(payload.get("search_text") or "").strip(),
            "media_terms": payload.get("media_terms") or [],
            "identity_terms": payload.get("identity_terms") or [],
            "strict_identity_filter": bool(payload.get("strict_identity_filter", False)),
            "time_hint": time_hint,
            "season": season,
            "time_period": time_period,
            "original_query": user_query,
            "reason": str(payload.get("reason") or "").strip(),
        }

        allowed_media_terms = {"album_cover", "poster", "stage_performance", "screen"}
        result["media_terms"] = [
            str(value).strip()
            for value in result["media_terms"]
            if str(value).strip() in allowed_media_terms
        ]
        result["identity_terms"] = [
            str(value).strip()
            for value in result["identity_terms"]
            if str(value).strip()
        ]

        allowed_seasons = {"春天", "夏天", "秋天", "冬天"}
        if result["season"] not in allowed_seasons:
            result["season"] = None

        allowed_periods = {"凌晨", "早晨", "上午", "中午", "下午", "傍晚", "夜晚"}
        if result["time_period"] not in allowed_periods:
            result["time_period"] = None

        if result["time_hint"] is not None:
            result["time_hint"] = str(result["time_hint"]).strip() or None

        return result

    def expand_query_intents(
        self,
        user_query: str,
        base_intent: Dict[str, Any],
        max_alternatives: int = 2,
    ) -> list[Dict[str, Any]]:
        current_time = datetime.now().strftime("%Y-%m-%d")
        system_message = f"""当前时间是 {current_time}。

你是照片搜索的第二轮查询扩写器。

目标：
- 在第一轮检索偏弱时，给出少量更容易召回的替代检索意图。
- 不是改写用户原意，而是围绕原意做保守补充。
- 不能发散到无关场景，不得虚构人物、物体、地点、动作。

只返回 JSON：
{{
  "alternatives": [
    {{
      "search_text": "",
      "media_terms": [],
      "identity_terms": [],
      "strict_identity_filter": false,
      "reason": ""
    }}
  ]
}}
"""
        prompt = f"""用户原始查询：{user_query}
第一轮意图：{json.dumps(base_intent, ensure_ascii=False)}

请生成不超过 {max_alternatives} 个替代检索意图。

要求：
- 如果原查询已经足够明确，允许返回空数组。
- 每个替代意图必须保持和原查询同一目标，不得偏题。
- 可以补充更容易检索的视觉表达、常见载体或同一意图下的保守语义重述。
- 对特定人物查询，可以补充演出、海报、舞台、大屏等高频载体，但不能凭空添加别的人名。
- strict_identity_filter 只有在替代意图仍然要求“必须是这个人本人”时才为 true。
- 只返回 JSON，不要解释。"""

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                    timeout=self.timeout,
                    extra_body={"reasoning_effort": self.reasoning_effort},
                )
                payload = json.loads(response.choices[0].message.content)
                alternatives = payload.get("alternatives") or []
                normalized: list[Dict[str, Any]] = []
                for item in alternatives[:max_alternatives]:
                    if not isinstance(item, dict):
                        continue
                    normalized.append(
                        self._normalize_intent_payload(
                            item,
                            user_query=user_query,
                            time_hint=base_intent.get("time_hint"),
                            season=base_intent.get("season"),
                            time_period=base_intent.get("time_period"),
                        )
                    )
                return normalized
            except Exception:
                if attempt == self.max_retries - 1:
                    break
                time.sleep(1)
        return []

    def reflect_on_weak_results(
        self,
        user_query: str,
        base_intent: Dict[str, Any],
        weak_results: list[Dict[str, Any]],
    ) -> Dict[str, Any]:
        current_time = datetime.now().strftime("%Y-%m-%d")
        system_message = f"""当前时间是 {current_time}。

你是照片搜索的第三轮反思器。

目标：
- 当第一轮和第二轮的结果都偏弱时，分析“为什么没搜准”，并给出一个更稳健的单一改进意图。
- 反思必须围绕原始用户目标，不得偏题，不得引入用户未表达的新人物或场景。
- 允许你在“更宽”或“更窄”之间做一次保守调整，但只能做一轮。

只返回 JSON：
{{
  "search_text": "",
  "media_terms": [],
  "identity_terms": [],
  "strict_identity_filter": false,
  "reason": ""
}}
"""

        weak_result_summaries = [
            {
                "description": str(item.get("description") or "").strip(),
                "score": float(item.get("score", 0.0)),
                "match_summary": item.get("match_summary") or {},
            }
            for item in weak_results[:5]
        ]

        prompt = f"""用户原始查询：{user_query}
第一轮基础意图：{json.dumps(base_intent, ensure_ascii=False)}
当前弱结果摘要：{json.dumps(weak_result_summaries, ensure_ascii=False)}

要求：
- 如果当前结果已经足够接近，允许返回空 JSON {{}}。
- 如果原查询是明确的人物检索，只有当你仍然认为必须找该人物本人时，strict_identity_filter 才为 true。
- 你可以决定“更强调媒介类型”、“更强调人物近景”、“去掉过强约束”、“把抽象描述改成更容易检索的视觉表达”。
- reason 必须简短说明你为何这样调整。
- 只返回 JSON，不要解释。"""

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                    timeout=self.timeout,
                    extra_body={"reasoning_effort": self.reasoning_effort},
                )
                payload = json.loads(response.choices[0].message.content)
                if not isinstance(payload, dict) or not payload:
                    return {}
                return self._normalize_intent_payload(
                    payload,
                    user_query=user_query,
                    time_hint=base_intent.get("time_hint"),
                    season=base_intent.get("season"),
                    time_period=base_intent.get("time_period"),
                )
            except Exception:
                if attempt == self.max_retries - 1:
                    break
                time.sleep(1)
        return {}
    
    def is_enabled(self) -> bool:
        """检查服务是否可用。"""
        return bool(self.api_key and self.model_name)
