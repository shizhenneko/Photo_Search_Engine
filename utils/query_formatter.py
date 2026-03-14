from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Dict, Optional

from openai import OpenAI

from utils.llm_compat import (
    create_chat_completion,
    extract_response_text,
    normalize_openai_base_url,
    requires_api_key,
    resolve_api_key,
)

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
        if requires_api_key(base_url) and not api_key:
            raise ValueError("QUERY_FORMAT_API_KEY 未设置")

        resolved_api_key = resolve_api_key(api_key, base_url)
        self.api_key = resolved_api_key
        self.model_name = model_name
        self.base_url = normalize_openai_base_url(base_url)
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = client or OpenAI(api_key=resolved_api_key, base_url=self.base_url)
    
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
2. retrieval_mode
3. media_terms
4. identity_terms
5. strict_identity_filter
6. intent_mode
7. intent_contract
8. time_hint
9. season
10. time_period

原则：
- 以理解用户真正想找的图像内容为目标，而不是机械抽词。
- search_text 可以做保守的语义归纳、压缩或措辞标准化，但不能编造用户没有表达的实体、动作、场景或属性。
- search_text 应优先表达“正确结果里应该能直接看到什么”，而不是只重复名字、标题、口号或其他非视觉标签。
- 如果 query 含有命名实体、身份称呼、稳定别称或抽象标签，且用户本质上在找某种可见主体/场景/构图，search_text 应尽量转成保守的可见表达；精确名字或称呼保留到 identity_terms。
- 如果 query 主要是时间、季节、时段、设备、文件属性或其他过滤条件，且缺少稳定的可见目标，search_text 可以为空。
- retrieval_mode 默认是 hybrid。只有当用户主要在表达时间、季节、时段、设备等过滤条件，且几乎没有稳定的可见主体需要语义召回时，才使用 filter_only。
- media_terms 是 LLM 认为对检索有帮助的媒介或载体词，可为空；不要为了迎合固定词表而改写用户意图。
- identity_terms 用于提取明确的名字、称呼、编号、系列名或其他稳定命名约束；不确定时宁可不填。
- media_terms、identity_terms、intent_contract 都是检索提示，不是硬过滤条件；除 retrieval_mode=filter_only 外，不要把它们理解成“只能返回字段命中的图片”。
- strict_identity_filter 表示该查询是否应把身份匹配视为硬约束。只有当用户明确表达“必须是本人/只要这个人/不要其他人/不能错人”这类强约束时才设为 true。
- 不要因为 query 里出现了命名对象，就自动把 strict_identity_filter 设为 true；只有当“不能错对象”是明确要求时才这样做。
- intent_mode 只能是 strict 或 open。
- 当用户在找明确且不可替换的目标时，intent_mode 应为 strict。这里的“明确目标”不只包括特定人物，也包括特定物体、特定载体、特定内容组合或明确不可替换的检索对象。
- intent_contract 用来表达这个 query 的最小不可丢失目标，格式固定为：
  {{
    "core_target": "",
    "must_keep": [],
    "avoid_drift": ""
  }}
- strict 模式下，后续任何扩写或反思都只能围绕 intent_contract 做保守重述，不能把目标替换成更泛的同类概念。
- time_hint 保留原始时间表达；season 和 time_period 做结构化归纳。
- 不要依赖具体示例去套用模式，应直接理解当前 query。
- 只返回 JSON，不要解释。
"""

        prompt = f"""输出 JSON，字段固定如下：
{{
  "search_text": "",
  "retrieval_mode": "hybrid",
  "media_terms": [],
  "identity_terms": [],
  "strict_identity_filter": false,
  "intent_mode": "open",
  "intent_contract": {{
    "core_target": "",
    "must_keep": [],
    "avoid_drift": ""
  }},
  "time_hint": null,
  "season": null,
  "time_period": null
}}

抽取规则：
- search_text 用于后续语义检索；可以更像“检索意图表达”，但必须保守。
- retrieval_mode 默认填 hybrid；只有在 query 明显是纯过滤需求时才填 filter_only。
- 删除礼貌词、任务词和空泛检索词，例如：帮我找、给我看、搜索、检索、照片、图片、相片、截图。
- 如果 query 中有名字、称呼、标题或抽象标签，不要让 search_text 退化成只重复这些词；优先保留能直接看到的主体、场景、动作、构图或载体特征。
- 如果 query 缺少稳定的视觉目标，且本质上只是过滤条件，search_text 允许为空。
- media_terms 若返回，应使用对检索有帮助的简短载体词或媒介词，不要求来自固定集合。
- media_terms、identity_terms 是帮助召回和后续排序的提示，不是要求结果必须在某个字段里命中。
- strict_identity_filter 只在用户明确要求“必须是这个特定人物本人，且不能接受其他人或仅文字提及”时设为 true。
- intent_mode 若 query 存在明确不可替换的核心目标，则设为 strict，否则设为 open。
- intent_contract:
  - core_target: 用一句短语概括用户真正要找的东西
  - must_keep: 只保留你确信属于不可丢失目标的词；不确定时宁可留空
  - avoid_drift: 简短说明后续不该漂移到什么方向
- 不要把明确目标替换成更泛的同类概念。
- 如果 query 同时包含时间和画面内容，search_text 主要保留画面内容；时间相关信息填入其余字段。

用户 query: {user_query}"""

        for attempt in range(self.max_retries):
            try:
                response = create_chat_completion(
                    self.client,
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                    timeout=self.timeout,
                    reasoning_effort=self.reasoning_effort,
                )

                payload = json.loads(extract_response_text(response))
                result = {
                    "search_text": str(payload.get("search_text") or "").strip(),
                    "retrieval_mode": payload.get("retrieval_mode"),
                    "media_terms": payload.get("media_terms") or [],
                    "identity_terms": payload.get("identity_terms") or [],
                    "strict_identity_filter": bool(payload.get("strict_identity_filter", False)),
                    "intent_mode": payload.get("intent_mode"),
                    "intent_contract": payload.get("intent_contract"),
                    "contract_satisfied": True,
                    "time_hint": payload.get("time_hint") or None,
                    "season": payload.get("season") or None,
                    "time_period": payload.get("time_period") or None,
                    "original_query": user_query,
                }

                result["media_terms"] = [
                    str(value).strip()
                    for value in result["media_terms"]
                    if str(value).strip()
                ]
                result["identity_terms"] = [
                    str(value).strip()
                    for value in result["identity_terms"]
                    if str(value).strip()
                ]
                result["retrieval_mode"] = self._normalize_retrieval_mode(
                    result.get("retrieval_mode"),
                    search_text=result["search_text"],
                    time_hint=result["time_hint"],
                    season=result["season"],
                    time_period=result["time_period"],
                )
                result["intent_mode"] = self._normalize_intent_mode(
                    result.get("intent_mode"),
                    strict_identity_filter=result["strict_identity_filter"],
                )
                result["intent_contract"] = self._normalize_intent_contract(
                    result.get("intent_contract"),
                    user_query=user_query,
                    search_text=result["search_text"],
                    media_terms=result["media_terms"],
                    identity_terms=result["identity_terms"],
                    intent_mode=result["intent_mode"],
                )

                if result["season"] is not None:
                    result["season"] = str(result["season"]).strip() or None

                if result["time_period"] is not None:
                    result["time_period"] = str(result["time_period"]).strip() or None

                if result["time_hint"] is not None:
                    result["time_hint"] = str(result["time_hint"]).strip() or None

                return result
                
            except Exception:
                if attempt == self.max_retries - 1:
                    # 降级：返回原始查询
                    return {
                        "search_text": user_query,
                        "retrieval_mode": "hybrid",
                        "media_terms": [],
                        "identity_terms": [],
                        "strict_identity_filter": False,
                        "intent_mode": "open",
                        "intent_contract": self._normalize_intent_contract(
                            {},
                            user_query=user_query,
                            search_text=user_query,
                            media_terms=[],
                            identity_terms=[],
                            intent_mode="open",
                        ),
                        "contract_satisfied": True,
                        "time_hint": None,
                        "season": None,
                        "time_period": None,
                        "original_query": user_query,
                    }
                time.sleep(1)
        
        return {
            "search_text": user_query,
            "retrieval_mode": "hybrid",
            "media_terms": [],
            "identity_terms": [],
            "strict_identity_filter": False,
            "intent_mode": "open",
            "intent_contract": self._normalize_intent_contract(
                {},
                user_query=user_query,
                search_text=user_query,
                media_terms=[],
                identity_terms=[],
                intent_mode="open",
            ),
            "contract_satisfied": True,
            "original_query": user_query,
        }

    @staticmethod
    def _normalize_intent_mode(value: Any, *, strict_identity_filter: bool) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in {"strict", "open"}:
            return normalized
        return "strict" if strict_identity_filter else "open"

    @staticmethod
    def _normalize_retrieval_mode(
        value: Any,
        *,
        search_text: str,
        time_hint: Any = None,
        season: Any = None,
        time_period: Any = None,
    ) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in {"hybrid", "filter_only"}:
            return normalized
        has_filter = bool(time_hint or season or time_period)
        if has_filter and not str(search_text or "").strip():
            return "filter_only"
        return "hybrid"

    @staticmethod
    def _normalize_intent_contract(
        value: Any,
        *,
        user_query: str,
        search_text: str,
        media_terms: list[str],
        identity_terms: list[str],
        intent_mode: str,
        base_contract: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        contract = value if isinstance(value, dict) else {}
        inherited = base_contract if isinstance(base_contract, dict) else {}

        core_target = str(contract.get("core_target") or inherited.get("core_target") or "").strip()
        if not core_target:
            core_target = (
                search_text.strip()
                or " ".join(identity_terms).strip()
                or " ".join(media_terms).strip()
                or user_query.strip()
            )

        raw_must_keep = contract.get("must_keep")
        if not isinstance(raw_must_keep, list):
            raw_must_keep = inherited.get("must_keep") if isinstance(inherited.get("must_keep"), list) else []
        must_keep = []
        seen = set()
        for item in raw_must_keep or []:
            normalized = str(item or "").strip()
            if not normalized:
                continue
            lowered = normalized.lower()
            if lowered in seen:
                continue
            must_keep.append(normalized)
            seen.add(lowered)

        avoid_drift = str(contract.get("avoid_drift") or inherited.get("avoid_drift") or "").strip()
        return {
            "core_target": core_target,
            "must_keep": must_keep,
            "avoid_drift": avoid_drift,
        }

    @staticmethod
    def _normalize_intent_payload(
        payload: Dict[str, Any],
        *,
        user_query: str,
        time_hint: Any = None,
        season: Any = None,
        time_period: Any = None,
        base_intent: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result = {
            "search_text": str(payload.get("search_text") or "").strip(),
            "retrieval_mode": payload.get("retrieval_mode"),
            "media_terms": payload.get("media_terms") or [],
            "identity_terms": payload.get("identity_terms") or [],
            "strict_identity_filter": bool(payload.get("strict_identity_filter", False)),
            "intent_mode": payload.get("intent_mode"),
            "intent_contract": payload.get("intent_contract"),
            "contract_satisfied": bool(payload.get("contract_satisfied", True)),
            "time_hint": time_hint,
            "season": season,
            "time_period": time_period,
            "original_query": user_query,
            "reason": str(payload.get("reason") or "").strip(),
        }

        result["media_terms"] = [
            str(value).strip()
            for value in result["media_terms"]
            if str(value).strip()
        ]
        result["identity_terms"] = [
            str(value).strip()
            for value in result["identity_terms"]
            if str(value).strip()
        ]
        result["retrieval_mode"] = QueryFormatter._normalize_retrieval_mode(
            result.get("retrieval_mode"),
            search_text=result["search_text"],
            time_hint=result["time_hint"],
            season=result["season"],
            time_period=result["time_period"],
        )
        base_contract = {}
        if isinstance(base_intent, dict):
            maybe_contract = base_intent.get("intent_contract")
            if isinstance(maybe_contract, dict):
                base_contract = maybe_contract
        result["intent_mode"] = QueryFormatter._normalize_intent_mode(
            result.get("intent_mode"),
            strict_identity_filter=result["strict_identity_filter"],
        )
        result["intent_contract"] = QueryFormatter._normalize_intent_contract(
            result.get("intent_contract"),
            user_query=user_query,
            search_text=result["search_text"],
            media_terms=result["media_terms"],
            identity_terms=result["identity_terms"],
            intent_mode=result["intent_mode"],
            base_contract=base_contract,
        )

        if result["season"] is not None:
            result["season"] = str(result["season"]).strip() or None

        if result["time_period"] is not None:
            result["time_period"] = str(result["time_period"]).strip() or None

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
      "retrieval_mode": "hybrid",
      "media_terms": [],
      "identity_terms": [],
      "strict_identity_filter": false,
      "intent_mode": "open",
      "intent_contract": {{
        "core_target": "",
        "must_keep": [],
        "avoid_drift": ""
      }},
      "contract_satisfied": true,
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
- 如果原查询的核心约束是名字、称呼、编号、标题或其他命名标签，且直接重复这些词不利于视觉召回，可以把 search_text 改写成更容易看见的主体、场景、动作、构图或载体表达，但不能改变目标。
- retrieval_mode 通常保持 hybrid；只有当原查询本身就是明确纯过滤需求时，才允许使用 filter_only。
- 你必须显式判断替代意图是否仍然遵守第一轮意图中的 intent_contract；如果遵守，contract_satisfied=true，否则为 false。
- 如果第一轮 intent_mode 是 strict，那么替代意图必须保留 core_target 和 must_keep，不允许把目标扩成更泛的同类对象、同类人物或同类场景。
- strict_identity_filter 只有在替代意图仍然要求“必须是这个人本人，且不能接受错人或仅文字提及”时才为 true。
- 不允许把“明确的人/物/载体/内容组合”改写成泛化类别词。
- 只返回 JSON，不要解释。"""

        for attempt in range(self.max_retries):
            try:
                response = create_chat_completion(
                    self.client,
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                    timeout=self.timeout,
                    reasoning_effort=self.reasoning_effort,
                )
                payload = json.loads(extract_response_text(response))
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
                            base_intent=base_intent,
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
  "retrieval_mode": "hybrid",
  "media_terms": [],
  "identity_terms": [],
  "strict_identity_filter": false,
  "intent_mode": "open",
  "intent_contract": {{
    "core_target": "",
    "must_keep": [],
    "avoid_drift": ""
  }},
  "contract_satisfied": true,
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
- 反思的目标是“在不改原始目标的前提下修正检索表达”，不是换目标。
- 如果原查询主要依赖名字、称呼、标题、编号或其他命名标签，而当前结果又偏弱，优先考虑把名字主导的表达改成更直接可见的主体、场景、动作、构图或载体表达。
- retrieval_mode 默认保持 hybrid；只有在原查询就是纯过滤需求时，才改成 filter_only。
- 你必须显式判断新的意图是否仍然遵守第一轮 intent_contract；如果遵守，contract_satisfied=true，否则为 false。
- 只有当你仍然判断“不能错对象”是刚性要求时，strict_identity_filter 才为 true。
- 如果第一轮 intent_mode 是 strict，那么新的意图必须保留 core_target 和 must_keep，只能做更稳健的重述或收紧，不能改成泛化替代品。
- 你可以决定“更强调媒介类型”、“更强调主体近景”、“把抽象描述改成更容易检索的视觉表达”，但不能改变原始目标。
- reason 必须简短说明你为何这样调整。
- 只返回 JSON，不要解释。"""

        for attempt in range(self.max_retries):
            try:
                response = create_chat_completion(
                    self.client,
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                    timeout=self.timeout,
                    reasoning_effort=self.reasoning_effort,
                )
                payload = json.loads(extract_response_text(response))
                if not isinstance(payload, dict) or not payload:
                    return {}
                return self._normalize_intent_payload(
                    payload,
                    user_query=user_query,
                    time_hint=base_intent.get("time_hint"),
                    season=base_intent.get("season"),
                    time_period=base_intent.get("time_period"),
                    base_intent=base_intent,
                )
            except Exception:
                if attempt == self.max_retries - 1:
                    break
                time.sleep(1)
        return {}
    
    def is_enabled(self) -> bool:
        """检查服务是否可用。"""
        return bool(self.api_key and self.model_name)
