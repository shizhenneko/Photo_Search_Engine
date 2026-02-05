from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from openai import OpenAI


class QueryFormatter:
    """
    使用 LLM 对用户查询进行格式化，提取核心检索需求。
    
    输出格式示例：
    "这张图片展示了一个洞穴内部的场景... | 文件名: IMG | 2020年11月 | 季节: 秋天 | 时段: 白天"
    
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
                "search_text": "公园 草地 树木 户外 休闲 | 时段: 白天",
                "scene": "公园",
                "time_hint": None,
                "season": None,
                "time_period": "白天",
                "original_query": "请展示一张公园的照片"
            }
        """
        prompt = f"""你是一个照片检索助手。用户想要搜索照片，请分析用户的查询意图，提取核心需求。

用户查询：{user_query}

请返回 JSON 格式：
{{
    "scene": "场景描述（如：公园、海滩、山顶、城市街道等）",
    "objects": ["主要物体列表"],
    "atmosphere": "氛围描述（如：宁静、热闹、浪漫等）",
    "time_hint": "时间提示（如：2020年11月、去年夏天）或 null",
    "season": "季节（春天/夏天/秋天/冬天）或 null",
    "time_period": "时段（凌晨/早晨/上午/中午/下午/傍晚/夜晚）或 null",
    "search_text": "生成一段适合向量检索的纯视觉描述，包含场景、物体、氛围的自然语言，不包含时间信息"
}}

规则：
1. 从用户查询中提取核心检索需求，忽略无关的礼貌用语（如"请"、"展示"）
2. search_text 是纯视觉语义描述，不包含时间、日期、季节、时段等信息
3. 时间相关信息单独提取到 time_hint、season、time_period 字段
4. 如果用户未提及时间，time_hint、season、time_period 返回 null
5. objects 列出可能出现在照片中的主要物体
6. time_period 必须是以下7档之一：凌晨/早晨/上午/中午/下午/傍晚/夜晚"""

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
                result["original_query"] = user_query
                
                # 架构改进：search_text 保持纯语义描述，不拼接时间信息
                # 时间信息（season, time_period, time_hint）作为独立字段返回
                # Searcher 会根据这些独立字段构建 ES 过滤条件
                result["search_text"] = result.get("search_text", user_query)
                
                return result
                
            except Exception:
                if attempt == self.max_retries - 1:
                    # 降级：返回原始查询
                    return {
                        "search_text": user_query,
                        "scene": None,
                        "time_hint": None,
                        "season": None,
                        "time_period": None,
                        "original_query": user_query,
                    }
                time.sleep(1)
        
        return {"search_text": user_query, "original_query": user_query}
    
    def is_enabled(self) -> bool:
        """检查服务是否可用。"""
        return bool(self.api_key and self.model_name)
