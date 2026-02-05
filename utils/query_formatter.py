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
                "search_text": "公园草地树木 户外休闲场景 绿意盎然",
                "scene": "公园",
                "time_hint": None,
                "season": None,
                "time_period": None,
                "original_query": "请展示一张公园的照片"
            }
        """
        # 注入当前时间，让模型能正确解析"去年"、"上个月"等相对时间
        current_time = datetime.now().strftime("%Y年%m月%d日")
        system_message = f"当前时间：{current_time}。你是照片检索助手。"
        
        prompt = f"""解析照片搜索意图，输出JSON：
{{
    "search_text": "纯视觉描述（禁止时间词）",
    "scene": "场景关键词",
    "time_hint": "时间提示或null",
    "season": "春天/夏天/秋天/冬天或null",
    "time_period": "凌晨/早晨/上午/中午/下午/傍晚/夜晚或null"
}}

规则：
- search_text只含视觉内容，禁止年月日季节时段
- time_hint保留原始表达如"去年夏天"、"2023年"
- 无时间信息时对应字段为null

用户查询：{user_query}"""

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
                )
                
                result = json.loads(response.choices[0].message.content)
                result["original_query"] = user_query
                
                # 架构改进：search_text 保持纯语义描述，不拼接时间信息
                # 时间信息（season, time_period, time_hint）作为独立字段返回
                # Searcher 会根据这些独立字段构建 ES 过滤条件
                
                # 空值防护：确保 search_text 不为空
                # 如果 LLM 返回空的 search_text（纯过滤查询场景），回退到原始查询
                # Searcher 会进一步判断是否为纯过滤查询并走相应分支
                search_text = result.get("search_text", "")
                if not search_text or not search_text.strip():
                    search_text = user_query
                result["search_text"] = search_text
                
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
