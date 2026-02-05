from __future__ import annotations

import json
import time
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
        prompt = f"""你是照片检索助手，负责解析用户的搜索意图。

## 用户查询
{user_query}

## 输出格式（JSON）
{{
    "scene": "场景关键词",
    "objects": ["物体1", "物体2"],
    "atmosphere": "氛围描述",
    "time_hint": "时间提示或null",
    "season": "季节或null",
    "time_period": "时段或null",
    "search_text": "纯视觉语义描述"
}}

## 字段说明

| 字段 | 用途 | 规则 |
|------|------|------|
| search_text | 生成embedding向量 | NEVER包含时间词汇，只描述视觉内容 |
| scene | 场景分类 | 简洁关键词：公园、海滩、山顶、街道、客厅等 |
| objects | 物体识别 | 列出1-5个可能出现的主要物体 |
| atmosphere | 氛围判断 | 宁静、热闹、浪漫、庄重等 |
| time_hint | ES时间过滤 | 格式如"2020年11月"、"去年"，无则null |
| season | ES季节过滤 | 春天/夏天/秋天/冬天，无则null |
| time_period | ES时段过滤 | 凌晨/早晨/上午/中午/下午/傍晚/夜晚，无则null |

## 处理流程

1. **过滤噪音**：忽略礼貌用语（请、帮我、展示）
2. **提取时间**：将时间信息提取到独立字段，从查询中移除
3. **识别场景**：判断用户想找什么类型的照片
4. **生成search_text**：组合场景+物体+氛围，形成自然语言描述

## 示例

**输入**："请帮我找一下去年夏天在海边拍的全家福"
**输出**：
{{
    "scene": "海滩",
    "objects": ["人物", "沙滩", "海浪", "家庭"],
    "atmosphere": "温馨",
    "time_hint": "去年",
    "season": "夏天",
    "time_period": null,
    "search_text": "海滩沙滩边的家庭合影 温馨的户外场景 多人站在一起"
}}

**输入**："早上拍的日出"
**输出**：
{{
    "scene": "户外",
    "objects": ["太阳", "天空", "云彩", "地平线"],
    "atmosphere": "宁静",
    "time_hint": null,
    "season": null,
    "time_period": "早晨",
    "search_text": "日出时分的天空 太阳从地平线升起 橙红色的云彩"
}}

## NEVER（绝对禁止）
- search_text中NEVER出现时间词汇（年、月、日、季节、时段）
- NEVER在无时间信息时编造时间字段

请分析用户查询并输出JSON："""

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
