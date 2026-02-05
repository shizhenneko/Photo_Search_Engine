"""
Rerank服务模块，使用Vision LLM对候选图片进行二次精排。

通过将多张候选图片与用户查询一起发送给Vision LLM，
让模型根据语义匹配度对图片进行重新排序。
"""

from __future__ import annotations

import base64
import json
import re
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from openai import OpenAI

from utils.image_parser import resize_and_optimize_image

if TYPE_CHECKING:
    from utils.vision_llm_service import VisionLLMService


class RerankService:
    """
    Vision LLM Rerank服务。

    对候选图片进行二次精排，使用Vision LLM根据用户查询
    对图片进行语义匹配度排序。
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str,
        timeout: int = 60,
        max_retries: int = 3,
        image_max_size: int = 512,
        image_quality: int = 75,
        image_format: str = "WEBP",
        max_images: int = 10,
        client: Optional[OpenAI] = None,
    ) -> None:
        """
        初始化Rerank服务。

        Args:
            api_key: OpenRouter/OpenAI API密钥
            model_name: Vision模型名称
            base_url: API基础URL
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            image_max_size: 图片最大边长（像素），默认512
            image_quality: 图片压缩质量（1-100），默认75
            image_format: 图片输出格式，默认WEBP
            max_images: 最大处理图片数量，默认10
            client: 可选的OpenAI客户端实例
        """
        if not api_key:
            raise ValueError("API密钥未设置")
        if not model_name:
            raise ValueError("模型名称未设置")
        if not base_url:
            raise ValueError("API基础URL未设置")

        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.image_max_size = max(256, min(2048, image_max_size))
        self.image_quality = max(1, min(100, image_quality))
        self.image_format = image_format.upper() if image_format.upper() in ["JPEG", "WEBP", "PNG"] else "WEBP"
        self.max_images = max(1, min(20, max_images))
        self.client = client or OpenAI(api_key=api_key, base_url=base_url)

    def _get_image_base64(self, image_path: str) -> str:
        """
        读取并优化图片，返回Base64编码的data URL。

        Args:
            image_path: 图片文件路径

        Returns:
            str: data:image/...;base64,... 格式的URL

        Raises:
            ValueError: 图片编码失败
        """
        try:
            image_bytes = resize_and_optimize_image(
                image_path,
                max_size=self.image_max_size,
                quality=self.image_quality,
                format=self.image_format,
            )
            base64_str = base64.b64encode(image_bytes).decode("utf-8")

            mime_type = {
                "JPEG": "image/jpeg",
                "PNG": "image/png",
                "WEBP": "image/webp",
            }.get(self.image_format, "image/jpeg")

            return f"data:{mime_type};base64,{base64_str}"
        except Exception as e:
            raise ValueError(f"图片编码失败: {image_path}, 错误: {e}")

    def _build_rerank_prompt(self, query: str, num_images: int) -> str:
        """
        构建Rerank的提示词。

        Args:
            query: 用户搜索查询
            num_images: 候选图片数量

        Returns:
            str: 提示词文本
        """
        return f"""用户正在搜索照片："{query}"

你将看到 {num_images} 张候选图片（编号 1 到 {num_images}）。
请根据每张图片与用户搜索意图的匹配程度，从最相关到最不相关排序。

要求：
1. 仔细分析每张图片的内容
2. 理解用户搜索意图的核心需求
3. 按相关性从高到低排序所有图片

只返回JSON格式，不要有其他文字：
{{"ranking": [3, 1, 5, 2, 4]}}

注意：ranking数组中的数字是图片编号，按相关性降序排列。"""

    def _parse_ranking_response(self, response_text: str, num_images: int) -> List[int]:
        """
        解析LLM返回的排序结果。

        Args:
            response_text: LLM的响应文本
            num_images: 候选图片数量

        Returns:
            List[int]: 排序后的图片编号列表（0-indexed）

        Raises:
            ValueError: 解析失败
        """
        # 尝试直接解析JSON
        try:
            # 清理可能的markdown代码块标记
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                # 移除markdown代码块
                cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
                cleaned = cleaned.rstrip("`").strip()

            data = json.loads(cleaned)
            if isinstance(data, dict) and "ranking" in data:
                ranking = data["ranking"]
                if isinstance(ranking, list):
                    # 验证排序结果
                    ranking = [int(x) for x in ranking]
                    # 确保所有编号都有效（1-indexed）
                    valid_ranking = [x for x in ranking if 1 <= x <= num_images]
                    if valid_ranking:
                        # 转换为0-indexed
                        return [x - 1 for x in valid_ranking]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # 尝试从文本中提取数字列表
        numbers = re.findall(r"\d+", response_text)
        if numbers:
            ranking = [int(x) for x in numbers]
            valid_ranking = [x for x in ranking if 1 <= x <= num_images]
            if valid_ranking:
                return [x - 1 for x in valid_ranking]

        raise ValueError(f"无法解析排序结果: {response_text[:200]}")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        rerank_top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        对候选图片进行Vision LLM重排序。

        Args:
            query: 用户搜索查询
            candidates: 候选图片列表，每个元素包含photo_path等字段
            rerank_top_k: 重排后保留的图片数量

        Returns:
            List[Dict[str, Any]]: 重排序后的图片列表
        """
        if not candidates:
            return []

        if not query or not query.strip():
            return candidates[:rerank_top_k]

        # 限制处理的图片数量
        candidates_to_process = candidates[: self.max_images]
        num_images = len(candidates_to_process)

        if num_images <= 1:
            return candidates[:rerank_top_k]

        # 构建多图消息
        try:
            content: List[Dict[str, Any]] = []

            # 添加文本提示
            prompt = self._build_rerank_prompt(query, num_images)
            content.append({"type": "text", "text": prompt})

            # 添加每张图片（带编号说明）
            for idx, candidate in enumerate(candidates_to_process, start=1):
                photo_path = candidate.get("photo_path", "")
                if not photo_path:
                    continue

                try:
                    image_url = self._get_image_base64(photo_path)
                    # 添加图片编号说明
                    content.append({
                        "type": "text",
                        "text": f"图片 {idx}:"
                    })
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    })
                except Exception as e:
                    print(f"[RERANK] 图片编码失败: {photo_path}, 错误: {e}")
                    # 如果图片编码失败，添加占位说明
                    content.append({
                        "type": "text",
                        "text": f"图片 {idx}: [图片加载失败]"
                    })

            messages = [{"role": "user", "content": content}]

            # 调用Vision LLM
            for attempt in range(self.max_retries):
                try:
                    print(f"[RERANK] Vision LLM调用 (第{attempt + 1}/{self.max_retries}次), "
                          f"模型: {self.model_name}, 图片数: {num_images}")

                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.1,  # 低温度以获得更确定性的结果
                        timeout=self.timeout,
                    )

                    response_text = response.choices[0].message.content.strip()
                    print(f"[RERANK] LLM响应: {response_text[:200]}")

                    # 解析排序结果
                    ranking = self._parse_ranking_response(response_text, num_images)
                    print(f"[RERANK] 解析后的排序: {ranking}")

                    # 根据排序重新组织结果
                    reranked_results = []
                    seen_indices = set()

                    for rank, idx in enumerate(ranking, start=1):
                        if idx < len(candidates_to_process) and idx not in seen_indices:
                            result = candidates_to_process[idx].copy()
                            result["original_rank"] = result.get("rank", idx + 1)
                            result["rank"] = rank
                            result["reranked"] = True
                            reranked_results.append(result)
                            seen_indices.add(idx)

                    # 如果解析结果不完整，补充未被包含的图片
                    for idx, candidate in enumerate(candidates_to_process):
                        if idx not in seen_indices:
                            result = candidate.copy()
                            result["original_rank"] = result.get("rank", idx + 1)
                            result["rank"] = len(reranked_results) + 1
                            result["reranked"] = True
                            reranked_results.append(result)

                    # 返回指定数量的结果
                    final_results = reranked_results[:rerank_top_k]

                    # 重新分配rank
                    for rank, item in enumerate(final_results, start=1):
                        item["rank"] = rank

                    print(f"[RERANK] 重排完成，返回 {len(final_results)} 个结果")
                    return final_results

                except Exception as e:
                    print(f"[RERANK] LLM调用失败 (第{attempt + 1}/{self.max_retries}次): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(1)
                    continue

            # 所有重试都失败，降级返回原始结果
            print("[RERANK] 所有重试都失败，返回原始排序结果")
            return candidates[:rerank_top_k]

        except Exception as e:
            print(f"[RERANK] Rerank过程异常: {e}，返回原始排序结果")
            return candidates[:rerank_top_k]

    def is_enabled(self) -> bool:
        """
        检查服务是否可用。

        Returns:
            bool: 服务是否可用
        """
        return bool(self.api_key)
