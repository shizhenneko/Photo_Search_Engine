from __future__ import annotations

import time
import urllib.parse
from abc import ABC, abstractmethod
from typing import List, Optional

from openai import OpenAI

from utils.image_parser import get_image_dimensions


class VisionLLMService(ABC):
    """
    Vision LLM服务抽象接口。
    """

    @abstractmethod
    def generate_description(self, image_path: str) -> str:
        """
        生成图片描述（中文，50-200字）。

        Args:
            image_path (str): 图片路径

        Returns:
            str: 描述文本
        """

    @abstractmethod
    def generate_description_batch(self, image_paths: List[str]) -> List[str]:
        """
        批量生成描述。

        Args:
            image_paths (List[str]): 图片路径列表

        Returns:
            List[str]: 描述列表
        """


class OpenRouterVisionLLMService(VisionLLMService):
    """
    OpenRouter Vision LLM服务实现。
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "openai/gpt-4o",
        base_url: str = "https://openrouter.ai/api/v1",
        server_host: str = "localhost",
        server_port: int = 5000,
        timeout: int = 30,
        max_retries: int = 3,
        client: Optional[OpenAI] = None,
    ) -> None:
        """
        初始化OpenRouter Vision服务。

        Args:
            api_key (str): OpenRouter API密钥
            model_name (str): 模型名称
            base_url (str): OpenRouter API地址
            server_host (str): 本地服务host
            server_port (int): 本地服务port
            timeout (int): API超时时间（秒）
            max_retries (int): 最大重试次数
            client (Optional[OpenAI]): OpenAI客户端实例
        """
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY 未设置")
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.server_host = server_host
        self.server_port = server_port
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = client or OpenAI(api_key=api_key, base_url=base_url)

    def _get_image_url(self, image_path: str) -> str:
        """
        生成本地HTTP图片URL。

        Args:
            image_path (str): 图片路径

        Returns:
            str: 图片URL
        """
        encoded_path = urllib.parse.quote(image_path)
        return f"http://{self.server_host}:{self.server_port}/photo?path={encoded_path}"  # 生成图片URL供Vision LLM访问

    def generate_description(self, image_path: str) -> str:
        """
        生成图片描述（中文）。

        Args:
            image_path (str): 图片路径

        Returns:
            str: 描述文本
        """
        prompt = (
            "请用中文描述这张图片，包含以下要素：\n"
            "1. 场景描述（室内/室外，具体地点）\n"
            "2. 主要主体（人物、物体、动物等）\n"
            "3. 动作或状态（在做什么）\n"
            "4. 环境细节（光线、天气、背景）\n"
            "5. 情绪氛围（欢乐、温馨、宁静等）\n\n"
            "描述长度：50-200字"
        )
        image_url = self._get_image_url(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.2,
                    timeout=self.timeout,
                )
                content = response.choices[0].message.content.strip()
                if not content:
                    raise ValueError("Vision LLM返回空描述")
                return content
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise ValueError(f"生成描述失败: {exc}") from exc  # 捕获API调用失败，避免中断整体流程
                time.sleep(1)

        raise ValueError("生成描述失败")

    def generate_description_batch(self, image_paths: List[str]) -> List[str]:
        """
        批量生成描述。

        Args:
            image_paths (List[str]): 图片路径列表

        Returns:
            List[str]: 描述列表
        """
        descriptions: List[str] = []
        for image_path in image_paths:
            descriptions.append(self.generate_description(image_path))
        return descriptions


class LocalVisionLLMService(VisionLLMService):
    """
    本地Vision服务（降级或离线场景使用）。
    """

    def generate_description(self, image_path: str) -> str:
        """
        生成本地描述（基于图片尺寸）。

        Args:
            image_path (str): 图片路径

        Returns:
            str: 描述文本
        """
        width, height = get_image_dimensions(image_path)
        if width == 0 or height == 0:
            return "一张本地生成的图片描述"
        return f"一张本地生成的图片描述，分辨率为{width}x{height}"

    def generate_description_batch(self, image_paths: List[str]) -> List[str]:
        """
        批量生成本地描述。

        Args:
            image_paths (List[str]): 图片路径列表

        Returns:
            List[str]: 描述列表
        """
        return [self.generate_description(path) for path in image_paths]


class OpenAIVisionLLMService(OpenRouterVisionLLMService):
    """
    OpenAI命名兼容类（行为与OpenRouter实现一致）。
    """
