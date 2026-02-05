from __future__ import annotations

import base64
import time
import urllib.parse
from abc import ABC, abstractmethod
from typing import List, Optional

from openai import OpenAI

from utils.image_parser import get_image_dimensions, resize_and_optimize_image


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

    默认使用 Base64 编码方式，因为 OpenRouter 远程服务器无法访问本地 localhost。

    成本优化策略：
    1. 图片自动缩放至 max_size（默认 1024 像素）
    2. JPEG/WebP 压缩质量默认 85
    3. 支持 WEBP 格式（更小的文件大小）
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
        public_base_url: Optional[str] = None,
        use_public_image_url: bool = False,
        use_base64: bool = True,
        image_max_size: int = 1024,
        image_quality: int = 85,
        image_format: str = "WEBP",
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
            public_base_url (Optional[str]): 公网可访问的服务基地址（如ngrok）
            use_public_image_url (bool): 是否使用公开可访问的测试图片URL（仅用于测试）
            use_base64 (bool): 是否使用 Base64 编码方式（默认 True，推荐）
            image_max_size (int): 图片最大边长（像素），默认 1024
            image_quality (int): JPEG/WebP 压缩质量（1-100），默认 85
            image_format (str): 图片输出格式，"JPEG" 或 "WEBP"（推荐）或 "PNG"
        """
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY 未设置")
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.server_host = server_host
        self.server_port = server_port
        self.public_base_url = public_base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = client or OpenAI(api_key=api_key, base_url=base_url)
        self.use_public_image_url = use_public_image_url
        self.use_base64 = use_base64
        self.image_max_size = max(256, min(4096, image_max_size))
        self.image_quality = max(1, min(100, image_quality))
        self.image_format = image_format.upper() if image_format.upper() in ["JPEG", "WEBP", "PNG"] else "WEBP"

    def _get_image_url(self, image_path: str) -> str:
        """
        生成图片URL。

        Args:
            image_path (str): 图片路径

        Returns:
            str: 图片URL
        """
        if self.use_public_image_url:
            return (
                "https://raw.githubusercontent.com/ultralytics/yolov5/master/"
                "data/images/bus.jpg"
            )
        encoded_path = urllib.parse.quote(image_path)
        if self.public_base_url:
            base = self.public_base_url.rstrip("/")
            return f"{base}/photo?path={encoded_path}"
        return f"http://{self.server_host}:{self.server_port}/photo?path={encoded_path}"

    def _get_image_base64(self, image_path: str) -> str:
        """
        生成 Base64 编码的图片 URL。

        使用优化策略降低 Token 消耗：
        1. 自动缩放至 max_size
        2. 使用 WEBP 格式压缩（质量 85）

        Args:
            image_path (str): 图片路径

        Returns:
            str: data:image/...;base64,... 格式的 URL
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
        except Exception:
            raise ValueError(f"图片编码失败: {image_path}")

    def generate_description(self, image_path: str) -> str:
        """
        生成图片描述（中文）。

        默认使用 Base64 编码方式，因为 OpenRouter 远程服务器无法访问本地 localhost。

        Args:
            image_path (str): 图片路径

        Returns:
            str: 描述文本
        """
        prompt = (
            "请用中文描述这张图片的视觉内容，包含以下要素：\n"
            "1. 场景类型（室内/室外，具体地点如公园、海滩、街道等）\n"
            "2. 主要主体（人物、物体、动物的具体特征）\n"
            "3. 动作或状态（正在发生的事情）\n"
            "4. 环境细节（光线强度、颜色、背景元素）\n"
            "5. 情绪氛围（欢乐、温馨、宁静等感受）\n\n"
            "描述长度：50-200字\n\n"
            "注意：只描述图片中可见的视觉内容，不要推测或描述拍摄时间（如\"下午\"、\"夏天\"等）。"
        )

        if self.use_base64:
            image_content = self._get_image_base64(image_path)
        else:
            image_content = self._get_image_url(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_content}},
                ],
            }
        ]

        for attempt in range(self.max_retries):
            try:
                print(f"[DEBUG] Vision LLM API调用 (第{attempt+1}/{self.max_retries}次), 模型: {self.model_name}, timeout: {self.timeout}s")
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
                import traceback
                print(f"[ERROR] Vision LLM API调用失败 (第{attempt+1}/{self.max_retries}次): {type(exc).__name__}: {exc}")
                if attempt < self.max_retries - 1:
                    print(f"[DEBUG] 将在1秒后重试...")
                else:
                    print(f"[ERROR] Vision LLM达到最大重试次数，放弃")
                    traceback.print_exc()
                    raise ValueError(f"生成描述失败: {exc}") from exc
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
