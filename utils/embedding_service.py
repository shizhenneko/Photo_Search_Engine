from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import List, Optional

import torch

from openai import OpenAI


class EmbeddingService(ABC):
    """
    Embedding服务抽象接口。
    """

    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """
        生成文本嵌入向量。

        Args:
            text (str): 输入文本

        Returns:
            List[float]: 向量
        """

    @abstractmethod
    def generate_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成嵌入向量。

        Args:
            texts (List[str]): 文本列表

        Returns:
            List[List[float]]: 向量列表
        """


class OpenAIEmbeddingService(EmbeddingService):
    """
    OpenAI Embedding服务实现（OpenRouter兼容）。
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "text-embedding-3-small",
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: int = 30,
        max_retries: int = 3,
        client: Optional[OpenAI] = None,
    ) -> None:
        """
        初始化Embedding服务。

        Args:
            api_key (str): OpenRouter API密钥
            model_name (str): 模型名称
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

    def generate_embedding(self, text: str) -> List[float]:
        """
        生成单条文本嵌入向量。

        Args:
            text (str): 输入文本

        Returns:
            List[float]: 向量
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=text,
                    timeout=self.timeout,
                )
                return response.data[0].embedding
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise ValueError(f"向量生成失败: {exc}") from exc
                time.sleep(1)
        raise ValueError("向量生成失败")

    def generate_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成嵌入向量。

        Args:
            texts (List[str]): 文本列表

        Returns:
            List[List[float]]: 向量列表
        """
        embeddings: List[List[float]] = []
        for text in texts:
            embeddings.append(self.generate_embedding(text))
        return embeddings


class T5EmbeddingService(EmbeddingService):
    """
    基于sentenceTransformers的T5嵌入服务。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: Optional[str] = None,
        model: Optional[object] = None,
    ) -> None:
        """
        初始化T5嵌入服务。

        Args:
            model_name (str): 模型名称
            device (Optional[str]): 运行设备
            model (Optional[object]): 注入模型（用于测试）
        """
        self.model_name = model_name
        if model is not None:
            self.model = model
            return

        import os

        os.environ["TRANSFORMERS_NO_TF"] = "1"
        os.environ["TRANSFORMERS_USE_TORCH"] = "1"

        if device is not None and device.lower() == "cuda":
            if not torch.cuda.is_available():
                import warnings
                warnings.warn(
                    "CUDA requested but not available. Falling back to CPU. "
                    "This is expected if PyTorch was not compiled with CUDA support.",
                    RuntimeWarning,
                    stacklevel=2
                )
                device = "cpu"

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("未安装sentence-transformers，请先安装依赖") from exc

        self.model = SentenceTransformer(model_name, device=device)

    def generate_embedding(self, text: str) -> List[float]:
        """
        生成单条文本嵌入向量。

        Args:
            text (str): 输入文本

        Returns:
            List[float]: 向量
        """
        embedding = self.model.encode(text)
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

    def generate_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成嵌入向量，提高效率。

        Args:
            texts (List[str]): 文本列表

        Returns:
            List[List[float]]: 向量列表
        """
        embeddings = self.model.encode(texts)
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
        return [list(item) for item in embeddings]


class SentenceTransformerService(T5EmbeddingService):
    """
    SentenceTransformer命名兼容类（行为与T5EmbeddingService一致）。
    """


class VolcanoEmbeddingService(EmbeddingService):
    """
    火山引擎 Doubao Embedding 服务实现。

    Attributes:
        api_key (str): 火山引擎 API 密钥
        model_name (str): 模型名称（默认 doubao-embedding-large-text-240915）
        base_url (str): API 基础地址
        dimension (int): 输出向量维度（4096）
        timeout (int): API 超时时间（秒）
        max_retries (int): 最大重试次数
    """

    # 模型固定输出维度
    dimension: int = 4096

    def __init__(
        self,
        api_key: str,
        model_name: str = "doubao-embedding-large-text-240915",
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        timeout: int = 30,
        max_retries: int = 3,
        client: Optional[OpenAI] = None,
    ) -> None:
        """
        初始化火山引擎 Embedding 服务。

        Args:
            api_key (str): 火山引擎 API 密钥（必填）
            model_name (str): 模型名称
            base_url (str): API 基础地址
            timeout (int): 超时时间（秒）
            max_retries (int): 最大重试次数
            client (Optional[OpenAI]): 可选的预配置客户端（用于测试注入）

        Raises:
            ValueError: API 密钥未设置时抛出
        """
        if not api_key:
            raise ValueError("VOLCANO_API_KEY 未设置")

        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = client or OpenAI(api_key=api_key, base_url=base_url)

    def generate_embedding(self, text: str) -> List[float]:
        """
        生成单条文本的嵌入向量。

        Args:
            text (str): 输入文本（建议长度不超过8192 tokens）

        Returns:
            List[float]: 4096维向量

        Raises:
            ValueError: API 调用失败时抛出
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=text,
                    timeout=self.timeout,
                )
                return response.data[0].embedding
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise ValueError(f"向量生成失败: {exc}") from exc
                time.sleep(1)
        raise ValueError("向量生成失败")

    def generate_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成嵌入向量。

        Args:
            texts (List[str]): 文本列表

        Returns:
            List[List[float]]: 向量列表，每个向量4096维

        Note:
            火山引擎支持批量请求，单次最多支持16条文本。
            超过时自动分批处理。
        """
        BATCH_SIZE = 16
        embeddings: List[List[float]] = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            for attempt in range(self.max_retries):
                try:
                    response = self.client.embeddings.create(
                        model=self.model_name,
                        input=batch,
                        timeout=self.timeout,
                    )
                    # 按原始顺序提取向量
                    batch_embeddings = [item.embedding for item in response.data]
                    embeddings.extend(batch_embeddings)
                    break
                except Exception as exc:
                    if attempt == self.max_retries - 1:
                        raise ValueError(f"批量向量生成失败: {exc}") from exc
                    time.sleep(1)

        return embeddings

