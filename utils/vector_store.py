from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

try:
    import faiss
except ImportError as exc:
    raise ImportError("未安装faiss-cpu，请先安装依赖") from exc


class VectorStore:
    """
    向量存储与检索封装。

    Attributes:
        dimension (Optional[int]): 向量维度
        index_path (str): 索引文件路径
        metadata_path (str): 元数据文件路径
    """

    def __init__(
        self,
        dimension: Optional[int],
        index_path: str,
        metadata_path: str,
        metric: str = "cosine",
    ) -> None:
        """
        初始化FAISS索引与存储路径。

        Args:
            dimension (Optional[int]): 向量维度
            index_path (str): 索引文件路径
            metadata_path (str): 元数据文件路径
        """
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.metric = metric.lower().strip() if metric else "l2"
        if self.metric not in {"l2", "cosine"}:
            raise ValueError("metric仅支持l2或cosine")

        self.index = self._create_index(dimension) if dimension else None
        self.metadata: List[Dict] = []
        self._normalize = self.metric == "cosine"

    def _create_index(self, dimension: int) -> faiss.Index:
        if self.metric == "cosine":
            return faiss.IndexFlatIP(dimension)
        return faiss.IndexFlatL2(dimension)

    def _normalize_vector(self, vector: List[float]) -> List[float]:
        if not self._normalize:
            return vector
        array = np.array(vector, dtype="float32")
        norm = np.linalg.norm(array)
        if norm == 0:
            return vector
        return (array / norm).astype("float32").tolist()

    def _is_l2_index(self) -> bool:
        return isinstance(self.index, faiss.IndexFlatL2)

    def _is_ip_index(self) -> bool:
        return isinstance(self.index, faiss.IndexFlatIP)

    # 内部接口：仅允许indexer模块调用，禁止直接暴露给前端
    def add_item(self, embedding: List[float], metadata: Dict) -> None:
        """
        写入向量与元数据。

        Args:
            embedding (List[float]): 向量
            metadata (Dict): 元数据

        Raises:
            ValueError: 向量维度不匹配
        """
        if embedding is None:
            raise ValueError("向量不能为空")
        if self.index is None:
            self.dimension = len(embedding)
            self.index = self._create_index(self.dimension)
        if len(embedding) != self.dimension:
            raise ValueError(f"向量维度不匹配: {len(embedding)} != {self.dimension}")

        normalized = self._normalize_vector(embedding)
        vector = np.array([normalized], dtype="float32")
        self.index.add(vector)
        self.metadata.append(metadata)

    # 内部接口：仅允许searcher模块调用，禁止前端直接访问向量数据库
    def search(self, query_embedding: List[float], top_k: int) -> List[Dict]:
        """
        相似度检索。

        Args:
            query_embedding (List[float]): 查询向量
            top_k (int): 返回数量

        Returns:
            List[Dict]: 结果列表
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        if len(query_embedding) != self.dimension:
            raise ValueError(f"向量维度不匹配: {len(query_embedding)} != {self.dimension}")

        k = min(top_k, self.index.ntotal)
        normalized = self._normalize_vector(query_embedding)
        vector = np.array([normalized], dtype="float32")
        distances, indices = self.index.search(vector, k)

        results: List[Dict] = []
        for distance, index in zip(distances[0].tolist(), indices[0].tolist()):
            if index == -1:
                continue
            results.append({"metadata": self.metadata[index], "distance": float(distance)})
        return results

    def save(self) -> None:
        """
        索引持久化。

        Raises:
            ValueError: 索引未初始化
        """
        if self.index is None:
            raise ValueError("索引未初始化")

        index_dir = os.path.dirname(self.index_path)
        metadata_dir = os.path.dirname(self.metadata_path)
        if index_dir:
            os.makedirs(index_dir, exist_ok=True)
        if metadata_dir:
            os.makedirs(metadata_dir, exist_ok=True)

        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as file:
            json.dump(self.metadata, file, ensure_ascii=False, indent=2)

    def load(self) -> bool:
        """
        加载索引与元数据。

        Returns:
            bool: 是否加载成功
        """
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            return False

        self.index = faiss.read_index(self.index_path)
        if self.metric == "cosine" and not self._is_ip_index():
            raise ValueError("索引度量与配置不一致，请重新构建索引")
        if self.metric == "l2" and not self._is_l2_index():
            raise ValueError("索引度量与配置不一致，请重新构建索引")

        with open(self.metadata_path, "r", encoding="utf-8") as file:
            self.metadata = json.load(file)
        if self.index.ntotal != len(self.metadata):
            raise ValueError("索引与元数据数量不一致，请重新构建索引")
        self.dimension = self.index.d
        return True

    def get_total_items(self) -> int:
        """
        获取当前向量数量。

        Returns:
            int: 向量数量
        """
        if self.index is None:
            return 0
        return int(self.index.ntotal)
