from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

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
        index_type: str = "flat",
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 96,
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
        self.meta_path = f"{self.index_path}.meta.json"
        self.metric = metric.lower().strip() if metric else "l2"
        if self.metric not in {"l2", "cosine"}:
            raise ValueError("metric仅支持l2或cosine")
        self.index_type = (index_type or "flat").strip().lower()
        if self.index_type not in {"flat", "hnsw"}:
            raise ValueError("index_type仅支持flat或hnsw")
        self.hnsw_m = max(4, int(hnsw_m))
        self.hnsw_ef_construction = max(8, int(hnsw_ef_construction))
        self.hnsw_ef_search = max(8, int(hnsw_ef_search))

        self.index = self._create_index(dimension) if dimension else None
        self.metadata: List[Dict] = []
        self._normalize = self.metric == "cosine"
        self._embeddings: List[Optional[List[float]]] = []
        self._path_to_index: Dict[str, int] = {}

    def _rebuild_path_index(self) -> None:
        path_to_index: Dict[str, int] = {}
        for index, metadata in enumerate(self.metadata):
            photo_path = metadata.get("photo_path")
            if isinstance(photo_path, str) and photo_path:
                path_to_index[photo_path] = index
        self._path_to_index = path_to_index

    def _create_index(self, dimension: int) -> faiss.Index:
        if self.index_type == "hnsw":
            metric_type = faiss.METRIC_INNER_PRODUCT if self.metric == "cosine" else faiss.METRIC_L2
            index = faiss.IndexHNSWFlat(dimension, self.hnsw_m, metric_type)
            index.hnsw.efConstruction = self.hnsw_ef_construction
            index.hnsw.efSearch = self.hnsw_ef_search
            return index
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
        return bool(self.index is not None and getattr(self.index, "metric_type", None) == faiss.METRIC_L2)

    def _is_ip_index(self) -> bool:
        return bool(
            self.index is not None
            and getattr(self.index, "metric_type", None) == faiss.METRIC_INNER_PRODUCT
        )

    def _is_hnsw_index(self) -> bool:
        return bool(self.index is not None and type(self.index).__name__ == "IndexHNSWFlat")

    def _write_index_meta(self) -> None:
        payload = {
            "index_type": self.index_type,
            "metric": self.metric,
            "dimension": self.dimension,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_construction": self.hnsw_ef_construction,
            "hnsw_ef_search": self.hnsw_ef_search,
        }
        with open(self.meta_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    def _load_index_meta(self) -> Dict[str, Any]:
        if not os.path.exists(self.meta_path):
            raise ValueError("索引元信息缺失，请重新构建索引")
        with open(self.meta_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError("索引元信息损坏，请重新构建索引")
        return payload

    def _validate_loaded_index(self, payload: Dict[str, Any]) -> None:
        index_type = str(payload.get("index_type") or "").strip().lower()
        metric = str(payload.get("metric") or "").strip().lower()
        if index_type != self.index_type:
            raise ValueError("索引类型与配置不一致，请重新构建索引")
        if metric != self.metric:
            raise ValueError("索引度量与配置不一致，请重新构建索引")
        if self.index_type == "hnsw":
            if not self._is_hnsw_index():
                raise ValueError("索引结构与配置不一致，请重新构建索引")
            self.index.hnsw.efSearch = self.hnsw_ef_search
        elif self.index_type == "flat":
            if self.metric == "cosine" and not self._is_ip_index():
                raise ValueError("索引度量与配置不一致，请重新构建索引")
            if self.metric == "l2" and not self._is_l2_index():
                raise ValueError("索引度量与配置不一致，请重新构建索引")

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
        self._embeddings.append(normalized)
        photo_path = metadata.get("photo_path")
        if isinstance(photo_path, str) and photo_path:
            self._path_to_index[photo_path] = len(self.metadata) - 1

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

    def get_embedding_by_photo_path(self, photo_path: str) -> Optional[List[float]]:
        index = self._path_to_index.get(photo_path)
        if index is None:
            return None
        if index < len(self._embeddings):
            cached = self._embeddings[index]
            if cached is None and self.index is not None:
                vector = self.index.reconstruct(index)
                cached = vector.astype("float32").tolist()
                self._embeddings[index] = cached
            if cached is not None:
                return list(cached)
        return None

    def has_photo_path(self, photo_path: str) -> bool:
        return photo_path in self._path_to_index

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
        self._write_index_meta()
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
        payload = self._load_index_meta()
        self._validate_loaded_index(payload)

        with open(self.metadata_path, "r", encoding="utf-8") as file:
            self.metadata = json.load(file)
        if self.index.ntotal != len(self.metadata):
            raise ValueError("索引与元数据数量不一致，请重新构建索引")
        self.dimension = self.index.d
        self._embeddings = [None] * self.index.ntotal
        self._rebuild_path_index()
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

    def clear(self) -> None:
        """
        清空索引与元数据。
        """
        self.index = self._create_index(self.dimension) if self.dimension else None
        self.metadata = []
        self._embeddings = []
        self._path_to_index = {}
