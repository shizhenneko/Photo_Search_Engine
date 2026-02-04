# Photo Search Engine 功能优化文档

> **文档版本**: v1.0  
> **最后更新**: 2026-02-04  
> **修改状态**: 待实施

---

## 目录

1. [修改概述](#修改概述)
2. [嵌入模型替换](#一嵌入模型替换)
3. [查询策略优化](#二查询策略优化)
4. [用户查询转换](#三用户查询转换)
5. [接口对齐检查](#四接口对齐检查)
6. [单元测试要求](#五单元测试要求)
7. [文档维护规范](#六文档维护规范)

---

## 修改概述

本次优化旨在提升照片搜索引擎的检索准确性和用户体验，主要包含三个核心改进：

| 改进项 | 当前方案 | 优化方案 | 预期效果 |
|--------|----------|----------|----------|
| 嵌入模型 | `BAAI/bge-small-zh-v1.5` (512维) | `Doubao-embedding-large` (4096维) | 语义理解能力提升 |
| 检索策略 | 纯向量相似度 | 向量+关键字混合检索 | 减少语义误判 |
| 查询处理 | 原始查询直接检索 | LLM格式化后检索 | 提取核心需求 |

---

## 一、嵌入模型替换

### 1.1 修改背景

当前使用本地 `BAAI/bge-small-zh-v1.5` 模型（512维），语义理解能力有限。升级为火山引擎的 `Doubao-embedding-large` 模型（4096维），可显著提升中文语义理解精度。

### 1.2 涉及文件

| 文件路径 | 修改类型 | 说明 |
|----------|----------|------|
| `config.py` | 新增配置项 | 添加火山引擎相关环境变量 |
| `utils/embedding_service.py` | 新增类 | 实现 `VolcanoEmbeddingService` |
| `main.py` | 修改初始化 | 切换为新嵌入服务 |
| `.env.example` | 新增示例 | 添加火山引擎配置示例 |
| `tests/test_embedding_service.py` | 新增测试 | 覆盖新服务单元测试 |

### 1.3 环境变量配置

在 `.env` 文件中新增以下配置（**禁止硬编码**）：

```bash
# 火山引擎 Embedding 配置
VOLCANO_API_KEY=your_volcano_api_key
VOLCANO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
VOLCANO_EMBEDDING_MODEL=doubao-embedding-large-text-240915

# 嵌入维度（必须与模型输出一致）
EMBEDDING_DIMENSION=4096
```

### 1.4 config.py 修改

```python
# 在 load_config() 函数中新增：

def load_config() -> Dict[str, Any]:
    # ... 现有代码 ...
    
    config: Dict[str, Any] = {
        # ... 现有配置 ...
        
        # 火山引擎 Embedding 配置（新增）
        "VOLCANO_API_KEY": os.getenv("VOLCANO_API_KEY"),
        "VOLCANO_BASE_URL": os.getenv("VOLCANO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        "VOLCANO_EMBEDDING_MODEL": os.getenv("VOLCANO_EMBEDDING_MODEL", "doubao-embedding-large-text-240915"),
        
        # 更新默认维度为4096
        "EMBEDDING_DIMENSION": _get_int("EMBEDDING_DIMENSION", 4096),
    }
    
    return config
```

### 1.5 新增 VolcanoEmbeddingService 类

**文件**: `utils/embedding_service.py`

```python
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
            batch = texts[i:i + BATCH_SIZE]
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
```

### 1.6 main.py 服务初始化修改

```python
def initialize_services(config: Dict[str, object]) -> Tuple[Indexer, Searcher]:
    # ... 现有代码 ...
    
    # 修改：使用火山引擎 Embedding 服务
    from utils.embedding_service import VolcanoEmbeddingService
    
    embedding_service = VolcanoEmbeddingService(
        api_key=str(config.get("VOLCANO_API_KEY", "")),
        model_name=str(config.get("VOLCANO_EMBEDDING_MODEL", "doubao-embedding-large-text-240915")),
        base_url=str(config.get("VOLCANO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")),
        timeout=int(config.get("TIMEOUT", 30)),
        max_retries=int(config.get("MAX_RETRIES", 3)),
    )
    
    # 更新 VectorStore 维度
    vector_store = VectorStore(
        dimension=4096,  # 与 Doubao-embedding-large 一致
        # ... 其他参数保持不变 ...
    )
    
    # ... 其余代码 ...
```

### 1.7 接口定义

#### VolcanoEmbeddingService 公开接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__` | `api_key, model_name, base_url, timeout, max_retries, client` | `None` | 初始化服务 |
| `generate_embedding` | `text: str` | `List[float]` (4096维) | 单条文本嵌入 |
| `generate_embedding_batch` | `texts: List[str]` | `List[List[float]]` | 批量嵌入 |

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `dimension` | `int` | 固定值 4096 |
| `api_key` | `str` | API 密钥 |
| `model_name` | `str` | 模型名称 |

---

## 二、查询策略优化

### 2.1 修改背景

当前检索仅基于向量余弦相似度，可能导致语义相近但实际不相关的结果排名靠前（如"公园"与"博物馆展览"因"展示"语义相近而误判）。

**优化方案**：引入 Elasticsearch + BM25 关键字检索，与向量检索结合。

| 检索方式 | 权重 | 说明 |
|----------|------|------|
| 向量相似度 | 80% | 语义匹配 |
| 关键字匹配 | 20% | 精确匹配 |

### 2.2 涉及文件

| 文件路径 | 修改类型 | 说明 |
|----------|----------|------|
| `requirements.txt` | 新增依赖 | elasticsearch |
| `utils/keyword_store.py` | 新建文件 | ES 关键字索引服务 |
| `core/searcher.py` | 修改检索逻辑 | 混合评分 |
| `core/indexer.py` | 修改索引逻辑 | 同步写入 ES |
| `config.py` | 新增配置 | ES 连接配置 |
| `tests/test_keyword_store.py` | 新建文件 | ES 服务单元测试 |

### 2.3 环境变量配置

```bash
# Elasticsearch 配置
ELASTICSEARCH_HOST=localhost
ELASTICSEARCH_PORT=9200
ELASTICSEARCH_INDEX=photo_keywords
ELASTICSEARCH_USERNAME=elastic
ELASTICSEARCH_PASSWORD=your_password

# 混合检索权重配置
VECTOR_WEIGHT=0.8
KEYWORD_WEIGHT=0.2
```

### 2.4 新建 KeywordStore 类

**文件**: `utils/keyword_store.py`

```python
from __future__ import annotations

from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch


class KeywordStore:
    """
    基于 Elasticsearch 的关键字索引服务。
    
    使用 BM25 算法进行文本相关性匹配。
    
    Attributes:
        es_client (Elasticsearch): ES 客户端
        index_name (str): 索引名称
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 9200,
        index_name: str = "photo_keywords",
        username: Optional[str] = None,
        password: Optional[str] = None,
        client: Optional[Elasticsearch] = None,
    ) -> None:
        """
        初始化 Elasticsearch 连接。
        
        Args:
            host (str): ES 主机地址
            port (int): ES 端口
            index_name (str): 索引名称
            username (Optional[str]): 用户名（可选）
            password (Optional[str]): 密码（可选）
            client (Optional[Elasticsearch]): 预配置客户端（用于测试）
        
        Raises:
            ConnectionError: ES 连接失败时抛出
        """
        self.index_name = index_name
        
        if client is not None:
            self.es_client = client
        else:
            auth = (username, password) if username and password else None
            self.es_client = Elasticsearch(
                hosts=[{"host": host, "port": port}],
                basic_auth=auth,
            )
        
        self._ensure_index()
    
    def _ensure_index(self) -> None:
        """
        确保索引存在，不存在则创建。
        
        索引 Mapping 配置：
        - photo_path: keyword（精确匹配）
        - description: text（中文分词，BM25）
        - file_name: text（文件名模糊匹配）
        - time_text: text（时间描述）
        """
        if not self.es_client.indices.exists(index=self.index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "photo_path": {"type": "keyword"},
                        "description": {
                            "type": "text",
                            "analyzer": "ik_max_word",  # 中文分词器
                            "search_analyzer": "ik_smart",
                        },
                        "file_name": {"type": "text"},
                        "time_text": {"type": "text"},
                    }
                },
                "settings": {
                    "index": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                    }
                },
            }
            self.es_client.indices.create(index=self.index_name, body=mapping)
    
    def add_document(self, doc_id: str, document: Dict[str, Any]) -> None:
        """
        添加或更新文档。
        
        Args:
            doc_id (str): 文档 ID（建议使用 photo_path 的 hash）
            document (Dict[str, Any]): 文档内容
                - photo_path: str（必填）
                - description: str（必填）
                - file_name: str（可选）
                - time_text: str（可选）
        
        Raises:
            ValueError: 文档格式错误时抛出
        """
        if "photo_path" not in document or "description" not in document:
            raise ValueError("文档必须包含 photo_path 和 description 字段")
        
        self.es_client.index(
            index=self.index_name,
            id=doc_id,
            document=document,
        )
    
    def search(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """
        执行 BM25 关键字搜索。
        
        Args:
            query (str): 查询文本
            top_k (int): 返回数量
        
        Returns:
            List[Dict[str, Any]]: 搜索结果列表
                - photo_path: str
                - score: float（BM25 分数，已归一化到 0-1）
        """
        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["description^3", "file_name", "time_text"],
                    "type": "best_fields",
                }
            },
            "size": top_k,
        }
        
        response = self.es_client.search(index=self.index_name, body=body)
        hits = response["hits"]["hits"]
        
        if not hits:
            return []
        
        # 归一化 BM25 分数到 0-1
        max_score = response["hits"]["max_score"] or 1.0
        
        results = []
        for hit in hits:
            results.append({
                "photo_path": hit["_source"]["photo_path"],
                "score": hit["_score"] / max_score,  # 归一化
            })
        
        return results
    
    def delete_index(self) -> None:
        """删除整个索引（用于重建）。"""
        if self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.delete(index=self.index_name)
    
    def get_document_count(self) -> int:
        """获取索引中的文档数量。"""
        self.es_client.indices.refresh(index=self.index_name)
        return self.es_client.count(index=self.index_name)["count"]
```

### 2.5 Searcher 类修改

**文件**: `core/searcher.py`

修改 `search` 方法，添加混合评分逻辑：

```python
class Searcher:
    def __init__(
        self,
        embedding: "EmbeddingService",
        time_parser: "TimeParser",
        vector_store: VectorStore,
        keyword_store: Optional["KeywordStore"] = None,  # 新增参数
        data_dir: str = "./data",
        top_k: int = 10,
        vector_weight: float = 0.8,   # 新增参数
        keyword_weight: float = 0.2,  # 新增参数
    ) -> None:
        """
        初始化检索器。
        
        Args:
            embedding: 嵌入服务
            time_parser: 时间解析器
            vector_store: 向量存储
            keyword_store: 关键字存储（可选，不传则禁用混合检索）
            data_dir: 数据目录
            top_k: 默认返回数量
            vector_weight: 向量检索权重（0-1）
            keyword_weight: 关键字检索权重（0-1）
        
        Raises:
            ValueError: 权重之和不为 1 时抛出
        """
        if abs(vector_weight + keyword_weight - 1.0) > 0.001:
            raise ValueError("vector_weight + keyword_weight 必须等于 1.0")
        
        self.embedding_service = embedding
        self.time_parser = time_parser
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.data_dir = data_dir
        self.top_k = max(1, top_k)
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.index_loaded = False
        # ... 其余初始化代码 ...
    
    def _hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        candidate_k: int,
    ) -> List[Dict[str, Any]]:
        """
        执行混合检索（向量 + 关键字）。
        
        Args:
            query: 原始查询文本
            query_embedding: 查询向量
            candidate_k: 候选数量
        
        Returns:
            List[Dict[str, Any]]: 混合排序后的结果
        """
        # 1. 向量检索
        vector_results = self.vector_store.search(query_embedding, candidate_k)
        
        # 2. 构建向量分数映射
        vector_scores: Dict[str, float] = {}
        for item in vector_results:
            metadata = item.get("metadata") or {}
            photo_path = metadata.get("photo_path", "")
            score = self._distance_to_score(float(item.get("distance", 0.0)))
            vector_scores[photo_path] = score
        
        # 3. 关键字检索（如果启用）
        keyword_scores: Dict[str, float] = {}
        if self.keyword_store is not None:
            keyword_results = self.keyword_store.search(query, candidate_k)
            for item in keyword_results:
                keyword_scores[item["photo_path"]] = item["score"]
        
        # 4. 混合评分
        all_paths = set(vector_scores.keys()) | set(keyword_scores.keys())
        combined_results: List[Dict[str, Any]] = []
        
        for photo_path in all_paths:
            v_score = vector_scores.get(photo_path, 0.0)
            k_score = keyword_scores.get(photo_path, 0.0)
            
            # 加权融合
            combined_score = (
                self.vector_weight * v_score + 
                self.keyword_weight * k_score
            )
            
            # 获取元数据
            metadata = self._get_metadata_by_path(photo_path, vector_results)
            
            combined_results.append({
                "photo_path": photo_path,
                "description": metadata.get("description", ""),
                "score": round(combined_score, 6),
                "vector_score": round(v_score, 6),
                "keyword_score": round(k_score, 6),
            })
        
        # 5. 按混合分数降序排序
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        
        return combined_results
    
    def _get_metadata_by_path(
        self,
        photo_path: str,
        vector_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """从向量检索结果中获取元数据。"""
        for item in vector_results:
            metadata = item.get("metadata") or {}
            if metadata.get("photo_path") == photo_path:
                return metadata
        return {}
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        解析查询、执行混合检索、时间过滤并返回排序结果。
        
        修改点：使用 _hybrid_search 替代纯向量检索。
        """
        # ... 查询验证、索引检查等代码保持不变 ...
        
        query_embedding = self.embedding_service.generate_embedding(cleaned_query)
        candidate_k = self._calculate_candidate_k(normalized_top_k, has_time_filter)
        
        # 修改：使用混合检索
        if self.keyword_store is not None:
            combined_results = self._hybrid_search(cleaned_query, query_embedding, candidate_k)
        else:
            # 降级为纯向量检索
            raw_results = self.vector_store.search(query_embedding, candidate_k)
            combined_results = []
            for item in raw_results:
                metadata = item.get("metadata") or {}
                score = self._distance_to_score(float(item.get("distance", 0.0)))
                combined_results.append({
                    "photo_path": metadata.get("photo_path"),
                    "description": metadata.get("description"),
                    "score": score,
                })
        
        # ... 时间过滤、阈值计算等代码保持不变 ...
```

### 2.6 Indexer 类修改

**文件**: `core/indexer.py`

在索引构建时同步写入 Elasticsearch：

```python
class Indexer:
    def __init__(
        self,
        # ... 现有参数 ...
        keyword_store: Optional["KeywordStore"] = None,  # 新增参数
    ) -> None:
        # ... 现有代码 ...
        self.keyword_store = keyword_store
    
    def _index_single_photo(self, photo_path: str) -> bool:
        """
        索引单张照片。
        
        修改点：同步写入 KeywordStore。
        """
        # ... 现有的向量索引逻辑 ...
        
        # 新增：同步写入 Elasticsearch
        if self.keyword_store is not None:
            import hashlib
            doc_id = hashlib.md5(photo_path.encode()).hexdigest()
            document = {
                "photo_path": photo_path,
                "description": description,
                "file_name": os.path.basename(photo_path),
                "time_text": self._extract_time_text(metadata),
            }
            self.keyword_store.add_document(doc_id, document)
        
        return True
```

### 2.7 接口定义

#### KeywordStore 公开接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__` | `host, port, index_name, username, password, client` | `None` | 初始化 ES 连接 |
| `add_document` | `doc_id: str, document: Dict` | `None` | 添加文档 |
| `search` | `query: str, top_k: int` | `List[Dict]` | BM25 搜索 |
| `delete_index` | `None` | `None` | 删除索引 |
| `get_document_count` | `None` | `int` | 获取文档数 |

#### Searcher 新增接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `_hybrid_search` | `query, query_embedding, candidate_k` | `List[Dict]` | 混合检索 |

---

## 三、用户查询转换

### 3.1 修改背景

用户查询可能存在歧义，如：
- "请展示一张公园的照片" → "展示"与"博物馆展览"语义相近
- "去年拍的海边照片" → 需要提取时间和场景信息

**优化方案**：引入 LLM 对用户查询进行格式化，提取核心需求。

### 3.2 涉及文件

| 文件路径 | 修改类型 | 说明 |
|----------|----------|------|
| `utils/query_formatter.py` | 新建文件 | 查询格式化服务 |
| `core/searcher.py` | 修改检索逻辑 | 调用格式化服务 |
| `config.py` | 新增配置 | LLM 配置项 |
| `tests/test_query_formatter.py` | 新建文件 | 单元测试 |

### 3.3 环境变量配置

```bash
# 查询格式化 LLM 配置
QUERY_FORMAT_API_KEY=your_api_key
QUERY_FORMAT_BASE_URL=https://your-llm-endpoint/api/v1
QUERY_FORMAT_MODEL=your-model-name
```

### 3.4 新建 QueryFormatter 类

**文件**: `utils/query_formatter.py`

```python
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
    "time_period": "时段（早晨/白天/傍晚/夜晚）或 null",
    "search_text": "生成一段适合向量检索的描述，包含场景、物体、氛围的自然语言描述"
}}

规则：
1. 从用户查询中提取核心检索需求，忽略无关的礼貌用语（如"请"、"展示"）
2. search_text 应该是一段自然的描述，适合与照片描述进行语义匹配
3. 如果用户未提及时间，time_hint、season、time_period 返回 null
4. objects 列出可能出现在照片中的主要物体"""

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
                
                # 构建完整的检索文本
                search_parts = [result.get("search_text", user_query)]
                if result.get("time_hint"):
                    search_parts.append(f"时间: {result['time_hint']}")
                if result.get("season"):
                    search_parts.append(f"季节: {result['season']}")
                if result.get("time_period"):
                    search_parts.append(f"时段: {result['time_period']}")
                
                result["search_text"] = " | ".join(search_parts)
                
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
```

### 3.5 Searcher 集成 QueryFormatter

**文件**: `core/searcher.py`

```python
class Searcher:
    def __init__(
        self,
        # ... 现有参数 ...
        query_formatter: Optional["QueryFormatter"] = None,  # 新增参数
    ) -> None:
        # ... 现有代码 ...
        self.query_formatter = query_formatter
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        解析查询、执行检索、返回结果。
        
        修改点：先调用 QueryFormatter 格式化查询。
        """
        if not self.validate_query(query):
            raise ValueError("查询内容不合法，请输入5-500字符的描述")
        
        if not self.index_loaded and not self.load_index():
            raise ValueError("索引未加载，请先初始化索引")
        
        normalized_top_k = max(1, min(int(top_k), 50))
        
        # 新增：查询格式化
        formatted_query = query
        time_hints = {}
        
        if self.query_formatter is not None and self.query_formatter.is_enabled():
            format_result = self.query_formatter.format_query(query)
            formatted_query = format_result.get("search_text", query)
            time_hints = {
                "time_hint": format_result.get("time_hint"),
                "season": format_result.get("season"),
            }
        
        # 时间约束提取（可结合格式化结果）
        constraints = {"start_date": None, "end_date": None, "precision": "none"}
        has_time_filter = self._has_time_terms(query)
        
        if has_time_filter:
            constraints = self._extract_time_constraints(query)
        
        # 使用格式化后的查询生成向量
        query_embedding = self.embedding_service.generate_embedding(formatted_query)
        
        # ... 后续检索逻辑 ...
```

### 3.6 接口定义

#### QueryFormatter 公开接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__` | `api_key, model_name, base_url, timeout, max_retries, client` | `None` | 初始化服务 |
| `format_query` | `user_query: str` | `Dict[str, Any]` | 格式化查询 |
| `is_enabled` | `None` | `bool` | 检查服务状态 |

#### format_query 返回结构

```python
{
    "search_text": str,      # 格式化后的检索文本
    "scene": str | None,     # 场景描述
    "objects": List[str],    # 物体列表
    "atmosphere": str | None, # 氛围描述
    "time_hint": str | None,  # 时间提示
    "season": str | None,     # 季节
    "time_period": str | None, # 时段
    "original_query": str,    # 原始查询
}
```

---

## 四、接口对齐检查

### 4.1 检查清单

修改完成后，必须验证以下接口保持兼容：

#### 4.1.1 EmbeddingService 接口

| 接口 | 原签名 | 修改后签名 | 兼容性 |
|------|--------|------------|--------|
| `generate_embedding` | `(text: str) -> List[float]` | 不变 | ✅ 兼容 |
| `generate_embedding_batch` | `(texts: List[str]) -> List[List[float]]` | 不变 | ✅ 兼容 |
| `dimension` 属性 | 512 | 4096 | ⚠️ 需重建索引 |

#### 4.1.2 Searcher 接口

| 接口 | 原签名 | 修改后签名 | 兼容性 |
|------|--------|------------|--------|
| `search` | `(query: str, top_k: int) -> List[Dict]` | 不变 | ✅ 兼容 |
| `load_index` | `() -> bool` | 不变 | ✅ 兼容 |
| `get_index_stats` | `() -> Dict[str, Any]` | 不变 | ✅ 兼容 |

#### 4.1.3 API 路由接口

| 路由 | 请求格式 | 响应格式 | 兼容性 |
|------|----------|----------|--------|
| `POST /search_photos` | `{"query": str, "top_k": int}` | 不变 | ✅ 兼容 |
| `POST /init_index` | 无 | 不变 | ✅ 兼容 |
| `GET /index_status` | 无 | 不变 | ✅ 兼容 |

#### 4.1.4 返回结果格式

原格式：
```json
{
    "photo_path": "...",
    "description": "...",
    "score": 0.85,
    "rank": 1
}
```

新格式（向后兼容）：
```json
{
    "photo_path": "...",
    "description": "...",
    "score": 0.85,
    "rank": 1,
    "vector_score": 0.88,    // 新增（可选）
    "keyword_score": 0.73    // 新增（可选）
}
```

### 4.2 兼容性处理

1. **向量维度变更**：修改后需要**完全重建索引**，旧索引不兼容
2. **KeywordStore 可选**：`keyword_store=None` 时降级为纯向量检索
3. **QueryFormatter 可选**：`query_formatter=None` 时使用原始查询

---

## 五、单元测试要求

### 5.1 测试原则

> **重要**: 单元测试是保证代码质量和接口稳定性的关键。每次修改必须：
> 1. **先写测试，后改代码**（TDD 原则）
> 2. **100% 覆盖公开接口**
> 3. **测试边界条件和异常情况**
> 4. **Mock 外部依赖**（API、数据库）

### 5.2 测试文件结构

```
tests/
├── test_embedding_service.py      # 嵌入服务测试（需新增 Volcano 测试）
├── test_keyword_store.py          # 关键字存储测试（新建）
├── test_query_formatter.py        # 查询格式化测试（新建）
├── test_searcher.py               # 检索器测试（需更新）
├── test_indexer.py                # 索引器测试（需更新）
└── conftest.py                    # 测试夹具
```

### 5.3 VolcanoEmbeddingService 测试用例

**文件**: `tests/test_embedding_service.py`

```python
class TestVolcanoEmbeddingService(unittest.TestCase):
    """火山引擎 Embedding 服务测试。"""
    
    def test_init_requires_api_key(self) -> None:
        """测试初始化必须提供 API 密钥。"""
        with self.assertRaises(ValueError) as context:
            VolcanoEmbeddingService(api_key="")
        self.assertIn("VOLCANO_API_KEY", str(context.exception))
    
    def test_dimension_is_4096(self) -> None:
        """测试向量维度固定为 4096。"""
        # Mock 客户端
        mock_client = Mock()
        service = VolcanoEmbeddingService(
            api_key="test-key",
            client=mock_client,
        )
        self.assertEqual(service.dimension, 4096)
    
    def test_generate_embedding_returns_list(self) -> None:
        """测试生成嵌入返回列表。"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 4096)]
        mock_client.embeddings.create.return_value = mock_response
        
        service = VolcanoEmbeddingService(
            api_key="test-key",
            client=mock_client,
        )
        
        result = service.generate_embedding("测试文本")
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4096)
    
    def test_generate_embedding_retries_on_failure(self) -> None:
        """测试失败后自动重试。"""
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = [
            Exception("网络错误"),
            Exception("网络错误"),
            Mock(data=[Mock(embedding=[0.1] * 4096)]),  # 第三次成功
        ]
        
        service = VolcanoEmbeddingService(
            api_key="test-key",
            max_retries=3,
            client=mock_client,
        )
        
        result = service.generate_embedding("测试文本")
        self.assertEqual(len(result), 4096)
        self.assertEqual(mock_client.embeddings.create.call_count, 3)
    
    def test_generate_embedding_raises_after_max_retries(self) -> None:
        """测试超过最大重试次数后抛出异常。"""
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("持续失败")
        
        service = VolcanoEmbeddingService(
            api_key="test-key",
            max_retries=3,
            client=mock_client,
        )
        
        with self.assertRaises(ValueError) as context:
            service.generate_embedding("测试文本")
        self.assertIn("向量生成失败", str(context.exception))
    
    def test_generate_embedding_batch(self) -> None:
        """测试批量生成嵌入。"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * 4096),
            Mock(embedding=[0.2] * 4096),
        ]
        mock_client.embeddings.create.return_value = mock_response
        
        service = VolcanoEmbeddingService(
            api_key="test-key",
            client=mock_client,
        )
        
        texts = ["文本1", "文本2"]
        results = service.generate_embedding_batch(texts)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(len(results[0]), 4096)
    
    @unittest.skipIf(
        not bool(os.getenv("VOLCANO_API_KEY")),
        "VOLCANO_API_KEY 未设置，跳过集成测试"
    )
    def test_real_api_integration(self) -> None:
        """集成测试：真实 API 调用（需要配置环境变量）。"""
        service = VolcanoEmbeddingService(
            api_key=os.getenv("VOLCANO_API_KEY"),
            base_url=os.getenv("VOLCANO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        )
        
        result = service.generate_embedding("测试火山引擎嵌入服务")
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4096)
        self.assertTrue(all(isinstance(x, float) for x in result))
```

### 5.4 KeywordStore 测试用例

**文件**: `tests/test_keyword_store.py`

```python
class TestKeywordStore(unittest.TestCase):
    """Elasticsearch 关键字存储测试。"""
    
    def setUp(self) -> None:
        """设置 Mock 客户端。"""
        self.mock_es = Mock()
        self.mock_es.indices.exists.return_value = True
        self.store = KeywordStore(
            index_name="test_index",
            client=self.mock_es,
        )
    
    def test_add_document_requires_fields(self) -> None:
        """测试添加文档必须包含必填字段。"""
        with self.assertRaises(ValueError):
            self.store.add_document("doc1", {"photo_path": "/test.jpg"})
        
        with self.assertRaises(ValueError):
            self.store.add_document("doc1", {"description": "test"})
    
    def test_add_document_success(self) -> None:
        """测试成功添加文档。"""
        document = {
            "photo_path": "/photos/test.jpg",
            "description": "海边日落照片",
            "file_name": "test.jpg",
        }
        
        self.store.add_document("doc1", document)
        
        self.mock_es.index.assert_called_once_with(
            index="test_index",
            id="doc1",
            document=document,
        )
    
    def test_search_returns_normalized_scores(self) -> None:
        """测试搜索返回归一化分数。"""
        self.mock_es.search.return_value = {
            "hits": {
                "max_score": 10.0,
                "hits": [
                    {"_source": {"photo_path": "/a.jpg"}, "_score": 10.0},
                    {"_source": {"photo_path": "/b.jpg"}, "_score": 5.0},
                ],
            }
        }
        
        results = self.store.search("海边", top_k=10)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["score"], 1.0)  # 10/10
        self.assertEqual(results[1]["score"], 0.5)  # 5/10
    
    def test_search_empty_results(self) -> None:
        """测试空结果处理。"""
        self.mock_es.search.return_value = {
            "hits": {"max_score": None, "hits": []}
        }
        
        results = self.store.search("不存在的内容")
        
        self.assertEqual(results, [])
    
    def test_delete_index(self) -> None:
        """测试删除索引。"""
        self.store.delete_index()
        
        self.mock_es.indices.delete.assert_called_once_with(index="test_index")
```

### 5.5 QueryFormatter 测试用例

**文件**: `tests/test_query_formatter.py`

```python
class TestQueryFormatter(unittest.TestCase):
    """查询格式化服务测试。"""
    
    def test_init_requires_api_key(self) -> None:
        """测试初始化必须提供 API 密钥。"""
        with self.assertRaises(ValueError):
            QueryFormatter(api_key="", model_name="test", base_url="test")
    
    def test_format_query_extracts_scene(self) -> None:
        """测试提取场景信息。"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=json.dumps({
            "scene": "公园",
            "objects": ["树木", "草地"],
            "atmosphere": "宁静",
            "time_hint": None,
            "season": None,
            "time_period": "白天",
            "search_text": "公园里有树木和草地，氛围宁静",
        })))]
        mock_client.chat.completions.create.return_value = mock_response
        
        formatter = QueryFormatter(
            api_key="test-key",
            model_name="test-model",
            base_url="https://test.com",
            client=mock_client,
        )
        
        result = formatter.format_query("请展示一张公园的照片")
        
        self.assertEqual(result["scene"], "公园")
        self.assertIn("公园", result["search_text"])
        self.assertEqual(result["original_query"], "请展示一张公园的照片")
    
    def test_format_query_fallback_on_error(self) -> None:
        """测试 API 失败时降级为原始查询。"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API 错误")
        
        formatter = QueryFormatter(
            api_key="test-key",
            model_name="test-model",
            base_url="https://test.com",
            max_retries=1,
            client=mock_client,
        )
        
        result = formatter.format_query("测试查询")
        
        self.assertEqual(result["search_text"], "测试查询")
        self.assertEqual(result["original_query"], "测试查询")
    
    def test_is_enabled(self) -> None:
        """测试服务状态检查。"""
        mock_client = Mock()
        
        formatter = QueryFormatter(
            api_key="test-key",
            model_name="test-model",
            base_url="https://test.com",
            client=mock_client,
        )
        
        self.assertTrue(formatter.is_enabled())
```

### 5.6 Searcher 混合检索测试用例

**文件**: `tests/test_searcher.py` (新增测试)

```python
class TestSearcherHybridSearch(unittest.TestCase):
    """混合检索测试。"""
    
    def test_hybrid_search_combines_scores(self) -> None:
        """测试混合检索正确融合向量和关键字分数。"""
        # Mock 依赖
        mock_embedding = Mock()
        mock_embedding.generate_embedding.return_value = [0.1] * 4096
        
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = [
            {"metadata": {"photo_path": "/a.jpg", "description": "测试A"}, "distance": 0.9},
            {"metadata": {"photo_path": "/b.jpg", "description": "测试B"}, "distance": 0.7},
        ]
        mock_vector_store.get_total_items.return_value = 100
        mock_vector_store.load.return_value = True
        mock_vector_store.dimension = 4096
        
        mock_keyword_store = Mock()
        mock_keyword_store.search.return_value = [
            {"photo_path": "/a.jpg", "score": 0.8},
            {"photo_path": "/c.jpg", "score": 0.6},
        ]
        
        mock_time_parser = Mock()
        mock_time_parser.extract_time_constraints.return_value = {
            "start_date": None, "end_date": None, "precision": "none"
        }
        
        searcher = Searcher(
            embedding=mock_embedding,
            time_parser=mock_time_parser,
            vector_store=mock_vector_store,
            keyword_store=mock_keyword_store,
            vector_weight=0.8,
            keyword_weight=0.2,
        )
        searcher.index_loaded = True
        
        results = searcher.search("测试查询", top_k=10)
        
        # 验证结果包含向量和关键字两边的照片
        paths = [r["photo_path"] for r in results]
        self.assertIn("/a.jpg", paths)  # 两边都有
    
    def test_hybrid_search_degrades_without_keyword_store(self) -> None:
        """测试无 KeywordStore 时降级为纯向量检索。"""
        mock_embedding = Mock()
        mock_embedding.generate_embedding.return_value = [0.1] * 512
        
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = [
            {"metadata": {"photo_path": "/a.jpg", "description": "测试"}, "distance": 0.9},
        ]
        mock_vector_store.get_total_items.return_value = 10
        mock_vector_store.load.return_value = True
        mock_vector_store.dimension = 512
        
        mock_time_parser = Mock()
        
        searcher = Searcher(
            embedding=mock_embedding,
            time_parser=mock_time_parser,
            vector_store=mock_vector_store,
            keyword_store=None,  # 不传入 KeywordStore
        )
        searcher.index_loaded = True
        
        results = searcher.search("测试查询", top_k=10)
        
        # 应该正常返回结果
        self.assertEqual(len(results), 1)
    
    def test_weight_validation(self) -> None:
        """测试权重必须和为 1。"""
        mock_embedding = Mock()
        mock_vector_store = Mock()
        mock_time_parser = Mock()
        
        with self.assertRaises(ValueError) as context:
            Searcher(
                embedding=mock_embedding,
                time_parser=mock_time_parser,
                vector_store=mock_vector_store,
                vector_weight=0.5,
                keyword_weight=0.3,  # 0.5 + 0.3 != 1
            )
        
        self.assertIn("必须等于 1.0", str(context.exception))
```

### 5.7 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定测试文件
python -m pytest tests/test_embedding_service.py -v

# 运行带覆盖率报告
python -m pytest tests/ --cov=. --cov-report=html

# 跳过集成测试（无 API Key 时）
python -m pytest tests/ -v -k "not integration"
```

---

## 六、文档维护规范

### 6.1 文档的重要性

> **警告**: 缺乏文档的代码修改会导致：
> 1. 后续维护者难以理解修改意图
> 2. 接口变更导致依赖方出错
> 3. 重复踩坑，浪费时间

### 6.2 修改前必须

1. **阅读本文档**：了解当前修改计划和接口定义
2. **阅读相关测试**：理解预期行为
3. **运行现有测试**：确保基线通过

### 6.3 修改过程中必须

1. **同步更新测试**：每修改一个接口，同步更新测试用例
2. **保持接口兼容**：除非明确需要破坏性变更，否则保持向后兼容
3. **记录变更**：在本文档的相应章节记录实际修改内容

### 6.4 修改完成后必须

1. **运行全部测试**：`python -m pytest tests/ -v`
2. **更新本文档**：
   - 将 "修改状态" 更新为 "已完成"
   - 记录实际修改与计划的差异
   - 更新 "最后更新" 日期
3. **提交规范**：
   ```
   feat(search): 实现混合检索策略
   
   - 新增 KeywordStore 类，基于 Elasticsearch 实现 BM25 检索
   - 修改 Searcher.search()，融合向量和关键字评分
   - 权重配置：向量 80%，关键字 20%
   
   BREAKING CHANGE: 需要重建索引
   ```

### 6.5 文档模板

新增模块时，使用以下模板记录：

```markdown
## 模块名称

### 修改背景
[简述为什么需要这个修改]

### 涉及文件
[列出所有涉及的文件]

### 接口定义
[详细的接口签名和参数说明]

### 测试用例
[关键测试用例列表]

### 注意事项
[使用注意事项和已知限制]
```

---

## 附录

### A. 环境变量完整清单

```bash
# 现有配置
PHOTO_DIR=/path/to/photos
DATA_DIR=./data
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
VISION_MODEL_NAME=openai/gpt-4o
TIME_PARSE_MODEL_NAME=openai/gpt-3.5-turbo
SERVER_HOST=localhost
SERVER_PORT=5000

# 新增：火山引擎 Embedding（替换原有嵌入服务）
VOLCANO_API_KEY=your_volcano_key
VOLCANO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
VOLCANO_EMBEDDING_MODEL=doubao-embedding-large-text-240915
EMBEDDING_DIMENSION=4096

# 新增：Elasticsearch
ELASTICSEARCH_HOST=localhost
ELASTICSEARCH_PORT=9200
ELASTICSEARCH_INDEX=photo_keywords
ELASTICSEARCH_USERNAME=elastic
ELASTICSEARCH_PASSWORD=your_password

# 新增：查询格式化 LLM
QUERY_FORMAT_API_KEY=your_query_format_key
QUERY_FORMAT_BASE_URL=https://your-llm-endpoint/api/v1
QUERY_FORMAT_MODEL=your-model-name

# 新增：混合检索权重
VECTOR_WEIGHT=0.8
KEYWORD_WEIGHT=0.2
```

### B. 依赖更新

`requirements.txt` 新增：

```
elasticsearch>=8.0.0
```

### C. 迁移步骤

1. 安装新依赖：`pip install -r requirements.txt`
2. 配置环境变量（参考附录 A）
3. 启动 Elasticsearch 服务
4. **删除旧索引**：`rm -rf ./data/*.index ./data/*.json`
5. 重新构建索引：访问 `/init_index`
6. 验证搜索功能

---

**文档结束**
