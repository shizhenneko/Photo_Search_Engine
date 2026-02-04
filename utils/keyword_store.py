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
            scheme = "https" if username and password else "http"
            url = f"{scheme}://{host}:{port}"
            if auth:
                self.es_client = Elasticsearch(url, basic_auth=auth)
            else:
                self.es_client = Elasticsearch(url)
        
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
            # 注意: ik_max_word 需要 ES 安装 IK 分词显卡，如果没有安装可能会报错。
            # 为了兼容性，如果没有 IK，ES 会报错。这里假设用户已配置好环境或后续处理。
            # 如果是本地简单测试，可以用 standard analyzer 作为 fallback，但为了中文效果保留 ik。
            try:
                self.es_client.indices.create(index=self.index_name, body=mapping)
            except Exception:
                # Fallback to standard analyzer if ik fails (optional robustness)
                fallback_mapping = mapping.copy()
                fallback_mapping["mappings"]["properties"]["description"].pop("analyzer")
                fallback_mapping["mappings"]["properties"]["description"].pop("search_analyzer")
                if not self.es_client.indices.exists(index=self.index_name):
                    self.es_client.indices.create(index=self.index_name, body=fallback_mapping)
    
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
        
        try:
            response = self.es_client.search(index=self.index_name, body=body)
            hits = response["hits"]["hits"]
            
            if not hits:
                return []
            
            # 归一化 BM25 分数到 0-1
            max_score = response["hits"]["max_score"] or 1.0
            if max_score == 0:
                max_score = 1.0
            
            results = []
            for hit in hits:
                results.append({
                    "photo_path": hit["_source"]["photo_path"],
                    "score": hit["_score"] / max_score,  # 归一化
                })
            
            return results
        except Exception as e:
            print(f"ES search failed: {e}")
            return []
    
    def delete_index(self) -> None:
        """删除整个索引（用于重建）。"""
        if self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.delete(index=self.index_name)
    
    def get_document_count(self) -> int:
        """获取索引中的文档数量。"""
        if self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.refresh(index=self.index_name)
            return int(self.es_client.count(index=self.index_name)["count"])
        return 0
