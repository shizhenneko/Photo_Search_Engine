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
            scheme = "http"
            url = f"{scheme}://{host}:{port}"
            if username and password:
                self.es_client = Elasticsearch(url, basic_auth=(username, password))
            else:
                self.es_client = Elasticsearch(url)
        
        self._ensure_index()

    def _description_mapping_uses_ik(self) -> bool:
        try:
            mapping = self.es_client.indices.get_mapping(index=self.index_name)
            properties = (
                mapping.get(self.index_name, {})
                .get("mappings", {})
                .get("properties", {})
            )
            description = properties.get("description", {})
            return (
                description.get("analyzer") == "ik_max_word"
                and description.get("search_analyzer") == "ik_smart"
            )
        except Exception:
            return False

    def _has_ik_analyzer(self) -> bool:
        try:
            response = self.es_client.indices.analyze(
                body={
                    "analyzer": "ik_smart",
                    "text": "照片搜索测试",
                }
            )
            return bool(response.get("tokens"))
        except Exception:
            return False

    def _apply_runtime_settings(self) -> None:
        """
        自愈单机 Elasticsearch 的索引副本配置。

        本项目默认以单机本地开发为主，副本数必须保持为 0，
        否则索引会因为未分配副本长期处于 yellow/red，进而影响
        关键字检索与重建流程。
        """
        try:
            self.es_client.indices.put_settings(
                index=self.index_name,
                body={
                    "index": {
                        "number_of_replicas": 0,
                    }
                },
            )
        except Exception:
            # 运行时自愈失败不应阻塞主流程，后续搜索会自动降级。
            pass

    @staticmethod
    def _build_text_query(query: Optional[str]) -> Optional[Dict[str, Any]]:
        if not query or not query.strip():
            return None
        text = query.strip()
        return {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": text,
                            "fields": [
                                "retrieval_text^3.0",
                                "description^2.0",
                                "inner_content_summary^1.8",
                                "outer_scene_summary^1.2",
                                "ocr_text^1.5",
                                "file_name^0.8",
                            ],
                            "type": "best_fields",
                            "minimum_should_match": "60%",
                        }
                    },
                    {
                        "match_phrase": {
                            "retrieval_text": {
                                "query": text,
                                "boost": 1.4,
                            }
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        }
    
    def _ensure_index(self) -> None:
        """
        确保索引存在，不存在则创建。
        
        索引 Mapping 配置：
        - photo_path: keyword（精确匹配）
        - description: text（中文分词，BM25）
        - outer_scene_summary / inner_content_summary / retrieval_text / ocr_text: text
        - file_name: text（文件名模糊匹配）
        - media_types / tags / identity_names / identity_evidence: keyword
        - EXIF 独立字段（用于精确过滤）：
          - year: integer
          - month: integer
          - day: integer
          - hour: integer
          - season: keyword
          - time_period: keyword
          - weekday: keyword
          - camera: keyword
          - datetime: date
        """
        if self.es_client.indices.exists(index=self.index_name):
            if self._has_ik_analyzer() and not self._description_mapping_uses_ik():
                self.es_client.indices.delete(index=self.index_name)
            else:
                self._apply_runtime_settings()
                return

        if not self.es_client.indices.exists(index=self.index_name):
            use_ik = self._has_ik_analyzer()
            mapping = {
                "mappings": {
                    "properties": {
                        "photo_path": {"type": "keyword"},
                        "description": {
                            "type": "text",
                        },
                        "outer_scene_summary": {"type": "text"},
                        "inner_content_summary": {"type": "text"},
                        "retrieval_text": {"type": "text"},
                        "ocr_text": {"type": "text"},
                        "file_name": {"type": "text"},
                        "media_types": {"type": "keyword"},
                        "tags": {"type": "keyword"},
                        "identity_names": {"type": "keyword"},
                        "identity_evidence": {"type": "keyword"},
                        # EXIF 独立字段（新增，用于精确过滤）
                        "year": {"type": "integer"},
                        "month": {"type": "integer"},
                        "day": {"type": "integer"},
                        "hour": {"type": "integer"},
                        "season": {"type": "keyword"},
                        "time_period": {"type": "keyword"},
                        "weekday": {"type": "keyword"},
                        "camera": {"type": "keyword"},
                        "datetime": {
                            "type": "date",
                            "format": "strict_date_optional_time||epoch_millis",
                            "ignore_malformed": True,
                        },
                    }
                },
                "settings": {
                    "index": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                    }
                },
            }
            if use_ik:
                mapping["mappings"]["properties"]["description"]["analyzer"] = "ik_max_word"
                mapping["mappings"]["properties"]["description"]["search_analyzer"] = "ik_smart"
            self.es_client.indices.create(index=self.index_name, body=mapping)
            self._apply_runtime_settings()
    
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
        query_clause = self._build_text_query(query)
        if query_clause is None:
            return []
        body = {
            "query": query_clause,
            "size": top_k,
            "min_score": 0.12,
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

    def search_with_filters(
        self,
        query: Optional[str],
        filters: Dict[str, Any],
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        执行带 EXIF 条件过滤的搜索。

        Args:
            query (Optional[str]): 查询文本（可为空，仅做过滤）
            filters (Dict[str, Any]): 过滤条件
                - year: int（年份）
                - month: int（月份）
                - day: int（日期）
                - season: str（季节：春天/夏天/秋天/冬天）
                - time_period: str（时段：凌晨/早晨/上午/中午/下午/傍晚/夜晚）
                - weekday: str（星期）
                - camera: str（相机型号）
                - start_date: str（起始日期，ISO格式）
                - end_date: str（结束日期，ISO格式）
            top_k (int): 返回数量

        Returns:
            List[Dict[str, Any]]: 搜索结果列表
                - photo_path: str
                - score: float（归一化分数）
        """
        must_clauses = []
        filter_clauses = []

        # 文本查询（如果有）
        text_query = self._build_text_query(query)
        if text_query is not None:
            must_clauses.append(text_query)

        # 精确匹配过滤条件
        exact_fields = ["year", "month", "day", "hour", "season", "time_period", "weekday", "camera"]
        
        for field in exact_fields:
            value = filters.get(field)
            if value is not None:
                filter_clauses.append({"term": {field: value}})

        # 日期范围过滤
        start_date = filters.get("start_date")
        end_date = filters.get("end_date")
        if start_date or end_date:
            date_range: Dict[str, Any] = {}
            if start_date:
                date_range["gte"] = start_date
            if end_date:
                date_range["lte"] = end_date
            filter_clauses.append({"range": {"datetime": date_range}})

        # 构建查询体
        if not must_clauses and not filter_clauses:
            # 无条件时返回所有
            body = {"query": {"match_all": {}}, "size": top_k}
        elif must_clauses and filter_clauses:
            body = {
                "query": {
                    "bool": {
                        "must": must_clauses,
                        "filter": filter_clauses,
                    }
                },
                "size": top_k,
            }
        elif must_clauses:
            body = {
                "query": {"bool": {"must": must_clauses}},
                "size": top_k,
            }
        else:
            # 仅有过滤条件
            body = {
                "query": {"bool": {"filter": filter_clauses}},
                "size": top_k,
            }

        try:
            if must_clauses:
                body["min_score"] = 0.12
            response = self.es_client.search(index=self.index_name, body=body)
            hits = response["hits"]["hits"]

            if not hits:
                return []

            # 归一化分数
            max_score = response["hits"]["max_score"] or 1.0
            if max_score == 0:
                max_score = 1.0

            results = []
            for hit in hits:
                score = hit["_score"] / max_score if hit["_score"] else 0.5
                results.append({
                    "photo_path": hit["_source"]["photo_path"],
                    "score": score,
                })

            return results
        except Exception as e:
            print(f"ES search_with_filters failed: {e}")
            return []

    def get_filtered_paths(self, filters: Dict[str, Any], top_k: int = 1000) -> List[str]:
        """
        获取满足过滤条件的照片路径列表（不进行文本搜索）。

        Args:
            filters (Dict[str, Any]): 过滤条件（同 search_with_filters）
            top_k (int): 返回数量

        Returns:
            List[str]: 满足条件的照片路径列表
        """
        results = self.search_with_filters(None, filters, top_k)
        return [r["photo_path"] for r in results]
    
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

    def clear(self) -> None:
        """清空索引并重新初始化。"""
        self.delete_index()
        self._ensure_index()
