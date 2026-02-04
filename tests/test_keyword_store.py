import unittest
from unittest.mock import Mock, patch
from utils.keyword_store import KeywordStore

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

if __name__ == "__main__":
    unittest.main()
