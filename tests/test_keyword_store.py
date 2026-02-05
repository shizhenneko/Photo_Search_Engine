import unittest
from unittest.mock import Mock, patch, call
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

    def test_add_document_with_exif_fields(self) -> None:
        """测试添加包含 EXIF 独立字段的文档。"""
        document = {
            "photo_path": "/photos/sunset.jpg",
            "description": "海边日落照片",
            "file_name": "sunset.jpg",
            "year": 2024,
            "month": 6,
            "day": 15,
            "hour": 18,
            "season": "夏天",
            "time_period": "傍晚",
            "weekday": "星期六",
            "camera": "iPhone 15",
            "datetime": "2024-06-15T18:30:00",
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


class TestKeywordStoreSearchWithFilters(unittest.TestCase):
    """测试带 EXIF 过滤的搜索功能。"""

    def setUp(self) -> None:
        """设置 Mock 客户端。"""
        self.mock_es = Mock()
        self.mock_es.indices.exists.return_value = True
        self.store = KeywordStore(
            index_name="test_index",
            client=self.mock_es,
        )

    def test_search_with_season_filter(self) -> None:
        """测试按季节过滤搜索。"""
        self.mock_es.search.return_value = {
            "hits": {
                "max_score": 8.0,
                "hits": [
                    {"_source": {"photo_path": "/summer1.jpg"}, "_score": 8.0},
                    {"_source": {"photo_path": "/summer2.jpg"}, "_score": 6.0},
                ],
            }
        }

        results = self.store.search_with_filters(
            query="海边",
            filters={"season": "夏天"},
            top_k=10,
        )

        self.assertEqual(len(results), 2)
        # 验证 ES 调用包含过滤条件
        call_args = self.mock_es.search.call_args
        body = call_args.kwargs.get("body") or call_args[1].get("body")
        self.assertIn("bool", body["query"])

    def test_search_with_time_period_filter(self) -> None:
        """测试按时段过滤搜索。"""
        self.mock_es.search.return_value = {
            "hits": {
                "max_score": 5.0,
                "hits": [
                    {"_source": {"photo_path": "/evening1.jpg"}, "_score": 5.0},
                ],
            }
        }

        results = self.store.search_with_filters(
            query="日落",
            filters={"time_period": "傍晚"},
            top_k=10,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["photo_path"], "/evening1.jpg")

    def test_search_with_year_month_filter(self) -> None:
        """测试按年月过滤搜索。"""
        self.mock_es.search.return_value = {
            "hits": {
                "max_score": 7.0,
                "hits": [
                    {"_source": {"photo_path": "/june2024.jpg"}, "_score": 7.0},
                ],
            }
        }

        results = self.store.search_with_filters(
            query="旅行",
            filters={"year": 2024, "month": 6},
            top_k=10,
        )

        self.assertEqual(len(results), 1)

    def test_search_with_date_range_filter(self) -> None:
        """测试按日期范围过滤搜索。"""
        self.mock_es.search.return_value = {
            "hits": {
                "max_score": 9.0,
                "hits": [
                    {"_source": {"photo_path": "/vacation.jpg"}, "_score": 9.0},
                ],
            }
        }

        results = self.store.search_with_filters(
            query="度假",
            filters={
                "start_date": "2024-06-01",
                "end_date": "2024-06-30",
            },
            top_k=10,
        )

        self.assertEqual(len(results), 1)
        # 验证包含 range 查询
        call_args = self.mock_es.search.call_args
        body = call_args.kwargs.get("body") or call_args[1].get("body")
        self.assertIn("bool", body["query"])

    def test_search_with_no_query_only_filters(self) -> None:
        """测试仅使用过滤条件，无文本查询。"""
        self.mock_es.search.return_value = {
            "hits": {
                "max_score": None,
                "hits": [
                    {"_source": {"photo_path": "/winter1.jpg"}, "_score": 1.0},
                    {"_source": {"photo_path": "/winter2.jpg"}, "_score": 1.0},
                ],
            }
        }

        results = self.store.search_with_filters(
            query=None,
            filters={"season": "冬天"},
            top_k=10,
        )

        self.assertEqual(len(results), 2)

    def test_search_with_empty_filters(self) -> None:
        """测试空过滤条件返回所有结果。"""
        self.mock_es.search.return_value = {
            "hits": {
                "max_score": 5.0,
                "hits": [
                    {"_source": {"photo_path": "/all1.jpg"}, "_score": 5.0},
                ],
            }
        }

        results = self.store.search_with_filters(
            query="风景",
            filters={},
            top_k=10,
        )

        self.assertEqual(len(results), 1)

    def test_get_filtered_paths(self) -> None:
        """测试获取满足过滤条件的照片路径列表。"""
        self.mock_es.search.return_value = {
            "hits": {
                "max_score": None,
                "hits": [
                    {"_source": {"photo_path": "/a.jpg"}, "_score": 1.0},
                    {"_source": {"photo_path": "/b.jpg"}, "_score": 1.0},
                ],
            }
        }

        paths = self.store.get_filtered_paths(
            filters={"year": 2024},
            top_k=100,
        )

        self.assertEqual(len(paths), 2)
        self.assertIn("/a.jpg", paths)
        self.assertIn("/b.jpg", paths)

    def test_search_with_combined_filters(self) -> None:
        """测试多条件组合过滤。"""
        self.mock_es.search.return_value = {
            "hits": {
                "max_score": 6.0,
                "hits": [
                    {"_source": {"photo_path": "/perfect.jpg"}, "_score": 6.0},
                ],
            }
        }

        results = self.store.search_with_filters(
            query="日落",
            filters={
                "year": 2024,
                "season": "夏天",
                "time_period": "傍晚",
            },
            top_k=10,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["photo_path"], "/perfect.jpg")


if __name__ == "__main__":
    unittest.main()
