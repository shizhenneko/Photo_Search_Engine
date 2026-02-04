import json
import unittest
from unittest.mock import Mock, patch

from utils.query_formatter import QueryFormatter


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
