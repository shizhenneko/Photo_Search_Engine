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
    
    def test_search_text_no_time_info(self) -> None:
        """测试 search_text 不包含时间信息（架构改进）。"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=json.dumps({
            "scene": "海滩",
            "objects": ["沙滩", "海浪"],
            "atmosphere": "宁静",
            "time_hint": "2024年",
            "season": "夏天",
            "time_period": "下午",
            "search_text": "海滩上有沙滩和海浪，氛围宁静",  # 纯视觉描述，无时间信息
        })))]
        mock_client.chat.completions.create.return_value = mock_response
        
        formatter = QueryFormatter(
            api_key="test-key",
            model_name="test-model",
            base_url="https://test.com",
            client=mock_client,
        )
        
        result = formatter.format_query("2024年夏天下午的海滩照片")
        
        # 验证 search_text 不包含时间信息
        self.assertNotIn("2024", result["search_text"])
        self.assertNotIn("夏天", result["search_text"])
        self.assertNotIn("下午", result["search_text"])
        self.assertNotIn("时间", result["search_text"])
        self.assertNotIn("季节", result["search_text"])
        self.assertNotIn("时段", result["search_text"])
        
        # 验证时间信息作为独立字段返回
        self.assertEqual(result["time_hint"], "2024年")
        self.assertEqual(result["season"], "夏天")
        self.assertEqual(result["time_period"], "下午")
        
        # 验证 search_text 包含纯视觉内容
        self.assertIn("海滩", result["search_text"])
    
    def test_time_period_seven_segments(self) -> None:
        """测试7档时段细分支持。"""
        mock_client = Mock()
        
        time_periods = ["凌晨", "早晨", "上午", "中午", "下午", "傍晚", "夜晚"]
        
        for period in time_periods:
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content=json.dumps({
                "scene": "街道",
                "objects": ["建筑"],
                "atmosphere": "热闹",
                "time_hint": None,
                "season": None,
                "time_period": period,
                "search_text": "城市街道上有建筑，氛围热闹",
            })))]
            mock_client.chat.completions.create.return_value = mock_response
            
            formatter = QueryFormatter(
                api_key="test-key",
                model_name="test-model",
                base_url="https://test.com",
                client=mock_client,
            )
            
            result = formatter.format_query(f"{period}的街道照片")
            
            # 验证时段正确提取
            self.assertEqual(result["time_period"], period)
            # 验证 search_text 不包含时段信息
            self.assertNotIn(period, result["search_text"])
