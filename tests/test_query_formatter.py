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
        """测试提取画面语义片段，不做发散改写。"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=json.dumps({
            "time_hint": None,
            "season": None,
            "time_period": None,
            "search_text": "公园",
            "media_terms": [],
            "identity_terms": [],
        })))]
        mock_client.chat.completions.create.return_value = mock_response
        
        formatter = QueryFormatter(
            api_key="test-key",
            model_name="test-model",
            base_url="https://test.com",
            client=mock_client,
        )
        
        result = formatter.format_query("请展示一张公园的照片")
        
        self.assertEqual(result["search_text"], "公园")
        self.assertEqual(result["media_terms"], [])
        self.assertEqual(result["identity_terms"], [])
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
        self.assertEqual(result["media_terms"], [])
        self.assertEqual(result["identity_terms"], [])
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
        """测试 search_text 只保留原 query 的视觉语义片段。"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=json.dumps({
            "time_hint": "2024年",
            "season": "夏天",
            "time_period": "下午",
            "search_text": "海滩",
            "media_terms": [],
            "identity_terms": [],
        })))]
        mock_client.chat.completions.create.return_value = mock_response
        
        formatter = QueryFormatter(
            api_key="test-key",
            model_name="test-model",
            base_url="https://test.com",
            client=mock_client,
        )
        
        result = formatter.format_query("2024年夏天下午的海滩照片")
        
        self.assertEqual(result["search_text"], "海滩")
        
        # 验证时间信息作为独立字段返回
        self.assertEqual(result["time_hint"], "2024年")
        self.assertEqual(result["season"], "夏天")
        self.assertEqual(result["time_period"], "下午")
        
    def test_time_period_seven_segments(self) -> None:
        """测试7档时段细分支持。"""
        mock_client = Mock()
        
        time_periods = ["凌晨", "早晨", "上午", "中午", "下午", "傍晚", "夜晚"]
        
        for period in time_periods:
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content=json.dumps({
                "time_hint": None,
                "season": None,
                "time_period": period,
                "search_text": "街道",
                "media_terms": [],
                "identity_terms": [],
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
            self.assertEqual(result["search_text"], "街道")
    
    def test_no_time_info_returns_null(self) -> None:
        """测试无时间信息时返回 null，不编造时间字段。"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=json.dumps({
            "time_hint": None,
            "season": None,
            "time_period": None,
            "search_text": "山顶",
            "media_terms": [],
            "identity_terms": [],
        })))]
        mock_client.chat.completions.create.return_value = mock_response
        
        formatter = QueryFormatter(
            api_key="test-key",
            model_name="test-model",
            base_url="https://test.com",
            client=mock_client,
        )
        
        result = formatter.format_query("山顶的风景照")
        
        # 验证时间字段为 null
        self.assertIsNone(result["time_hint"])
        self.assertIsNone(result["season"])
        self.assertIsNone(result["time_period"])
        
        self.assertEqual(result["search_text"], "山顶")
    
    def test_search_text_comes_directly_from_query(self) -> None:
        """测试 search_text 直接来自 query 的视觉片段，不发散。"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=json.dumps({
            "time_hint": "去年",
            "season": "夏天",
            "time_period": None,
            "search_text": "海边 全家福",
            "media_terms": [],
            "identity_terms": [],
        })))]
        mock_client.chat.completions.create.return_value = mock_response
        
        formatter = QueryFormatter(
            api_key="test-key",
            model_name="test-model",
            base_url="https://test.com",
            client=mock_client,
        )
        
        result = formatter.format_query("请帮我找一下去年夏天在海边拍的全家福")
        
        self.assertEqual(result["search_text"], "海边 全家福")
        
        self.assertEqual(result["time_hint"], "去年")
        self.assertEqual(result["season"], "夏天")

    def test_format_query_passes_reasoning_effort(self) -> None:
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=json.dumps({
            "time_hint": None,
            "season": None,
            "time_period": None,
            "search_text": "街道 夜景",
            "media_terms": [],
            "identity_terms": [],
        })))]
        mock_client.chat.completions.create.return_value = mock_response

        formatter = QueryFormatter(
            api_key="test-key",
            model_name="test-model",
            base_url="https://test.com",
            reasoning_effort="medium",
            client=mock_client,
        )

        formatter.format_query("夜晚街道照片")
        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["extra_body"], {"reasoning_effort": "medium"})

    def test_prompt_requires_non_divergent_extraction(self) -> None:
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=json.dumps({
            "search_text": "频谱分析仪 屏幕",
            "media_terms": ["screen"],
            "identity_terms": [],
            "time_hint": None,
            "season": None,
            "time_period": None,
        })))]
        mock_client.chat.completions.create.return_value = mock_response

        formatter = QueryFormatter(
            api_key="test-key",
            model_name="test-model",
            base_url="https://test.com",
            client=mock_client,
        )

        formatter.format_query("频谱分析仪 屏幕")
        _, kwargs = mock_client.chat.completions.create.call_args
        messages = kwargs["messages"]
        self.assertIn("照片搜索查询理解器", messages[0]["content"])
        self.assertIn("不能编造用户没有表达的实体", messages[0]["content"])
        self.assertIn("不要把“海边”发散成“沙滩海浪蓝天”", messages[1]["content"])

    def test_extracts_media_and_identity_terms(self) -> None:
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=json.dumps({
            "search_text": "",
            "media_terms": ["stage_performance"],
            "identity_terms": ["周杰伦"],
            "strict_identity_filter": True,
            "time_hint": None,
            "season": None,
            "time_period": None,
        })))]
        mock_client.chat.completions.create.return_value = mock_response

        formatter = QueryFormatter(
            api_key="test-key",
            model_name="test-model",
            base_url="https://test.com",
            client=mock_client,
        )

        result = formatter.format_query("周杰伦演唱会")
        self.assertEqual(result["media_terms"], ["stage_performance"])
        self.assertEqual(result["identity_terms"], ["周杰伦"])
        self.assertTrue(result["strict_identity_filter"])

    def test_expand_query_intents_returns_alternatives(self) -> None:
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=json.dumps({
            "alternatives": [
                {
                    "search_text": "舞台男歌手 现场演出",
                    "media_terms": ["stage_performance"],
                    "identity_terms": ["陶喆"],
                    "strict_identity_filter": True,
                    "reason": "补充现场语义"
                },
                {
                    "search_text": "华语男歌手 演唱会大屏",
                    "media_terms": ["stage_performance", "screen"],
                    "identity_terms": ["陶喆"],
                    "strict_identity_filter": False,
                    "reason": "补充常见拍摄载体"
                }
            ]
        })))]
        mock_client.chat.completions.create.return_value = mock_response

        formatter = QueryFormatter(
            api_key="test-key",
            model_name="test-model",
            base_url="https://test.com",
            client=mock_client,
        )

        result = formatter.expand_query_intents(
            user_query="请帮我找陶喆的照片",
            base_intent={
                "search_text": "",
                "media_terms": [],
                "identity_terms": ["陶喆"],
                "strict_identity_filter": True,
                "time_hint": None,
                "season": None,
                "time_period": None,
                "original_query": "请帮我找陶喆的照片",
            },
            max_alternatives=2,
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["identity_terms"], ["陶喆"])
        self.assertEqual(result[1]["media_terms"], ["stage_performance", "screen"])

    def test_reflect_on_weak_results_returns_refined_intent(self) -> None:
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=json.dumps({
            "search_text": "舞台特写 男歌手 近景",
            "media_terms": ["stage_performance"],
            "identity_terms": ["陶喆"],
            "strict_identity_filter": True,
            "reason": "前两轮结果大多只有泛化男歌手，需要更强调近景舞台人像",
        })))]
        mock_client.chat.completions.create.return_value = mock_response

        formatter = QueryFormatter(
            api_key="test-key",
            model_name="test-model",
            base_url="https://test.com",
            client=mock_client,
        )

        result = formatter.reflect_on_weak_results(
            user_query="请帮我找陶喆的照片",
            base_intent={
                "search_text": "",
                "media_terms": [],
                "identity_terms": ["陶喆"],
                "strict_identity_filter": True,
                "time_hint": None,
                "season": None,
                "time_period": None,
                "original_query": "请帮我找陶喆的照片",
            },
            weak_results=[
                {"photo_path": "/tmp/a.jpg", "description": "男歌手舞台远景", "score": 0.41},
                {"photo_path": "/tmp/b.jpg", "description": "舞台大屏", "score": 0.39},
            ],
        )

        self.assertEqual(result["search_text"], "舞台特写 男歌手 近景")
        self.assertEqual(result["media_terms"], ["stage_performance"])
        self.assertEqual(result["identity_terms"], ["陶喆"])
        self.assertTrue(result["strict_identity_filter"])
        self.assertIn("更强调近景舞台人像", result["reason"])
