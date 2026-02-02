import os
import sys
import unittest
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import get_config

from utils.time_parser import TimeParser


class TimeParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """加载配置"""
        cls.config = get_config()
        cls.has_api_key = bool(cls.config.get("OPENROUTER_API_KEY"))

    def test_time_parser_requires_api_key(self) -> None:
        """测试TimeParser需要API密钥"""
        with self.assertRaises(ValueError):
            TimeParser(api_key="")

    @unittest.skipIf(
        not bool(os.getenv("OPENROUTER_API_KEY")),
        "OPENROUTER_API_KEY未设置，跳过集成测试"
    )
    def test_time_parser_init(self) -> None:
        """测试TimeParser初始化（集成测试）"""
        api_key = self.config["OPENROUTER_API_KEY"]
        base_url = self.config.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        model_name = self.config.get("TIME_PARSE_MODEL_NAME", "openai/gpt-3.5-turbo")

        parser = TimeParser(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
        )

        self.assertEqual(parser.api_key, api_key)
        self.assertEqual(parser.model_name, model_name)
        self.assertEqual(parser.base_url, base_url)

    @unittest.skipIf(
        not bool(os.getenv("OPENROUTER_API_KEY")),
        "OPENROUTER_API_KEY未设置，跳过集成测试"
    )
    def test_extract_time_constraints_without_time(self) -> None:
        """测试无时间约束查询（集成测试）"""
        api_key = self.config["OPENROUTER_API_KEY"]
        base_url = self.config.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        parser = TimeParser(api_key=api_key, base_url=base_url)

        result = parser.extract_time_constraints("海边的照片")

        self.assertIsNone(result.get("start_date"))
        self.assertIsNone(result.get("end_date"))
        self.assertEqual(result.get("precision"), "none")

    @unittest.skipIf(
        not bool(os.getenv("OPENROUTER_API_KEY")),
        "OPENROUTER_API_KEY未设置，跳过集成测试"
    )
    def test_extract_time_constraints_with_year(self) -> None:
        """测试年份时间约束（集成测试）"""
        api_key = self.config["OPENROUTER_API_KEY"]
        base_url = self.config.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        parser = TimeParser(api_key=api_key, base_url=base_url)

        result = parser.extract_time_constraints("去年的照片")

        self.assertIsNotNone(result.get("start_date"))
        self.assertIsNotNone(result.get("end_date"))
        self.assertIn(result.get("precision"), ["year", "season", "month", "range"])

    @unittest.skipIf(
        not bool(os.getenv("OPENROUTER_API_KEY")),
        "OPENROUTER_API_KEY未设置，跳过集成测试"
    )
    def test_extract_time_constraints_with_season(self) -> None:
        """测试季节时间约束（集成测试）"""
        api_key = self.config["OPENROUTER_API_KEY"]
        base_url = self.config.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        parser = TimeParser(api_key=api_key, base_url=base_url)

        result = parser.extract_time_constraints("冬天的风景")

        self.assertIsNotNone(result.get("start_date"))
        self.assertIsNotNone(result.get("end_date"))
        self.assertEqual(result.get("precision"), "season")


if __name__ == "__main__":
    unittest.main()
