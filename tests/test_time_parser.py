import json
import unittest
from unittest.mock import Mock

from utils.time_parser import TimeParser


class TimeParserTests(unittest.TestCase):
    def test_time_parser_requires_api_key(self) -> None:
        with self.assertRaises(ValueError):
            TimeParser(api_key="", model_name="gpt-5.1", base_url="https://www.su8.codes/codex/v1")

    def test_has_time_terms(self) -> None:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content=json.dumps({"has_time_constraint": True})))]
        )
        parser = TimeParser(
            api_key="test-key",
            model_name="gpt-5.1",
            base_url="https://www.su8.codes/codex/v1",
            client=mock_client,
        )
        self.assertTrue(parser.has_time_terms("去年的照片"))
        mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content=json.dumps({"has_time_constraint": False})))]
        )
        self.assertFalse(parser.has_time_terms("海边的照片"))

    def test_extract_time_constraints_without_time(self) -> None:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content=json.dumps({"has_time_constraint": False})))]
        )
        parser = TimeParser(
            api_key="test-key",
            model_name="gpt-5.1",
            base_url="https://www.su8.codes/codex/v1",
            client=mock_client,
        )
        result = parser.extract_time_constraints("海边的照片")
        self.assertIsNone(result.get("start_date"))
        self.assertEqual(result.get("precision"), "none")

    def test_extract_time_constraints_with_year(self) -> None:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content=json.dumps({
                "has_time_constraint": True,
                "start_date": "2025-01-01",
                "end_date": "2025-12-31",
            })))]
        )
        parser = TimeParser(
            api_key="test-key",
            model_name="gpt-5.1",
            base_url="https://www.su8.codes/codex/v1",
            client=mock_client,
        )
        result = parser.extract_time_constraints("去年的照片")
        self.assertEqual(result["start_date"], "2025-01-01")
        self.assertEqual(result["precision"], "year")


if __name__ == "__main__":
    unittest.main()
