import os
import tempfile
import unittest
from unittest.mock import Mock

from PIL import Image

from utils.vision_llm_service import LocalVisionLLMService, SU8VisionLLMService


class VisionServiceTests(unittest.TestCase):
    def test_local_vision_service(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "local.jpg")
            image = Image.new("RGB", (80, 60), color=(1, 2, 3))
            image.save(path)
            service = LocalVisionLLMService()
            result = service.generate_description(path)
            self.assertIn("80x60", result)
            analysis = service.analyze_image(path)
            self.assertEqual(analysis["media_types"], ["photo"])
            self.assertIn("retrieval_text", analysis)

    def test_su8_vision_service_requires_api_key(self) -> None:
        with self.assertRaises(ValueError):
            SU8VisionLLMService(api_key="", model_name="gpt-5.4", base_url="https://www.su8.codes/codex/v1")

    def test_base64_image_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.jpg")
            image = Image.new("RGB", (2000, 1500), color=(100, 150, 200))
            image.save(path)

            service = SU8VisionLLMService(
                api_key="test-key",
                model_name="gpt-5.4",
                base_url="https://www.su8.codes/codex/v1",
                use_base64=True,
                image_max_size=1024,
                image_quality=85,
                image_format="WEBP",
            )
            base64_url = service._get_image_base64(path)
            self.assertIn("data:image/webp;base64,", base64_url)

    def test_generate_description(self) -> None:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content='{"description":"海边有沙滩和海浪","outer_scene_summary":"海边照片","inner_content_summary":"","media_types":["photo"],"tags":[{"tag":"海边","confidence":0.9}],"ocr_text":"","person_roles":[],"identity_candidates":[],"analysis_flags":{}}'))]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.jpg")
            Image.new("RGB", (200, 150), color=(100, 150, 200)).save(path)
            service = SU8VisionLLMService(
                api_key="test-key",
                model_name="gpt-5.4",
                base_url="https://www.su8.codes/codex/v1",
                client=mock_client,
            )
            result = service.generate_description(path)
            self.assertEqual(result, "海边有沙滩和海浪")
            analysis = service.analyze_image(path)
            self.assertEqual(analysis["media_types"], ["photo"])
            self.assertIn("海边", analysis["retrieval_text"])

    def test_analyze_image_exposes_internal_timing_metrics(self) -> None:
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [
            Mock(
                choices=[
                    Mock(
                        message=Mock(
                            content='{"description":"键盘速查表","outer_scene_summary":"文档照片","inner_content_summary":"vim 键位速查","media_types":["document"],"tags":[{"tag":"vim","confidence":0.98}],"ocr_text":"vim hjkl dd yy","person_roles":[],"identity_candidates":[],"analysis_flags":{"text_heavy":true}}'
                        )
                    )
                ]
            ),
            Mock(
                choices=[
                    Mock(
                        message=Mock(
                            content='{"description":"vim 键盘速查表","outer_scene_summary":"文档照片","inner_content_summary":"包含 vim 常用命令和键位说明的速查图","media_types":["document"],"tags":[{"tag":"vim","confidence":0.99},{"tag":"快捷键","confidence":0.91}],"ocr_text":"vim hjkl dd yy p u ctrl r","person_roles":[],"identity_candidates":[],"analysis_flags":{"text_heavy":true}}'
                        )
                    )
                ]
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sheet.jpg")
            Image.new("RGB", (400, 300), color=(100, 150, 200)).save(path)
            service = SU8VisionLLMService(
                api_key="test-key",
                model_name="gpt-5.4",
                base_url="https://www.su8.codes/codex/v1",
                client=mock_client,
            )

            analysis = service.analyze_image(path)
            metrics = service.get_last_analysis_metrics()

            self.assertEqual(analysis["media_types"], ["document"])
            self.assertIsNotNone(metrics)
            assert metrics is not None
            self.assertIn("image_encode_seconds", metrics)
            self.assertIn("attempts", metrics)
            self.assertEqual(len(metrics["attempts"]), 1)
            self.assertTrue(metrics["enhanced_triggered"])
            self.assertTrue(metrics["enhanced_succeeded"])
            self.assertGreaterEqual(metrics["base_analysis_seconds"], 0.0)
            self.assertGreaterEqual(metrics["enhanced_analysis_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
