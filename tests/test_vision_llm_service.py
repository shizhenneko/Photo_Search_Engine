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
            self.assertIn("embedding_text", analysis)
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

    def test_description_prompt_is_compact_but_keeps_required_fields(self) -> None:
        service = SU8VisionLLMService(
            api_key="test-key",
            model_name="gpt-5.4",
            base_url="https://www.su8.codes/codex/v1",
            client=Mock(),
        )

        prompt = service._build_description_prompt()

        self.assertIn("\"description\"", prompt)
        self.assertIn("\"inner_content_summary\"", prompt)
        self.assertIn("\"identity_candidates\"", prompt)
        self.assertIn("\"evidence_types\"", prompt)
        self.assertIn("\"scope\"", prompt)
        self.assertIn("不使用固定词表", prompt)
        self.assertIn("200 字内", prompt)
        self.assertIn("区分画面中真正出现的主体", prompt)
        self.assertIn("不要把 OCR 片段或名字机械重复成标签", prompt)
        self.assertNotIn("album_cover, poster, stage_performance", prompt)
        self.assertNotIn("signature_stage_pose", prompt)

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

    def test_analyze_image_accepts_raw_string_response(self) -> None:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = (
            '{"description":"文档截图","outer_scene_summary":"一页文档","inner_content_summary":"布尔代数公式表",'
            '"media_types":["document"],"tags":["布尔代数"],"ocr_text":"Properties of Boolean Algebra",'
            '"person_roles":[],"identity_candidates":[],"analysis_flags":{"text_heavy":true}}'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sheet.png")
            Image.new("RGB", (320, 240), color=(255, 255, 255)).save(path)
            service = SU8VisionLLMService(
                api_key="test-key",
                model_name="gpt-5.4",
                base_url="https://www.su8.codes/codex/v1",
                client=mock_client,
            )

            analysis = service.analyze_image(path)

            self.assertEqual(analysis["media_types"], ["document"])
            self.assertIn("布尔代数", analysis["retrieval_text"])

    def test_extract_response_text_accepts_content_list(self) -> None:
        service = SU8VisionLLMService(
            api_key="test-key",
            model_name="gpt-5.4",
            base_url="https://www.su8.codes/codex/v1",
            client=Mock(),
        )
        response = Mock(
            choices=[
                Mock(
                    message=Mock(
                        content=[
                            {"type": "output_text", "text": '{"description":"截图","outer_scene_summary":"页面","inner_content_summary":"","media_types":["document"],"tags":[],"ocr_text":"","person_roles":[],"identity_candidates":[],"analysis_flags":{}}'}
                        ]
                    )
                )
            ]
        )

        text = service._extract_response_text(response)

        self.assertIn('"description":"截图"', text)

    def test_parse_json_response_accepts_wrapped_text(self) -> None:
        service = SU8VisionLLMService(
            api_key="test-key",
            model_name="gpt-5.4",
            base_url="https://www.su8.codes/codex/v1",
            client=Mock(),
        )

        payload = service._parse_json_response(
            '分析如下：```json {"description":"截图","outer_scene_summary":"页面","inner_content_summary":"","media_types":["document"],"tags":[],"ocr_text":"","person_roles":[],"identity_candidates":[],"analysis_flags":{}} ```'
        )

        self.assertEqual(payload["description"], "截图")

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

    def test_analyze_image_repairs_enhanced_json_when_needed(self) -> None:
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
                            content='结果如下：{"description":"vim 键盘速查表","inner_content_summary":"包含 vim 常用命令和键位说明的速查图","ocr_text":"vim hjkl dd yy p u ctrl r","tags":["vim","快捷键"],"analysis_flags":{"text_heavy":true}}'
                        )
                    )
                ]
            ),
            Mock(
                choices=[
                    Mock(
                        message=Mock(
                            content='{"description":"vim 键盘速查表","inner_content_summary":"包含 vim 常用命令和键位说明的速查图","ocr_text":"vim hjkl dd yy p u ctrl r","tags":["vim","快捷键"],"analysis_flags":{"text_heavy":true}}'
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

            self.assertIn("快捷键", analysis["tags"])
            assert metrics is not None
            self.assertTrue(metrics["enhanced_triggered"])
            self.assertTrue(metrics["enhanced_succeeded"])
            self.assertGreaterEqual(metrics["enhanced_repair_seconds"], 0.0)

    def test_enhanced_prompt_requests_visual_identity_recheck(self) -> None:
        service = SU8VisionLLMService(
            api_key="test-key",
            model_name="gpt-5.4",
            base_url="https://www.su8.codes/codex/v1",
            client=Mock(),
        )
        prompt = service._build_enhanced_prompt(
            {
                "description": "舞台上的男歌手",
                "media_types": ["stage_performance"],
                "identity_candidates": [],
            },
            "public_figure_needs_review",
        )
        self.assertIn("第二轮复核", prompt)
        self.assertIn("只返回需要修改或补充的字段", prompt)
        self.assertIn("触发原因：public_figure_needs_review", prompt)
        self.assertIn("identity_candidates", prompt)
        self.assertIn("区分画面中真正出现的人物", prompt)
        self.assertIn("evidence_types", prompt)
        self.assertIn("scope", prompt)

    def test_enhanced_prompt_uses_compact_context(self) -> None:
        service = SU8VisionLLMService(
            api_key="test-key",
            model_name="gpt-5.4",
            base_url="https://www.su8.codes/codex/v1",
            client=Mock(),
        )
        long_analysis = {
            "description": "舞台上的男歌手，背后有大屏幕和灯光，手持麦克风，正在进行演出",
            "outer_scene_summary": "观众席前方拍摄到的室内舞台现场，画面包含灯光和屏幕",
            "inner_content_summary": "屏幕上出现演出视觉素材和人物特写，整体像大型商业演出现场的舞台记录",
            "media_types": ["stage_performance"],
            "tags": ["舞台", "男歌手", "演出现场", "大屏幕"],
            "ocr_text": "SUPER LIVE WORLD TOUR 2026 SPECIAL GUEST ARTIST",
            "person_roles": ["singer"],
            "identity_names": ["某歌手"],
            "identity_candidates": [
                {
                    "name": "某歌手",
                    "confidence": 0.73,
                    "evidence_sources": ["face_similarity", "hairstyle", "stage_pose"],
                    "evidence_types": ["visual"],
                    "scope": "depicted",
                }
            ],
            "analysis_flags": {"has_stage": True, "has_public_figure_likelihood": True},
        }

        prompt = service._build_enhanced_prompt(long_analysis, "public_figure_needs_review")

        self.assertIn("第一次结果摘要：", prompt)
        self.assertIn("\"ocr_text_excerpt\"", prompt)
        self.assertIn("\"evidence_types\"", prompt)
        self.assertNotIn("\"ocr_text\":\"", prompt)
        self.assertNotIn("\"identity_evidence\"", prompt)


if __name__ == "__main__":
    unittest.main()
