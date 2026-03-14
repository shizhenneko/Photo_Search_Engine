import unittest

from utils.structured_analysis import (
    build_match_summary,
    normalize_analysis_payload,
    normalize_media_types,
    should_run_enhanced_analysis,
)


class StructuredAnalysisTests(unittest.TestCase):
    def test_normalize_media_types(self) -> None:
        result = normalize_media_types(["专辑", "海报", "album_cover", "未知"])
        self.assertEqual(result, ["album_cover", "poster"])

    def test_normalize_analysis_payload_selects_high_confidence_identity(self) -> None:
        payload = normalize_analysis_payload(
            {
                "description": "舞台照片",
                "outer_scene_summary": "演出现场",
                "inner_content_summary": "大屏幕出现歌手",
                "media_types": ["演出"],
                "tags": [{"tag": "舞台", "confidence": 0.9}, {"tag": "模糊", "confidence": 0.2}],
                "ocr_text": "Jay Chou Live",
                "identity_candidates": [
                    {
                        "name": "周杰伦",
                        "aliases": ["Jay Chou"],
                        "confidence": 0.8,
                        "evidence_sources": ["ocr_text"],
                    }
                ],
            },
            tag_min_confidence=0.65,
            identity_text_threshold=0.7,
            identity_visual_threshold=0.92,
        )
        self.assertIn("周杰伦", payload["identity_names"])
        self.assertEqual(payload["tags"], ["舞台"])
        self.assertIn("album_cover", normalize_media_types(["专辑封面"]))

    def test_normalize_analysis_payload_accepts_visual_identity_with_high_confidence(self) -> None:
        payload = normalize_analysis_payload(
            {
                "description": "舞台男歌手",
                "outer_scene_summary": "演出现场",
                "inner_content_summary": "",
                "media_types": ["演出"],
                "tags": [{"tag": "男歌手", "confidence": 0.9}],
                "ocr_text": "",
                "identity_candidates": [
                    {
                        "name": "陶喆",
                        "aliases": ["David Tao"],
                        "confidence": 0.96,
                        "evidence_sources": ["face_similarity", "signature_stage_pose"],
                    }
                ],
            },
            tag_min_confidence=0.65,
            identity_text_threshold=0.7,
            identity_visual_threshold=0.92,
        )
        self.assertIn("陶喆", payload["identity_names"])
        self.assertIn("face_similarity", payload["identity_evidence"])

    def test_normalize_analysis_payload_treats_chinese_readable_text_as_text_evidence(self) -> None:
        payload = normalize_analysis_payload(
            {
                "description": "榜单截图",
                "outer_scene_summary": "手机截图",
                "inner_content_summary": "",
                "media_types": ["screen"],
                "tags": [{"tag": "音乐榜单", "confidence": 0.9}],
                "ocr_text": "河南说唱之神",
                "identity_candidates": [
                    {
                        "name": "河南说唱之神",
                        "aliases": [],
                        "confidence": 0.9,
                        "evidence_sources": ["可读文字"],
                    }
                ],
            },
            tag_min_confidence=0.65,
            identity_text_threshold=0.7,
            identity_visual_threshold=0.92,
        )
        self.assertIn("河南说唱之神", payload["identity_names"])

    def test_should_run_enhanced_analysis(self) -> None:
        self.assertTrue(
            should_run_enhanced_analysis(
                {
                    "media_types": ["album_cover"],
                    "person_roles": [],
                    "analysis_flags": {},
                    "ocr_text": "",
                }
            )
        )

    def test_build_match_summary(self) -> None:
        summary = build_match_summary(
            {
                "media_types": ["album_cover"],
                "top_tags": ["专辑", "封面"],
                "identity_names": ["周杰伦"],
                "identity_evidence": ["ocr_text"],
                "ocr_text": "Jay Chou 2025 Tour",
            }
        )
        self.assertEqual(summary["media_types"], ["album_cover"])
        self.assertEqual(summary["identities"], ["周杰伦"])
        self.assertIn("Jay Chou", summary["ocr_excerpt"])


if __name__ == "__main__":
    unittest.main()
