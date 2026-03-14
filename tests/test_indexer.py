import os
import sys
import tempfile
import time
import unittest
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from PIL import Image

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.indexer import Indexer
from tests.helpers import FakeEmbeddingService
from utils.vector_store import VectorStore
from utils.vision_llm_service import LocalVisionLLMService


def _create_image(path: str, size: tuple[int, int] = (64, 48)) -> None:
    image = Image.new("RGB", size, color=(255, 0, 0))
    image.save(path)


class IndexerTests(unittest.TestCase):
    def test_indexer_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            photo_dir = os.path.join(tmp, "photos")
            data_dir = os.path.join(tmp, "data")
            os.makedirs(photo_dir)
            os.makedirs(data_dir)

            vector_store = VectorStore(dimension=8, index_path=os.path.join(tmp, "idx"), metadata_path=os.path.join(tmp, "meta.json"))
            indexer = Indexer(
                photo_dir=photo_dir,
                vision=LocalVisionLLMService(),
                embedding=FakeEmbeddingService(dimension=8),
                vector_store=vector_store,
                data_dir=data_dir,
            )
            self.assertEqual(indexer.photo_dir, os.path.abspath(photo_dir))

    def test_scan_photos_filters_and_sorts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            photo_dir = os.path.join(tmp, "photos")
            os.makedirs(photo_dir)

            img1 = os.path.join(photo_dir, "a.jpg")
            img2 = os.path.join(photo_dir, "b.jpg")
            txt = os.path.join(photo_dir, "readme.txt")
            _create_image(img1)
            _create_image(img2)
            with open(txt, "w", encoding="utf-8") as file:
                file.write("x")

            os.utime(img1, (1, 1))
            os.utime(img2, (2, 2))

            vector_store = VectorStore(dimension=8, index_path=os.path.join(tmp, "idx"), metadata_path=os.path.join(tmp, "meta.json"))
            indexer = Indexer(
                photo_dir=photo_dir,
                vision=LocalVisionLLMService(),
                embedding=FakeEmbeddingService(dimension=8),
                vector_store=vector_store,
                data_dir=tmp,
            )
            photos = indexer.scan_photos()
            self.assertEqual(photos, [img1, img2])

    def test_build_index_fails_when_no_photos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            photo_dir = os.path.join(tmp, "photos")
            os.makedirs(photo_dir)
            vector_store = VectorStore(dimension=8, index_path=os.path.join(tmp, "idx"), metadata_path=os.path.join(tmp, "meta.json"))
            indexer = Indexer(
                photo_dir=photo_dir,
                vision=LocalVisionLLMService(),
                embedding=FakeEmbeddingService(dimension=8),
                vector_store=vector_store,
                data_dir=tmp,
            )
            result = indexer.build_index()
            self.assertEqual(result["status"], "failed")

    def test_build_index_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            photo_dir = os.path.join(tmp, "photos")
            data_dir = os.path.join(tmp, "data")
            os.makedirs(photo_dir)
            os.makedirs(data_dir)
            for index in range(3):
                _create_image(os.path.join(photo_dir, f"photo_{index}.jpg"))

            vector_store = VectorStore(dimension=8, index_path=os.path.join(data_dir, "idx"), metadata_path=os.path.join(data_dir, "meta.json"))
            indexer = Indexer(
                photo_dir=photo_dir,
                vision=LocalVisionLLMService(),
                embedding=FakeEmbeddingService(dimension=8),
                vector_store=vector_store,
                data_dir=data_dir,
            )
            result = indexer.build_index()
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["indexed_count"], 3)
            self.assertIn("retrieval_text", vector_store.metadata[0])
            self.assertIn("media_types", vector_store.metadata[0])
            self.assertEqual(
                result["timing_log_path"],
                os.path.join(data_dir, "index_timing.jsonl"),
            )

    def test_build_index_incrementally_adds_only_new_photos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            photo_dir = os.path.join(tmp, "photos")
            data_dir = os.path.join(tmp, "data")
            os.makedirs(photo_dir)
            os.makedirs(data_dir)

            first_photo = os.path.join(photo_dir, "photo_0.jpg")
            second_photo = os.path.join(photo_dir, "photo_1.jpg")
            _create_image(first_photo)

            index_path = os.path.join(data_dir, "idx")
            metadata_path = os.path.join(data_dir, "meta.json")
            vector_store = VectorStore(dimension=8, index_path=index_path, metadata_path=metadata_path)
            indexer = Indexer(
                photo_dir=photo_dir,
                vision=LocalVisionLLMService(),
                embedding=FakeEmbeddingService(dimension=8),
                vector_store=vector_store,
                data_dir=data_dir,
            )

            first_result = indexer.build_index()
            self.assertEqual(first_result["status"], "success")
            self.assertEqual(first_result["indexed_count"], 1)
            self.assertEqual(vector_store.get_total_items(), 1)

            _create_image(second_photo)
            with patch.object(indexer, "process_batch", wraps=indexer.process_batch) as process_batch:
                second_result = indexer.build_index()

            self.assertEqual(second_result["status"], "success")
            self.assertEqual(second_result["indexed_count"], 2)
            self.assertEqual(vector_store.get_total_items(), 2)
            processed_paths = []
            for call in process_batch.call_args_list:
                processed_paths.extend(call.args[0])
            self.assertEqual(processed_paths, [second_photo])
            self.assertEqual(
                sorted(item["photo_path"] for item in vector_store.metadata),
                sorted([first_photo, second_photo]),
            )

            reloaded_store = VectorStore(dimension=8, index_path=index_path, metadata_path=metadata_path)
            self.assertTrue(reloaded_store.load())
            self.assertEqual(reloaded_store.get_total_items(), 2)
            self.assertEqual(
                sorted(item["photo_path"] for item in reloaded_store.metadata),
                sorted([first_photo, second_photo]),
            )

    def test_build_index_skips_already_indexed_photos_without_duplication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            photo_dir = os.path.join(tmp, "photos")
            data_dir = os.path.join(tmp, "data")
            os.makedirs(photo_dir)
            os.makedirs(data_dir)

            for index in range(2):
                _create_image(os.path.join(photo_dir, f"photo_{index}.jpg"))

            vector_store = VectorStore(
                dimension=8,
                index_path=os.path.join(data_dir, "idx"),
                metadata_path=os.path.join(data_dir, "meta.json"),
            )
            indexer = Indexer(
                photo_dir=photo_dir,
                vision=LocalVisionLLMService(),
                embedding=FakeEmbeddingService(dimension=8),
                vector_store=vector_store,
                data_dir=data_dir,
            )

            first_result = indexer.build_index()
            self.assertEqual(first_result["status"], "success")
            self.assertEqual(vector_store.get_total_items(), 2)

            with patch.object(indexer, "process_batch", wraps=indexer.process_batch) as process_batch:
                second_result = indexer.build_index()
            self.assertEqual(second_result["status"], "success")
            self.assertEqual(second_result["indexed_count"], 2)
            self.assertEqual(vector_store.get_total_items(), 2)
            process_batch.assert_not_called()

    def test_build_index_force_rebuild_reprocesses_existing_photos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            photo_dir = os.path.join(tmp, "photos")
            data_dir = os.path.join(tmp, "data")
            os.makedirs(photo_dir)
            os.makedirs(data_dir)

            for index in range(2):
                _create_image(os.path.join(photo_dir, f"photo_{index}.jpg"))

            vector_store = VectorStore(
                dimension=8,
                index_path=os.path.join(data_dir, "idx"),
                metadata_path=os.path.join(data_dir, "meta.json"),
            )
            indexer = Indexer(
                photo_dir=photo_dir,
                vision=LocalVisionLLMService(),
                embedding=FakeEmbeddingService(dimension=8),
                vector_store=vector_store,
                data_dir=data_dir,
            )

            first_result = indexer.build_index()
            self.assertEqual(first_result["status"], "success")

            with patch.object(indexer, "process_batch", wraps=indexer.process_batch) as process_batch:
                second_result = indexer.build_index(force_rebuild=True)

            self.assertEqual(second_result["status"], "success")
            self.assertEqual(vector_store.get_total_items(), 2)
            processed_paths = []
            for call in process_batch.call_args_list:
                processed_paths.extend(call.args[0])
            self.assertEqual(len(processed_paths), 2)

    def test_start_build_in_background_marks_processing_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            photo_dir = os.path.join(tmp, "photos")
            data_dir = os.path.join(tmp, "data")
            os.makedirs(photo_dir)
            os.makedirs(data_dir)
            _create_image(os.path.join(photo_dir, "photo_0.jpg"))

            vector_store = VectorStore(
                dimension=8,
                index_path=os.path.join(data_dir, "idx"),
                metadata_path=os.path.join(data_dir, "meta.json"),
            )
            indexer = Indexer(
                photo_dir=photo_dir,
                vision=LocalVisionLLMService(),
                embedding=FakeEmbeddingService(dimension=8),
                vector_store=vector_store,
                data_dir=data_dir,
            )

            with patch.object(indexer, "build_index", side_effect=lambda **kwargs: time.sleep(0.2)) as build_index:
                result = indexer.start_build_in_background(force_rebuild=False)
                self.assertEqual(result["status"], "processing")
                self.assertEqual(result["total_count"], 1)
                self.assertTrue(os.path.exists(indexer._lock_path))
                for _ in range(20):
                    if build_index.called:
                        break
                    time.sleep(0.02)
                self.assertTrue(build_index.called)

    def test_get_status_clears_stale_legacy_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            photo_dir = os.path.join(tmp, "photos")
            data_dir = os.path.join(tmp, "data")
            os.makedirs(photo_dir)
            os.makedirs(data_dir)

            vector_store = VectorStore(
                dimension=8,
                index_path=os.path.join(data_dir, "idx"),
                metadata_path=os.path.join(data_dir, "meta.json"),
            )
            indexer = Indexer(
                photo_dir=photo_dir,
                vision=LocalVisionLLMService(),
                embedding=FakeEmbeddingService(dimension=8),
                vector_store=vector_store,
                data_dir=data_dir,
                timeout=1,
                batch_size=1,
            )
            indexer._lock_stale_seconds = 1

            with open(indexer._lock_path, "w", encoding="utf-8") as file:
                file.write(datetime.now().isoformat())
            with open(indexer._status_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "status": "processing",
                        "message": "索引构建中",
                        "total_count": 10,
                        "indexed_count": 4,
                        "failed_count": 0,
                        "fallback_ratio": 0.0,
                        "index_path": indexer.vector_store.index_path,
                        "elapsed_time": 30.0,
                    },
                    file,
                    ensure_ascii=False,
                    indent=2,
                )
            stale_at = time.time() - 5
            os.utime(indexer._lock_path, (stale_at, stale_at))
            os.utime(indexer._status_path, (stale_at, stale_at))

            status = indexer.get_status()

            self.assertFalse(os.path.exists(indexer._lock_path))
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["indexed_count"], 4)
            self.assertIn("已中断", status["message"])

    def test_start_build_in_background_recovers_from_stale_json_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            photo_dir = os.path.join(tmp, "photos")
            data_dir = os.path.join(tmp, "data")
            os.makedirs(photo_dir)
            os.makedirs(data_dir)
            _create_image(os.path.join(photo_dir, "photo_0.jpg"))

            vector_store = VectorStore(
                dimension=8,
                index_path=os.path.join(data_dir, "idx"),
                metadata_path=os.path.join(data_dir, "meta.json"),
            )
            indexer = Indexer(
                photo_dir=photo_dir,
                vision=LocalVisionLLMService(),
                embedding=FakeEmbeddingService(dimension=8),
                vector_store=vector_store,
                data_dir=data_dir,
            )

            with open(indexer._lock_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "pid": 99999999,
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
                    },
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            with patch.object(indexer, "build_index", side_effect=lambda **kwargs: time.sleep(0.1)):
                result = indexer.start_build_in_background(force_rebuild=False)

            self.assertEqual(result["status"], "processing")
            lock_payload = indexer._read_lock_payload()
            self.assertEqual(lock_payload.get("pid"), os.getpid())

    def test_extract_time_info_requires_exif_datetime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            photo_dir = os.path.join(tmp, "photos")
            data_dir = os.path.join(tmp, "data")
            os.makedirs(photo_dir)
            os.makedirs(data_dir)

            vector_store = VectorStore(
                dimension=8,
                index_path=os.path.join(data_dir, "idx"),
                metadata_path=os.path.join(data_dir, "meta.json"),
            )
            indexer = Indexer(
                photo_dir=photo_dir,
                vision=LocalVisionLLMService(),
                embedding=FakeEmbeddingService(dimension=8),
                vector_store=vector_store,
                data_dir=data_dir,
            )

            time_info = indexer._extract_time_info(
                exif_data={},
                file_time="2025-08-17T12:34:56",
            )

            self.assertIsNone(time_info["year"])
            self.assertIsNone(time_info["season"])
            self.assertIsNone(time_info["time_period"])
            self.assertIsNone(time_info["datetime_str"])

    def test_build_index_writes_step_timing_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            photo_dir = os.path.join(tmp, "photos")
            data_dir = os.path.join(tmp, "data")
            os.makedirs(photo_dir)
            os.makedirs(data_dir)

            photo_path = os.path.join(photo_dir, "photo_0.jpg")
            _create_image(photo_path)

            vector_store = VectorStore(
                dimension=8,
                index_path=os.path.join(data_dir, "idx"),
                metadata_path=os.path.join(data_dir, "meta.json"),
            )
            indexer = Indexer(
                photo_dir=photo_dir,
                vision=LocalVisionLLMService(),
                embedding=FakeEmbeddingService(dimension=8),
                vector_store=vector_store,
                data_dir=data_dir,
            )

            result = indexer.build_index(force_rebuild=True)

            self.assertEqual(result["status"], "success")
            timing_log_path = os.path.join(data_dir, "index_timing.jsonl")
            self.assertTrue(os.path.exists(timing_log_path))

            with open(timing_log_path, "r", encoding="utf-8") as file:
                records = [json.loads(line) for line in file if line.strip()]

            events = [record["event"] for record in records]
            self.assertIn("build_started", events)
            self.assertIn("build_finished", events)
            self.assertIn("photo_timing", events)
            self.assertIn("photo_persist_timing", events)

            photo_record = next(record for record in records if record["event"] == "photo_timing")
            self.assertEqual(photo_record["photo_path"], photo_path)
            self.assertEqual(photo_record["status"], "success")
            self.assertIn("generate_analysis", photo_record["steps"])
            self.assertIn("extract_exif", photo_record["steps"])
            self.assertIn("extract_time_info", photo_record["steps"])
            self.assertIn("generate_embedding", photo_record["steps"])
            self.assertGreaterEqual(photo_record["total_elapsed_seconds"], 0.0)

            persist_record = next(record for record in records if record["event"] == "photo_persist_timing")
            self.assertIn("vector_store_add_item", persist_record["steps"])
            self.assertGreaterEqual(persist_record["steps"]["vector_store_add_item"], 0.0)

            run_ids = {record.get("run_id") for record in records}
            self.assertEqual(len(run_ids), 1)
            self.assertIsNotNone(next(iter(run_ids)))


if __name__ == "__main__":
    unittest.main()
