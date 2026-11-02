"""Synthetic contracts only: never misrepresent these as real material benchmarks."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from benchmark import BenchmarkError, evaluate, load_manifest, markdown_report


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for folder in ("photos", "render/baseline", "render/photo", "render/refined"):
            (self.root / folder).mkdir(parents=True)
        self.spec = {
            "schema_version": 1,
            "experiment_id": "synthetic-contract-test",
            "material": "checker-patch",
            "capture_protocol": {
                "render_camera_locked": True,
                "exposure_locked": True,
                "white_balance_locked": True,
            },
            "candidates": {
                "baseline": "render/baseline",
                "photo": "render/photo",
                "refined": "render/refined",
            },
            "captures": [
                {"id": "train", "view_id": "front", "light_id": "top", "split": "conditioning", "reference": "photos/train.png"},
                {"id": "test-a", "view_id": "side", "light_id": "left", "split": "held_out", "reference": "photos/test-a.png", "render_file": "test-a.png", "mask": "photos/mask.png"},
                {"id": "test-b", "view_id": "side", "light_id": "right", "split": "held_out", "reference": "photos/test-b.png", "render_file": "test-b.png", "mask": "photos/mask.png"},
            ],
        }
        self.save_image("photos/train.png", np.full((20, 20, 3), 12, dtype=np.uint8))
        self.save_image("photos/mask.png", np.full((20, 20), 255, dtype=np.uint8))
        self.a = np.full((20, 20, 3), 70, dtype=np.uint8)
        self.b = np.full((20, 20, 3), 155, dtype=np.uint8)
        self.a[5:15, 5:15] = 110
        self.b[5:15, 5:15] = 210
        self.save_image("photos/test-a.png", self.a)
        self.save_image("photos/test-b.png", self.b)
        for cid, arr in (("test-a", self.a), ("test-b", self.b)):
            self.save_image(f"render/baseline/{cid}.png", np.full((20, 20, 3), 100, dtype=np.uint8))
            self.save_image(f"render/photo/{cid}.png", arr)
            self.save_image(f"render/refined/{cid}.png", arr)
        self.write_manifest()

    def save_image(self, path, data):
        Image.fromarray(data).save(self.root / path)

    def write_manifest(self):
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps(self.spec))

    def test_perfect_photo_candidate_beats_baseline_on_heldout_and_light_response(self):
        result = evaluate(self.manifest)
        self.assertEqual(result["conditioning_count"], 1)
        self.assertEqual(result["held_out_count"], 2)
        self.assertEqual(result["novel_held_out_views"], ["side"])
        self.assertEqual(result["results"]["photo"]["aggregate"]["color_mae"], 0)
        self.assertEqual(result["results"]["photo"]["aggregate"]["detail_mae"], 0)
        self.assertEqual(result["results"]["photo"]["aggregate"]["light_response_mae"], 0)
        self.assertGreater(result["results"]["baseline"]["aggregate"]["light_response_mae"], 0)
        self.assertGreater(result["paired_improvement_vs_baseline"]["photo"]["color_mae"], 0)
        self.assertIn("| photo |", markdown_report(result))

    def test_mask_excludes_out_of_roi_errors(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[7:13, 7:13] = 255
        self.save_image("photos/mask.png", mask)
        changed = self.a.copy()
        changed[:6, :] = 255
        self.save_image("render/photo/test-a.png", changed)
        report = evaluate(self.manifest)
        self.assertEqual(report["results"]["photo"]["held_out_views"]["test-a"]["color_mae"], 0)
        self.assertEqual(report["results"]["photo"]["held_out_views"]["test-a"]["roi_pixels"], 36)

    def test_mismatched_dimensions_fail_closed(self):
        self.save_image("render/photo/test-a.png", self.a[:10])
        with self.assertRaisesRegex(BenchmarkError, "dimensions differ"):
            evaluate(self.manifest)

    def test_identical_train_holdout_pixels_rejected_even_if_encoded_differently(self):
        self.save_image("photos/train.png", self.a)
        with self.assertRaisesRegex(BenchmarkError, "leakage"):
            evaluate(self.manifest)

    def test_traversal_from_manifest_rejected(self):
        self.spec["captures"][1]["reference"] = "../untrusted.jpg"
        self.write_manifest()
        with self.assertRaisesRegex(BenchmarkError, "escapes"):
            load_manifest(self.manifest)

    def test_duplicate_views_rejected(self):
        self.spec["captures"][2]["light_id"] = "left"
        self.write_manifest()
        with self.assertRaisesRegex(BenchmarkError, "Duplicate view/light"):
            load_manifest(self.manifest)

    def test_light_response_not_claimed_without_exposure_lock(self):
        self.spec["capture_protocol"]["exposure_locked"] = False
        self.write_manifest()
        result = evaluate(self.manifest)
        self.assertIsNone(result["results"]["photo"]["aggregate"]["light_response_mae"])
        self.assertTrue(any("skipped" in w for w in result["warnings"]))

    def test_outputs_are_reproducible_and_have_hashes(self):
        r1 = evaluate(self.manifest)
        r2 = evaluate(self.manifest)
        self.assertEqual(json.dumps(r1, sort_keys=True), json.dumps(r2, sort_keys=True))
        self.assertEqual(len(r1["reference_hashes"]["train"]), 64)

    def test_empty_mask_rejected(self):
        self.save_image("photos/mask.png", np.zeros((20, 20), dtype=np.uint8))
        with self.assertRaisesRegex(BenchmarkError, "Empty ROI"):
            evaluate(self.manifest)

    def test_missing_render_rejected(self):
        (self.root / "render/photo/test-b.png").unlink()
        with self.assertRaisesRegex(BenchmarkError, "Cannot read image"):
            evaluate(self.manifest)

    def test_staged_manifest_validates_before_images_arrive(self):
        self.assertEqual(load_manifest(self.manifest)["experiment_id"], "synthetic-contract-test")


if __name__ == "__main__":
    unittest.main()
