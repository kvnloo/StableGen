#!/usr/bin/env python3
"""Offline, Blender-free evaluation of photo-grounded StableGen material renders.

All measurements are deterministic, masked, paired and strictly held-out.
Lower errors are better; this is an evaluation tool, not a PBR estimator.
Requires numpy and Pillow (also used by StableGen).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
import json
import re
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

METRICS = ("color_mae", "structure_mae", "detail_mae")
SCHEMA_VERSION = 1


class BenchmarkError(ValueError):
    """Invalid benchmark input: no partial evaluation should be trusted."""


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"{label} must be a nonempty string")
    return value


def _safe_id(value: Any, label: str) -> str:
    value = _require_string(value, label)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value) is None:
        raise BenchmarkError(f"{label} may contain only ASCII letters, digits, '_' and '-'")
    return value


def _within(base: Path, relative: str, label: str) -> Path:
    """Disallow absolute/traversing paths to avoid publishing private sources."""
    relative = _require_string(relative, label)
    if Path(relative).is_absolute():
        raise BenchmarkError(f"{label} must be relative to manifest directory")
    resolved = (base / relative).resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise BenchmarkError(f"{label} escapes manifest directory: {relative}")
    return resolved


def load_manifest(filename: Path) -> dict:
    filename = Path(filename).resolve()
    try:
        spec = json.loads(filename.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"Cannot parse manifest {filename}: {exc}") from exc
    if not isinstance(spec, dict) or spec.get("schema_version") != SCHEMA_VERSION:
        raise BenchmarkError("Expected schema_version: 1")
    _safe_id(spec.get("experiment_id"), "experiment_id")
    _require_string(spec.get("material"), "material")
    protocol = spec.get("capture_protocol")
    if not isinstance(protocol, dict) or protocol.get("render_camera_locked") is not True:
        raise BenchmarkError("capture_protocol.render_camera_locked must be true")
    for field in ("exposure_locked", "white_balance_locked"):
        if type(protocol.get(field)) is not bool:
            raise BenchmarkError(f"capture_protocol.{field} must be boolean")
    candidates = spec.get("candidates")
    if not isinstance(candidates, dict) or "baseline" not in candidates or len(candidates) < 2:
        raise BenchmarkError("candidates must include baseline and at least one experiment")
    for candidate, folder in candidates.items():
        _safe_id(candidate, "candidate id")
        _within(filename.parent, folder, f"candidates.{candidate}")
    captures = spec.get("captures")
    if not isinstance(captures, list) or not captures:
        raise BenchmarkError("captures must be a nonempty list")
    seen_ids, held_out, seen_view_lights, seen_paths = set(), 0, set(), set()
    for i, capture in enumerate(captures):
        if not isinstance(capture, dict):
            raise BenchmarkError(f"captures[{i}] must be an object")
        ident = _safe_id(capture.get("id"), f"captures[{i}].id")
        if ident in seen_ids:
            raise BenchmarkError(f"Duplicate capture id: {ident}")
        seen_ids.add(ident)
        split = capture.get("split")
        if split not in ("conditioning", "held_out"):
            raise BenchmarkError(f"{ident}: split must be conditioning or held_out")
        view = _require_string(capture.get("view_id"), f"{ident}.view_id")
        light = _require_string(capture.get("light_id"), f"{ident}.light_id")
        if (view, light) in seen_view_lights:
            raise BenchmarkError(f"Duplicate view/light combination: {view}/{light}")
        seen_view_lights.add((view, light))
        ref = _within(filename.parent, capture.get("reference"), f"{ident}.reference")
        if ref in seen_paths:
            raise BenchmarkError(f"Reference path reused across captures: {ref}")
        seen_paths.add(ref)
        if capture.get("mask") is not None:
            _within(filename.parent, capture["mask"], f"{ident}.mask")
        if split == "held_out":
            held_out += 1
            _within(filename.parent, capture.get("render_file"), f"{ident}.render_file")
    if not held_out:
        raise BenchmarkError("At least one held_out capture is required")
    return spec


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def _image(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as im:
            if im.format not in ("PNG", "JPEG", "TIFF", "WEBP"):
                raise BenchmarkError(f"Unsupported image format: {path}")
            if im.mode == "RGBA" and im.getchannel("A").getextrema() != (255, 255):
                raise BenchmarkError(f"Nonopaque alpha in {path}; provide a separate ROI mask")
            return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    except (OSError, ValueError) as exc:
        raise BenchmarkError(f"Cannot read image {path}: {exc}") from exc


def _mask(path: Path | None, shape: tuple) -> np.ndarray:
    if path is None:
        return np.ones(shape[:2], dtype=bool)
    try:
        with Image.open(path) as im:
            if im.size != (shape[1], shape[0]):
                raise BenchmarkError(f"Mask dimensions mismatch: {path}")
            mask = np.asarray(im.convert("L"), dtype=np.uint8) >= 128
    except (OSError, ValueError) as exc:
        raise BenchmarkError(f"Cannot read mask {path}: {exc}") from exc
    if not np.any(mask):
        raise BenchmarkError(f"Empty ROI mask: {path}")
    return mask


def _blur(image: np.ndarray, radius: float = 4.0) -> np.ndarray:
    # Deterministic Gaussian low-pass for structural vs fine-detail diagnostics.
    pixels = np.rint(image * 255.0).clip(0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(pixels).filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0


def _linear_srgb(image: np.ndarray) -> np.ndarray:
    """Convert gamma-encoded sRGB to approximately linear scene-light RGB."""
    return np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)


def _mae(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    return float(np.abs(left - right)[mask].mean())


def _metrics(reference: np.ndarray, rendered: np.ndarray, roi: np.ndarray) -> dict:
    if rendered.shape != reference.shape:
        raise BenchmarkError(f"Render/reference dimensions differ: {rendered.shape} vs {reference.shape}; align externally")
    blurred_ref, blurred_render = _blur(reference), _blur(rendered)
    return {
        "color_mae": _mae(reference, rendered, roi),
        "structure_mae": _mae(blurred_ref, blurred_render, roi),
        "detail_mae": _mae(reference - blurred_ref, rendered - blurred_render, roi),
        "roi_pixels": int(roi.sum()),
    }


def evaluate(manifest_file: Path, *, diff_dir: Path | None = None) -> dict:
    """Evaluate only held-out captures; fail closed for missing/misaligned data."""
    manifest_file = Path(manifest_file).resolve()
    spec = load_manifest(manifest_file)
    base = manifest_file.parent
    captures = spec["captures"]
    ref_arrays: dict[str, np.ndarray] = {}
    ref_hashes: dict[str, str] = {}
    roi_masks: dict[str, np.ndarray] = {}
    pixel_hash_splits: dict[str, str] = {}
    warnings: list[str] = []
    for capture in captures:
        ident = capture["id"]
        path = _within(base, capture["reference"], f"{ident}.reference")
        source = _image(path)
        ref_arrays[ident] = source
        ref_hashes[ident] = _sha_file(path)
        pixel_hash = sha256(source.tobytes()).hexdigest()
        earlier = pixel_hash_splits.get(pixel_hash)
        if earlier is not None:
            if earlier != capture["split"]:
                raise BenchmarkError("Conditioning/held-out image leakage: duplicate decoded image across splits")
            raise BenchmarkError("Duplicate decoded reference image across captures; held-out conditions must be independent")
        pixel_hash_splits[pixel_hash] = capture["split"]
        mask_path = capture.get("mask")
        roi_masks[ident] = _mask(_within(base, mask_path, f"{ident}.mask") if mask_path else None, source.shape)

    held_out = [capture for capture in captures if capture["split"] == "held_out"]
    train_views = {capture["view_id"] for capture in captures if capture["split"] == "conditioning"}
    novel_views = sorted({capture["view_id"] for capture in held_out} - train_views)
    if not novel_views:
        warnings.append("No camera view is held out from conditioning: angle-generalization remains untested")
    if not any(cap.get("mask") for cap in held_out):
        warnings.append("No ROI mask: background may dominate the material error")

    light_pairs = []
    by_view: dict[str, list[dict]] = defaultdict(list)
    for cap in held_out:
        by_view[cap["view_id"]].append(cap)
    for view, matched in by_view.items():
        for left, right in combinations(matched, 2):
            light_pairs.append((left, right))

    calibrated = (
        spec["capture_protocol"]["exposure_locked"]
        and spec["capture_protocol"]["white_balance_locked"]
    )
    if not light_pairs:
        warnings.append("No same-camera multi-light held-out pair: lighting-response error unavailable")
    if light_pairs and not calibrated:
        warnings.append("Light-response error skipped: lock camera exposure and white balance across light pairs")

    if diff_dir is not None:
        diff_dir = Path(diff_dir)
        diff_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for candidate, folder in spec["candidates"].items():
        out_root = _within(base, folder, f"candidates.{candidate}")
        views = {}
        paths = {}
        for cap in held_out:
            ident = cap["id"]
            render_path = _within(out_root, cap["render_file"], f"{candidate}/{ident}.render_file")
            rendered = _image(render_path)
            reference = ref_arrays[ident]
            roi = roi_masks[ident]
            measures = _metrics(reference, rendered, roi)
            measures["reference_sha256"] = ref_hashes[ident]
            measures["render_sha256"] = _sha_file(render_path)
            views[ident] = measures
            paths[ident] = render_path
            if diff_dir is not None:
                # Masked scalar residual preview, not a photorealism score.
                heat = np.rint(np.abs(reference - rendered).mean(axis=2) * 255).clip(0, 255).astype(np.uint8)
                heat[~roi] = 0
                (diff_dir / candidate).mkdir(parents=True, exist_ok=True)
                Image.fromarray(heat).save(diff_dir / candidate / f"{ident}-error.png")

        weights = np.array([v["roi_pixels"] for v in views.values()], dtype=np.float64)
        aggregate = {metric: float(np.average([v[metric] for v in views.values()], weights=weights)) for metric in METRICS}
        light_errors = []
        light_weights = []
        if calibrated:
            for cap_a, cap_b in light_pairs:
                a, b = cap_a["id"], cap_b["id"]
                ref_a, ref_b = ref_arrays[a], ref_arrays[b]
                ren_a, ren_b = _image(paths[a]), _image(paths[b])
                if ref_a.shape != ref_b.shape or ren_a.shape != ren_b.shape:
                    raise BenchmarkError(f"Same-camera light pair must share dimensions: {a}, {b}")
                intersection = roi_masks[a] & roi_masks[b]
                if not np.any(intersection):
                    raise BenchmarkError(f"Same-camera light pair has no shared ROI: {a}, {b}")
                response_error = _mae(
                    _linear_srgb(ref_a) - _linear_srgb(ref_b),
                    _linear_srgb(ren_a) - _linear_srgb(ren_b), intersection
                )
                light_errors.append(response_error)
                light_weights.append(int(intersection.sum()))
        aggregate["light_response_mae"] = float(np.average(light_errors, weights=light_weights)) if light_errors else None
        results[candidate] = {
            "held_out_views": views,
            "aggregate": aggregate,
            "light_pairs_evaluated": len(light_errors),
        }

    baseline = results["baseline"]["aggregate"]
    paired_deltas = {}
    for candidate, result in results.items():
        if candidate == "baseline":
            continue
        scores = result["aggregate"]
        paired_deltas[candidate] = {
            metric: (baseline[metric] - scores[metric] if baseline[metric] is not None and scores[metric] is not None else None)
            for metric in (*METRICS, "light_response_mae")
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": spec["experiment_id"],
        "material": spec["material"],
        "source_revision": spec.get("source_revision"),
        "capture_protocol": spec["capture_protocol"],
        "candidate_metadata": spec.get("candidate_metadata", {}),
        "conditioning_count": len(captures) - len(held_out),
        "held_out_count": len(held_out),
        "novel_held_out_views": novel_views,
        "reference_hashes": ref_hashes,
        "results": results,
        "paired_improvement_vs_baseline": paired_deltas,
        "warnings": warnings,
        "interpretation": "Lower error is better. Positive paired improvement means lower error than baseline. No physical PBR accuracy is inferred from 2D image agreement alone.",
    }


def markdown_report(report: dict) -> str:
    lines = [f"# Material benchmark: {report['experiment_id']}", "",
             f"Material: {report['material']}. Held-out captures: {report['held_out_count']}. Novel views: {len(report['novel_held_out_views'])}.",
             "", "All errors are [0,1] normalized, lower is better. Paired improvements are baseline minus candidate.", "",
             "| Candidate | Color MAE | Structure MAE | Detail MAE | Multi-light MAE |", "|---|---:|---:|---:|---:|"]
    for candidate, details in report["results"].items():
        scores = details["aggregate"]
        vals = [f"{scores[k]:.5f}" if scores[k] is not None else "not evaluated" for k in (*METRICS, "light_response_mae")]
        lines.append(f"| {candidate} | " + " | ".join(vals) + " |")
    if report["paired_improvement_vs_baseline"]:
        lines.extend(["", "## Paired improvements vs baseline", ""])
        for candidate, delta in report["paired_improvement_vs_baseline"].items():
            values = ", ".join(f"{k}={v:+.5f}" if v is not None else f"{k}=n/a" for k, v in delta.items())
            lines.append(f"- **{candidate}**: {values}")
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    lines.extend(["", "## Limitations", "", "Pixel alignment, capture calibration, surface geometry and user leakage controls still require review. These errors alone cannot establish measured reflectance or material authenticity.", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--check", action="store_true", help="Validate manifest structure only; do not read images")
    parser.add_argument("--report", type=Path, help="Write deterministic JSON report")
    parser.add_argument("--markdown", type=Path, help="Write human-readable Markdown summary")
    parser.add_argument("--diff-dir", type=Path, help="Write per-view masked residual images")
    args = parser.parse_args(argv)
    try:
        if args.check:
            spec = load_manifest(args.manifest)
            print(f"Manifest valid: {spec['experiment_id']} ({len(spec['captures'])} capture records)")
            return 0
        result = evaluate(args.manifest, diff_dir=args.diff_dir)
        report_text = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report_text, encoding="utf-8")
        if args.markdown:
            args.markdown.parent.mkdir(parents=True, exist_ok=True)
            args.markdown.write_text(markdown_report(result), encoding="utf-8")
        print(markdown_report(result))
        return 0
    except (BenchmarkError, OSError) as exc:
        print(f"Benchmark invalid: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
