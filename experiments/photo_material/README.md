# Photo-grounded material experiment (E0; offline evaluator)

Tracking: [kvnloo/StableGen #1](https://github.com/kvnloo/StableGen/issues/1). Upstream proposal draft: [UPSTREAM_PROPOSAL.md](UPSTREAM_PROPOSAL.md). Related upstream discussion: [sakalond/StableGen #32](https://github.com/sakalond/StableGen/issues/32) (custom ComfyUI workflows). This experiment does **not** change StableGen's Blender operators, material outputs or default behavior.

## Hypothesis and limits

A properly registered smartphone/DSLR photo may improve material detail more than prompt-only texture generation. This claim is **unproven** until a photo-conditioned material beats a same-model baseline on *held-out* camera and lighting conditions without worse seams/geometry. A sharp texture or one matched still does not demonstrate physical reflectance. Marigold estimates, rather than measures, PBR maps.

E0 supplies a Blender-free, deterministic, masked, paired **evaluation** CLI for three candidates (or more). It does not synthesize textures, rectify photos, optimize shaders, or automatically rank by a single aesthetic score. GPU/Blender/ComfyUI rendering is a separate step. We intentionally keep camera/crop geometry out of the material-quality loop, as DreamLoop requires.

## Capture protocol (repeat separately for wood, stone, painted metal)

1. Choose a single real material on an already modeled mesh, or create a planar sample with **known physical scale**. Capture perpendicular source with scale marker and gray/color checker if possible. RAW/DNG is useful for controlled color; export color-managed PNG/JPEG at a fixed resolution. DSLR is optional; phone with manual ISO/shutter/white-balance lock is enough to begin.
2. Capture a conditioning photo (e.g. front under diffuse light). **Never feed a held-out photo into Qwen, IPAdapter, an edit prompt, StableGen context, DreamLoop target, or any other optimization step.** Keep exact train/test filenames and SHA-256 checksums.
3. Collect at least two *unseen* photos of the same material at the same fixed camera with **different light directions** (e.g. side view, raking light vs fill). Keep aperture, exposure and white balance fixed. Capture another view for camera generalization. Record actual light positions if possible.
4. For each held-out photo, align the scene's camera, focal length, mesh, scale, lighting and postprocessing **before** testing textures; render a gray/clay check to confirm camera and massing. Rectify/crop reference and render externally to exactly the same pixels. **The evaluator will not rescale or auto-register images.** A mismatched crop is a failed test setup, not a PBR failure.
5. Create a binary material ROI mask for each held-out condition (white = valid pixels). Use identical ROI masks for all candidates. Avoid cast shadows/background/occlusions, and ensure shared ROI for alternate-light images at a single camera. Save photos, masks and renders under the gitignored `data/` directory only; do not commit private samples.
6. Check the color pipeline (sRGB transfer, tonemapping, exposure, WB), surface normals, and lighting intensity. Our metrics are image-domain diagnostics, not calibrated BRDF/reflectance estimation. For meaningful physical conclusions, repeat with calibrated lighting and a known reference material.

## Comparable variants

| Candidate | StableGen/loop treatment | Held constant |
| --- | --- | --- |
| `baseline` | Same model (e.g. Qwen), no external photo conditioning | Mesh/UVs, camera, light, seed, output size, bake settings |
| `photo` | Enable existing Qwen external style image; provide only conditioning photos | Everything else |
| `refined` | Start from photo variant, manually identify failed material patch, apply one masked edit / tile refinement per DreamLoop round | Same held-out views; no test image as target |

Baseline is **not** SDXL against Qwen photo mode: model substitution would confound the experiment. StableGen's current seams include `stablegen/texturing/workflows.py` (Qwen external style image), `stablegen/texturing/pbr.py` (per-channel extraction), `stablegen/texturing/pbr_projection.py` (project PBR through existing UVs), `stablegen/texturing/view_plan.py` (camera manifests), and `stablegen/texturing/generator.py` (Reproject from stored views). Existing `Regenerate`, `Masked Edit`, and `Reproject` can be used manually; **do not replace generated-image files in place** or confuse saved per-camera images with held-out reference photos. Duplicate the `.blend` and StableGen revision directory for every candidate.

## CLI and reproducibility

```bash
python -m pip install 'numpy>=1.24,<3' 'Pillow>=10,<12'
cp experiments/photo_material/capture_manifest.example.json experiments/photo_material/manifest.json
# Edit manifest.json; put photos/ROI masks/renders into experiments/photo_material/data/
python experiments/photo_material/benchmark.py experiments/photo_material/manifest.json --check
python experiments/photo_material/benchmark.py experiments/photo_material/manifest.json \
  --report experiments/photo_material/runs/wood.json \
  --markdown experiments/photo_material/runs/wood.md \
  --diff-dir experiments/photo_material/runs/diffs
python -m unittest discover -s experiments/photo_material -p 'test_*.py' -v
```

`--check` checks manifest **structure only**; it does not assert the photos exist or that a real render has been measured. A nonzero evaluation exit signals invalid, missing, leaked or misaligned inputs; do not report a pass. Capture paths and candidate roots are relative to the manifest folder and cannot escape it. The `example.json` is a **template**, not a measured dataset. Real `.blend` files, camera captures and rendered images are not present in this repository.

`benchmark.py` compares masked, full-resolution **color MAE** in sRGB, 4-pixel Gaussian-blurred **structure MAE**, and masked high-pass **detail MAE**. Same-camera alternate-light differences use **linearized sRGB**, comparing `(photo light A − photo light B)` to `(render light A − render light B)` in the common valid ROI; this light-response diagnostic requires exposure and white balance locked. It is only computed for held-out pairs with identical camera IDs, and skipped with a warning otherwise. All errors are normalized `0..1` (lower is better). `paired_improvement_vs_baseline` is baseline error minus candidate error: **positive means improvement**. Reports include per-condition reference/render SHA-256, counts, novel held-out views and diagnostics; no unconditional pass/fail threshold is invented.

Known limitations: accurate pixel registration is a manual precondition; absolute color accuracy requires calibrated capture/display; motion, variable lens corrections, ICC differences, uncalibrated light sources, specular anisotropy, scale and geometry mismatches confound pixel metrics; exact decoded-image hashes catch duplicate leakage but not near-duplicates or deliberate training exposure. Detailed texture can score poorly after a one-pixel shift. Use material experts and held-out photographic comparisons as additional checks.

## DreamLoop integration after E0

Use a single locked material region as the target; first gate camera/mesh alignment. On each material-only round, preserve existing `target.png`, append an intervention to DreamLoop's `.dream-loop/rounds.jsonl` using `scripts/loop-state.mjs` in [kvnloo/dream-loop](https://github.com/kvnloo/dream-loop), and attach the generated `runs/*.json`, `runs/diffs/*` paths as artifacts. Record cost, reject or rollback worsening branches. **Never ingest held-out photos into the agent's generation/critique context**; the offline evaluation gate can return scalar measurements or an approval decision without exposing test pixels. Keep the GPU/image-generation process separate from this tool until repeatable E0 results exist.

## Completion gate

Three material datasets, three model-controlled variants each, at least two held-out camera positions and a two-light pair for each sample, calibrated captures when claiming reflectance behavior, example image grids, per-view residual previews, exact software/model revisions, seconds and peak VRAM. Compare the same candidates with and without the adaptive patch loop. All example numbers must come from actual GPU runs, not the synthetic unit tests.
