# Upstream proposal (not posted: GitHub integration lacks upstream issue-write permission)

**Suggested issue title:** Photo-grounded PBR refinement and multi-view/light validation (smartphone / DSLR references)

## Problem

StableGen can project generated views and estimate PBR channels, but a diffusion *style reference* does not guarantee reproduction of actual photographed microstructure or reflectance. Texture sharpness can mask projected smearing, baked-in shadows, incorrect surface roughness or mismatched responses under changed lighting.

Users could supply one or more high-resolution phone/DSLR photographs of a real material (wood, stone, metal or fabric) to reproduce its actual appearance on existing meshes, rather than merely matching the stylistic category.

## Suggested opt-in solution

1. **Capture input & provenance:** associate source photos with a material/mesh, known camera or surface; optional RAW export, intrinsics, physical scale, color/gray card, white balance, light condition and data provenance. Keep unshared photos local.
2. **Registration:** segment the material and align the reference to surface UV or camera projection. Reject masks over occlusions or grazing angles, instead of bleeding detail onto wrong surfaces.
3. **PBR:** use photo evidence for albedo; estimate roughness/normal/height only with explicit confidence or multiple lighting conditions. Delighting is needed to avoid hard-coded photographic shadows. A single RGB photo alone cannot uniquely determine the full BRDF.
4. **Refinement loop:** combine existing `Masked Edit`, `Reproject`, per-view confidence/blending, and PBR map caching to improve a localized material patch, then test it on fixed held-out camera/light renders. Preserve old versions; record all attempted interventions and rollback decisions.
5. **Additional capture requests:** when insufficient evidence remains, suggest a specific new angle or lighting direction rather than hallucinating physical measurements.

## Small upstream-compatible milestones

- **E0:** downstream-only offline benchmark: three sample materials, StableGen baseline vs existing external-style-image conditioning vs iterative masked edit. *Same model, seed, geometry, camera and bake settings*. Hold out photos from generation entirely; compare masked color/structure/detail and unseen multi-light response. No core API changes.
- **E1:** reversible single-material registered-photo intake plus explicit reference manifest, isolated output and Reproject interop.
- **E2:** confidence-masked selective refinement and replayable edit decisions.
- **E3:** optional active capture recommendations and advanced physical calibration.

Potential integration points verified in current sources: `stablegen/texturing/workflows.py` (external Qwen references), `pbr.py`, `pbr_projection.py`, `view_plan.py`, `quality_masks.py`, `surface_blend.py`, `regression.py`, plus the `Reproject` operator in `generator.py`.

Related existing requests: [#32 (custom ComfyUI workflows)](https://github.com/sakalond/StableGen/issues/32), [#64 (baked-in lighting concern, closed)](https://github.com/sakalond/StableGen/issues/64). This is a distinct feature about material-specific real-photo grounding and independent validation, not a request to reimplement general custom-workflow plumbing.

## Relevant addon functionality

- [x] Generating Textures (Prompts, AI models, Core process)
- [x] Scene Setup (Cameras, Object preparation)
- [x] Controlling Texture Appearance (ControlNet, IPAdapter)
- [x] Texture Application & Blending
- [x] Managing Settings & Presets
- [x] Other: calibration, provenance, confidence and evaluation

## Alternatives considered

- Qwen external style image alone: useful baseline, but not calibrated material matching.
- Tiled super-resolution: may look sharper without changing physical correctness.
- Full material scanner or photogrammetry requirement: better constraints but excessive for a lightweight opt-in starting point.

## Evidence planned

Downstream tracker: https://github.com/kvnloo/StableGen/issues/1

Provide held-out views, image grids, per-condition error and VRAM/time receipts before asking for upstream API changes. No quantitative claims are made without completed photographic experiments.
