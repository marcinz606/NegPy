# IR/Auto Dust Removal — infrared-cleaning quality upgrade plan

Status: **implemented 2026-07-15** on `feat/heal` (all 4 phases + tests, `make all` green; verified end-to-end on the Ektar sample — γ auto-fit 2.1–2.2, smooth no-cliff slider sweep, 32 defects at default). Deviations from the plan below:
- **No "Digital ICE" / "ICE" naming** anywhere user-facing (Nikon trademark) — the toggle is **"IR Restore"**; code/comments say "infrared-cleaning" / "IR-division attenuation". The config field stays `ir_attenuation`.
- Overlay lost the blank-frame **"Spots"** mode: cycle is now **Off → Marked → IR**.
- Retouch sidebar **redesigned into three source-grouped sections** (AUTO DUST / MANUAL HEAL / IR DUST) with the Overlay inspector at the bottom; manual-heal count merged into the section header (`MANUAL HEAL · N`).
- CLAUDE.md was **not** updated (user reverted the edit).

## Context

IR dust removal is effectively broken; goal is Nikon Digital ICE quality. Reported issues → verified root causes:

| # | Issue | Root cause (verified in code + measured on `samples/20260619SP_EKTAR100_120_1_09_ME_4000PPI.tif`) |
|---|---|---|
| 1 | IR heals = big hard clone blobs | Detection at preview scale merges specks into big components; pad inflates radius; gate=0 clones full capsule with only `0.25·rad` rim feather; hue-blind source interior cloned verbatim |
| 2 | Misses obvious dust | `detect_ir_regions` thresholds **raw** IR (`ir < 1−slider`). Clean content sits at 0.72–0.77, dust dips to 0.53 → default cutoff 0.5 catches ~nothing (p0.01=0.534) |
| 3 | Slider cliff destroys preview | Raw-IR usable band is razor thin (0.68→0.72 = 0.13%→1.6% coverage). Below it the whole frame crosses threshold → giant capsules + golden-angle fallback offsets → whole-frame smear. No coverage guard exists |
| 4 | Wrong clone source | Auto/IR use `_pick_source_offsets` (`logic.py:466`): single-channel density proxy, mean+std only, 4–6 dirs × 3 rings, direction-blind golden-angle fallback. Manual strokes use the much better RGB-SSD `select_source_offset` |

## Research: how Digital ICE actually works

Patent US5266805 (Edgar / Applied Science Fiction) + US6614946:

- 4-channel RGBI capture; dust/scratches are **colourless** → attenuate IR and visible equally (multiplicatively); dyes are IR-transparent → IR plane = pure "dust map".
- **Dye crosstalk normalization**: cyan dye absorbs some IR (worst on C-41 negatives). Patent: 4×4 matrix in cube-root domain. Practical equivalent: normalize IR against a local clean-base estimate (and/or regress out red density).
- **Two-tier correction** — the key quality difference vs clone-based healing:
  - Semi-transparent defects (fringes, fine dust): **attenuation division** `RGB_corrected = RGB / IR_norm^γ` per pixel (patent γ ≈ 1.03/1.06/1.10 R/G/B refraction compensation). Recovers the *actual image* under translucent dust — zero fabricated content → invisible result.
  - Opaque cores (IR below noise threshold): spatial fill only there; gain-corrected fringe pixels participate in the estimate.
- **Fill-in** (US6614946): adaptive region grown until enough non-defective pixels (`NDP = 100/(1+A/25)`), per-pixel reliability `R = 1−K·D`, weighted median/average across quadrant subregions — surround-fill, not single-offset clone.

Sources: [US5266805](https://patents.google.com/patent/US5266805A/en) · [US6614946](https://patents.google.com/patent/US6614946B1/en) · [Digital ICE](https://en.wikipedia.org/wiki/Digital_ICE) · [Infrared cleaning](https://en.wikipedia.org/wiki/Infrared_cleaning)

## Current implementation map (explored 2026-07-15)

### Detection path

- `ImageProcessor._augment_retouch` (`negpy/services/rendering/image_processor.py:120-191`): gates, one-slot detect cache (key: source_key + flags + thresholds + dust_size + ir_inpaint_radius + process_mode), stats sub-cache `(source_key, dust_size)`, detection at `preview_render_size` via `_detection_downsample` (INTER_AREA). Budget `512 − manual counts`, positional truncation (IR first, luma dropped wholesale). Result merged render-locally into `manual_heal_strokes`, auto flags cleared (synth strokes never persist).
- **IR detect** = `detect_ir_regions(ir_ds, 1.0 − ret.ir_threshold, pad_px=ir_inpaint_radius, guide=stats[0])` (`negpy/features/retouch/logic.py:599-618`): literally `mask = ir < threshold` → `_mask_to_strokes` → `_pick_source_offsets` → `_finalize_strokes(gate=0)`. **No normalization** (no flat-field, no divide-by-red/blur, no registration — only dtype scale + orientation + resize). **No structural guards**. **No coverage/area limit.**
- **Luma detect** `_detect_dust_mask_jit` (`logic.py:360-399`): local contrast vs box means; guards: floor `l>0.15`, 3σ ratio, cubic wide-window texture penalty ×800, local-max requirement. Runs on `_detection_proxy` (percentile-normalized density, `logic.py:402-411`); `compute_dust_stats` (`logic.py:563-578`).
- `_mask_to_strokes` (`logic.py:414-463`): connectedComponents; every area≥1 becomes a stroke; round → point, radius=`sqrt(area/π)+pad_px`; elongated (ext≥2.5·w and ≥8) → ≤8-pt PCA capsule radius=`half_w+pad`; keep-largest 512 per source.
- `_pick_source_offsets` (`logic.py:466-543`): integral-image box stats on single-channel guide, score `|Δmean| + max(0, std−dest_std)`, 4–6 dirs × rings (2.6/3.6/4.6)·r, golden-angle blind fallback. **Hue-blind.**
- `_finalize_strokes` (`logic.py:546-560`): source-normalized points, `size = 2·radius·HEAL_SIZE_REF/max(w,h)` (`HEAL_SIZE_REF=1600`).
- IR channel sources: 4-ch LinearRaw DNG (`rawpy_loader.py:19-87`), TIFF extrasamples / `_IR` sidecar / SilverFast page-2 (`tiff_loader.py`), SANE. Preview: `preview_manager.py:157-201` orients+resizes → `state.preview_ir` → `run_pipeline(ir_buffer=...)`; export: `_load_source_f32` returns `ir_full` (used at `image_processor.py:403`, `:609`).
- Overlay (`negpy/desktop/view/canvas/overlay.py:654-727`): Spots/Marked draw the **synthesized stroke lists** from metrics `detected_dust_luma/ir` (pre-budget-cap — can show strokes that never heal); IR mode draws **raw IR transmittance grayscale** (uv_grid-remapped), not the threshold mask → "visible on overlay but missed" = gray dip above threshold.
- UI (`negpy/desktop/view/sidebar/retouch.py`): dust_threshold 0.01–1.0 (def 0.66), dust_size 3–8 (def 4), ir_threshold 0.05–0.95 (def 0.5); `ir_inpaint_radius` (def 3) not exposed — reused as detect pad.

### Heal path

- `_membrane_heal_jit` (`logic.py:140-299`): true MVC membrane clone `out(p)=img(p+off)+Σŵ_i(img(b_i)−img(b_i+off))`; rim feather only outer `0.25·rad` (≥1.5px), interior hard clone; alpha compositing; dust gate smoothstep 0.04..0.12 (gate=1 bright-only) vs unconditional (gate=0); boundary ring at `radius+2px` (`_MEMBRANE_RIM_PX`), 3×3/5×5 dust-guard sampling. CPU brackets working-OETF (`apply_manual_heals` `logic.py:796-812`); GPU solves linear (no bracket) — pinned by loose parity test atol 0.15 (`tests/test_pipeline_parity.py:509-580`).
- Regions: shared `build_heal_regions` (`logic.py:~640-690`); GPU upload `_update_retouch_storage` (`gpu_engine.py:1319-1369`); WGSL mirror `negpy/features/retouch/shaders/retouch.wgsl` structurally identical.
- Two source-offset scorers: manual → `select_source_offset` (`logic.py:704-793`, RGB SSD rim band + interior-structure penalty — good); auto/IR → `_pick_source_offsets` (hue-blind, see above) → issue 4.
- Blob mechanism (issue 1) = merged/padded capsules × gate=0 full-capsule clone × thin rim feather × hue-blind source interior cloned verbatim (membrane corrects only the boundary).

### Measurements (Ektar C-41 sample with real IR plane, 1600px detection scale)

- Raw IR: clean content 0.72–0.77 (median 0.740), dust dips ~0.53; corr(IR, red)=0.29 (dye crosstalk). Default slider misses nearly everything; raw threshold sweep 0.68→0.72 explodes 0.13%→1.6% coverage.
- Normalized ratio `ir / blur(dilate(ir, 25px rect), 25px)` (9.6 ms at 1600px): clean ≈ 0.98–1.0; cutoff 0.80→0.95 sweeps coverage 0.028%→0.37% smoothly, comps 165→2426, max comp ≤196px — wide stable band, image-independent.
- Best-fit attenuation γ (log-log LS, visible-ratio vs IR-ratio on the semi-transparent band): **R 1.69 / G 1.78 / B 1.92** — patent constants (1.03–1.10) recover only ~half the dip on file-based scanner IR → γ must be auto-fitted per frame, clamp [1.0, 2.2], fallback 1.5.
- Residual after division at best γ: σ ≈ 0.10 (local-ratio units) — division restores level, not pixels; deepest semi-transparent dust fades to near-invisibility, not bit-perfect.
- Opaque-core coverage (ratio<0.5) ~0.0015% → a handful of strokes.
- Full-res costs (8780×6499): dilate 400 ms, blur 81 ms, div 124 ms, 3-ch multiply 169 ms → compute gain at detection scale, upsample at apply (det-scale vs full-res gain agrees to mean |Δ| 0.009, p99 0.05 on dust). Export overhead ≈ +0.3–0.5 s, ~+1.3 GB transient.

### Verified architecture facts the plan rests on

- Three `_augment_retouch` chokepoints: `run_pipeline` (`image_processor.py:220`), `process_export` (`:403`), `render_display_array` (`:609`) — each has buffer + IR + `detect_key` in hand.
- `_augment_retouch` clears `ir_dust_remove` on the render-local config → nested CPU-fallback `run_pipeline` calls are naturally idempotent for a bake gated on that flag.
- GPU preview re-uploads the source texture every frame (`run_pipeline` never passes `source_hash` to `process_to_texture`, only `analysis_source_hash` → upload branch `gpu_engine.py:433`) → **CPU-baked correction reaches the GPU with zero WGSL work, parity-free**.
- Latent bug found (Phase 4): CPU export fallback double-applies flat-field — `_load_source_f32:372` bakes it, then nested `run_pipeline:211` bakes again (`process_export:420`, `render_display_array:617`).

Decisions (user-confirmed 2026-07-15): all 3 phases in one PR on `feat/heal`; overlay shows cores + dim tint for division-corrected regions.

---

## Phase 1 — Ratio-normalized IR detection (fixes #2, #3)

`negpy/features/retouch/logic.py`:
- New `normalize_ir(ir)`: `base = cv2.blur(cv2.dilate(ir, RECT(25,25)), (25,25)); return ir / np.maximum(base, 1e-4)` — `_IR_BASE_WIN = 25` px at detection scale (fixed like `HEAL_SIZE_REF`; rect kernel = separable van Herk max-filter, O(1)/px). Clean film ≈ 1.0 everywhere, defects dip; image/illumination independent. Ceiling: defects wider than ~12px radius at 1600 depress their own base — max-area guard / Scratch-tool territory (document in constant comment).
- New `ir_detect_cutoff(slider, attenuation: bool)` pure mapping (lower slider = catches more, direction unchanged):
  - attenuation **on**: `cutoff = 0.85 − 0.40·slider` (core band; default 0.35 → 0.71 ≈ gain-clamp handover `2.0^(−1/1.8) ≈ 0.68`)
  - attenuation **off**: `cutoff = 0.95 − 0.20·slider` (detection band; 0.35 → 0.88, measured stable)
- `detect_ir_regions(ratio, cutoff, pad_px, max_n, guide)` — takes the precomputed **ratio plane** and resolved cutoff (the `1−ir_threshold` inversion at the call site dies). Guards: coverage abort `mask.mean() > 0.01` → log warning + return `[]` (never destroy the preview); component guards via new `_mask_to_strokes(min_area=2, max_area=2000)` params (luma path passes neither — unchanged behavior).
- Module logger (logic.py has none yet).

`negpy/services/rendering/image_processor.py`:
- One-slot ratio cache beside `_dust_stats_*`, keyed `source_key` only → threshold drags reuse ratio + downsample (~11 ms/step saved). Detect-cache key already contains `ir_threshold`; add `ir_attenuation`.

`negpy/features/retouch/models.py`: `ir_threshold` default 0.5 → **0.35** (comment tying 0.71/0.68 handover).

`negpy/desktop/view/sidebar/retouch.py`: tooltip only — sensitivity on locally-normalized IR, smooth response. Range/label/direction unchanged.

## Phase 2 — ICE attenuation division tier (fixes #1)

All CPU-side, pre-engine, **source transmittance space** — the only domain where dust is a clean multiplication (negative-density space: additive offset; post-inversion/print-curve: not separable). Mirrors the `apply_flatfield` bake pattern. **Threshold-free** — slider drags never re-bake. Bonus: dust no longer skews exposure meters/auto-bounds.

`negpy/features/retouch/logic.py` — pure functions:
- `ir_ratio_and_gain(ir_det, img_det) -> (ratio_det, gain_det HxWx3, degenerate, gammas)`:
  - `degenerate = mean(ratio_det < 0.90) > 0.05` (B&W silver / Kodachrome: IR sees image content → skip bake AND strokes; real-dust ceiling measured 0.10–0.4% → ≥12× margin)
  - γ fit per channel on the semi-transparent band `0.70 < ratio < 0.92`: `γ_c = clip(Σ(x·y)/Σ(x·x), 1.0, 2.2)` with `x = log(ratio_det[band])`, `y = log(vis_ratio_c[band])` (`vis_ratio_c` = same dilate+blur ratio on channel c of `img_det`, ~35 ms once per source); fallback 1.5 when band < 500 px
  - `gain_det_c = minimum(2.0, clip(ratio_det/0.97, 0, 1)^(−γ_c))` — identity ≥0.97 (soft self-gate, global patent-style correction, no seams; clean-pixel harm ≤ ~2%), clamp 2.0 caps misregistration halos; detection-scale provenance = free ~5px pre-blur at 4000ppi (the "IR PSF match" mitigation)
- `apply_ir_attenuation(img, gain_det)` — `img * cv2.resize(gain_det, (w,h), INTER_LINEAR)`, returns new array (preview buffers are read-only by contract)
- `ir_bake_token(retouch, has_ir) -> str` — `"|irdiv1"` when active, else `""` (mirrors `flatfield_token`; config identity only — content identity already rides in `source_hash`)
- Upgrade paths as `ponytail:` comments: regress red density out of IR; 2–4× detection-scale ratio for sharper correction boundaries (~650 ms measured).

`negpy/features/retouch/models.py`: `ir_attenuation: bool = True` (escape hatch; `asdict`/`filter_keys` serialize automatically; `from_flat_dict({})` backcompat via dataclass default).

`negpy/services/rendering/image_processor.py`:
- One-slot gain cache keyed `(source_key, ir_det.shape)` holding `(ratio_det, gain_det, degenerate, gammas)` — this **is** Phase 1's ratio cache (one cache, both consumers).
- `_ir_bake(img, ir_buffer, settings, source_key)` gated on `ir_dust_remove and ir_attenuation and ir_buffer is not None and not _is_flat and not degenerate`.
- Wire the three chokepoints: fold `ir_bake_token` into `base_hash` (`run_pipeline:217`) / `detect_key` (`process_export:402`, `render_display_array:608`); bake **before** `_augment_retouch` (detection stats/guide then see the corrected buffer). Nested fallback `run_pipeline` calls: flag already cleared → no double-apply.
- `_augment_retouch`: IR strokes from `detect_ir_regions(ratio_det, ir_detect_cutoff(ret.ir_threshold, ret.ir_attenuation), ...)`; skip when degenerate; emit `metrics["ir_degenerate"]` and `metrics["ir_corrected_mask"]` (`ratio_det < 0.97` bool at detection scale, only when bake active).

Desktop:
- Overlay (`negpy/desktop/view/canvas/overlay.py`): in Marked/Spots modes draw `ir_corrected_mask` as a dim magenta-tinted layer remapped through `uv_grid` (reuse the `_ir_layer_qimage` remap machinery + its id-cache pattern), core capsules on top as today.
- `controller.py` `_on_metrics_updated`: copy `ir_degenerate` to `AppState`; sidebar hint on the IR toggle ("IR channel carries image content (B&W/Kodachrome) — IR correction disabled").
- Small `ir_attenuation` toggle in the IR row (checkable QPushButton per house style).

Cache-invalidation behavior (design intent — verify while implementing):

| Event | gain/ratio cache | detect cache | engines |
|---|---|---|---|
| `ir_threshold` drag | hit | miss → components re-run on cached ratio (~ms) | retouch stage only |
| `ir_dust_remove`/`ir_attenuation` toggle | hit | miss | token flips → source_hash change → full re-render (checkbox — acceptable) |
| file switch | recompute once (~50 ms det scale incl. γ fit) | recompute | new source |
| export | same det-scale gain map reused (**WYSIWYG: preview and export share one gain map by construction**) | hit (resolution excluded from key by design) | corrected buffer flows into GPU/tiles/meters |

## Phase 3 — Inpainting quality (fixes #4 + residual blob softness; benefits auto-luma too)

`negpy/features/retouch/logic.py`:
- `compute_dust_stats` returns 5th element `proxy_rgb` = per-channel density normalized with the **same** lo/spread as the luma proxy (channels comparable, exposure-invariant); refactor `_detection_proxy` to share the percentile math. Cached in the existing stats sub-cache for free.
- `_pick_source_offsets` channel-generic: `guide.ndim==2` → 1-channel path (existing tests pass unchanged); 3-channel → per-channel `cv2.integral/integral2`, `score = Σ_c |Δmean_c| + Σ_c max(0, std_c − dest_std_c)` — per-channel Δmean in density space is exactly the wrong-colour detector. Callers pass `guide=stats[4]` (cost ~2 extra integral images ≈ 1 ms each at 1600px).
- More candidates: compact comps 8 directions (golden-angle-seeded + k·π/4); capsules perp/along + perp-diagonals; rings `(2.6, 3.6, 4.6, 6.2)·r`. If pass 1 (mask-free, in-bounds) empty: **pass 2** re-scores all candidates + rings `(8, 11)·r` with `score += 10·mask_overlap_fraction` instead of hard rejection — content-scored pick guaranteed; golden-angle `fallback_source_offset` only when every window is off-image (tiny test frames).
- Wider rim feather for ungated regions via the existing gate lane (zero layout change):
  - CPU `_membrane_heal_jit`: `fth = max((0.25 + 0.15·(1−gate))·rad, 1.5)`
  - WGSL `retouch.wgsl` (~line 173): `let fth = max(mix(0.4, 0.25, reg.gate) * reg.radius, 1.5);`
  - 0.4 (not 0.5) stays on the pad ring (pad 2.5–3 px ≥ 40% of synthesized radii) — never half-heals the defect core. `ponytail:` tune 0.35–0.5 by eye; per-region feather lane if ever needed.

`negpy/services/rendering/image_processor.py`: cap truncation becomes size-priority across the merged list: `synth = sorted(ir + luma, key=lambda s: -s[1])[:budget]` (today IR head-truncates luma wholesale). Overlay lists stay pre-cap.

Skipped deliberately: reliability-weighted 2-source blend (needs a second offset lane through `reg_f`/WGSL struct/upload — revisit only if wrong-source reports persist); erode-to-core (deletes the 1–2 px specks that are the missed-dust complaint); multi-circle clump decomposition; dedupe of double-detected dust (second membrane over clean pixels is harmless).

## Phase 4 — Flat-field double-apply fix (latent bug, found during investigation)

`run_pipeline` applies `apply_flatfield` unconditionally (`image_processor.py:211`); `process_export`/`render_display_array` CPU fallbacks pass an already-flatfielded `f32_buffer` into it → photometric correction applied **twice** when a profile is active. Fix: `run_pipeline(..., skip_flatfield=False)` kwarg, passed `True` from the two fallback sites. (Cannot strip `flatfield` from params instead — `apply` also gates geometry k1.) Regression test: flat-field profile active, CPU export fallback → pixel-equal to single-apply.

## CLAUDE.md

Rewrite the retouch paragraph: raw-IR threshold sentence → ratio normalization + two-tier attenuation/core split + degenerate guard + `ir_attenuation` field; note the gain bake at the three chokepoints and the token.

## Tests

Update (`tests/test_ir_dust.py`):
- `test_detect_ir_regions_threshold_inversion_convention` → replaced by ratio-mapping test (`ir_detect_cutoff` both modes + direction: lower slider catches more).
- `test_detect_ir_regions_heals_defect_end_to_end`, `_no_defect_is_empty` → construct ratio input (via `normalize_ir`), pass UI slider values.

New:
- `normalize_ir` flat plane → ≈1.0 everywhere; dust dip on a strong illumination gradient detected at default (headline sensitivity fix — old absolute logic misses it); coverage abort at 50% → `[]`; min/max component-area guards (1px noise dropped, >2000px² blob skipped, normal speck beside it survives).
- Gain: identity ≥0.97, clamp at 2.0, continuity at the gate; γ fit recovers synthetic known γ ±0.05 and clamps; degenerate guard (IR=luma image → True, sparse dips → False).
- Two-tier end-to-end (CPU `run_pipeline`): semi-transparent speck (ratio ~0.85) vanishes with `detected["ir"]` empty (no stroke); opaque core (ratio ~0.1) → exactly one stroke + membrane heal.
- `ir_bake_token` empty/active; `ir_attenuation` hash invalidation (mirror `test_ir_dust_remove_field_invalidates_retouch_hash`); `from_flat_dict({})` backcompat.
- Cap size-priority (`tests/test_dust_overlay.py`): monkeypatch detectors, tiny budget → survivors are the largest across ir+luma.
- Ungated feather wider (`tests/test_retouch_logic.py`): identical spot gate=0 vs gate=1 → outer-rim |out−img| strictly smaller for gate=0.
- Source pick: 3-channel guide picks colour-matched over luma-matched candidate; ring-surrounded defect → lowest-overlap pass-2 pick (never blind golden-angle).
- Flat-field single-apply regression (Phase 4).

Surviving pins (stay green): `test_detect_ir_regions_speck_ungated`, capsule geometry (`test_detect_ir_elongated_scratch_becomes_capsule`), `HEAL_SIZE_REF` radius math (`test_heal_radius_matches_cursor_fraction`), `_pick_source_offsets` mask-free/detail-avoid (2-D path), `TestRetouchParity` (atol 0.15 absorbs the feather change; gate=0 case already exercised), `test_retouch_config_defaults_include_ir_fields` (0.35 in range), `test_dust_overlay.py` suite.

## Verification (headless driver + `make all`)

`QT_QPA_PLATFORM=offscreen NEGPY_USER_DIR=<scratch>` on `samples/20260619SP_EKTAR100_120_1_09_ME_4000PPI.tif` (real IR plane), per the headless-verify pattern (compose repo → AppController → MainWindow; `update_config_section`; `window.grab()` evidence):
1. IR Dust off → on: grabs; thousands of shallow specks gone with only a handful of core capsules in Marked overlay; corrected-mask tint aligns with visible dust on the raw-IR layer.
2. Sweep `ir_threshold` 0.05/0.3/0.5/0.7/0.95: `state.last_metrics["detected_dust_ir"]` count monotone, no step >3× previous (cliff regression), no giant capsules at any stop.
3. Heal-diff: render IR on vs off, diff-mask components — max changed-region diameter bounded by largest detected size × scale + feather (no blob explosion); healed patch mean RGB vs 2-radius surround (wrong-colour regression).
4. Export 16-bit TIFF, crop a dust neighbourhood, compare against preview crop (shared gain map → matching correction).
5. `ir_attenuation=False` → degrades to conservative detection+membrane behavior. Substitute IR plane with luma → degenerate log line, no correction, sidebar hint.
6. `make all` green.

## Risks

- **γ auto-fit is the load-bearing novel piece** — patent constants measured wrong for file-based IR (1.7–1.9 vs 1.03–1.10); if the fit proves unstable across a roll, fall back to fixed 1.7 (still ~90% of dip recovery vs ~50% at patent γ).
- Division restores level, not pixels — σ≈0.1 residual grain-scale texture on the deepest semi-transparent dust; cores + membrane cover those. Don't promise bit-perfect invisibility.
- Saved edits with IR enabled will heal (much) more — old default detected ~nothing; deliberate, flag in commit message.
- `_IR_BASE_WIN=25` ceiling: defects wider than ~12px radius at 1600 depress their own base → max-area guard / manual Scratch tool.
- Upsampled gain blurs correction boundaries ~5px at 4000ppi; upgrade path = 2–4× detection-scale ratio (~650 ms measured).
- Badly misregistered or absent IR: feature inert or worse → `ir_attenuation=False` escape hatch degrades to the conservative design.

## Critical files

`negpy/features/retouch/logic.py` · `negpy/features/retouch/models.py` · `negpy/services/rendering/image_processor.py` · `negpy/features/retouch/shaders/retouch.wgsl` · `negpy/desktop/view/canvas/overlay.py` · `negpy/desktop/view/sidebar/retouch.py` · `negpy/desktop/controller.py` · `tests/test_ir_dust.py` · `tests/test_retouch_logic.py` · `tests/test_dust_overlay.py` · `CLAUDE.md`
