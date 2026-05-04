# Batch Normalization Progress Bar

**Goal:** Show a live progress bar during batch normalization, matching the existing export progress UX.

**Root cause:** `NormalizationWorker.process()` runs all files in parallel via `asyncio.gather`, then emits all progress signals in a post-hoc loop after the batch completes. All signals fire at once → bar flashes and disappears before the user sees it.

**Fix:** Emit progress inside `_analyze_file` as each file finishes (success or failure), using a shared counter. Remove the post-hoc loop. Add a "batch analysis complete" status message in `_on_normalization_finished`.

**No new signals needed** — `status_progress_requested` and `set_progress` already handle show/hide automatically (bar auto-hides when `current >= total`).

---

## Files

- Modify: `negpy/desktop/workers/render.py` — emit progress during processing
- Modify: `negpy/desktop/controller.py` — add completion status message

---

## Changes

### 1. `negpy/desktop/workers/render.py` — `NormalizationWorker.process()`

Add `completed = [0]` counter before `_analyze_file`. At the end of `_analyze_file` (both success and except branches), emit:
```python
completed[0] += 1
self.progress.emit(completed[0], total, f_info["name"])
```

Remove the post-hoc loop:
```python
# DELETE:
for i, (_, _, name) in enumerate(valid_results, start=1):
    self.progress.emit(i, total, name)
```

### 2. `negpy/desktop/controller.py` — `_on_normalization_finished()`

After applying settings, add:
```python
self.set_status("batch analysis complete", timeout=3000)
```

---

## Verification

```bash
make all
```

Manual: load 5+ files → click Batch Analysis → progress bar appears and fills file-by-file → bar disappears, status shows "batch analysis complete".
