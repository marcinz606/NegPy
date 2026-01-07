# Plan: Split Working Copy Size by Orientation

## Objective
Persist separate `working_copy_size` settings for vertical and horizontal images. The UI will continue to use a single slider, but its value will sync with the appropriate persistent setting based on the current image's orientation (considering rotation).

## Proposed Changes

### 1. State Persistence (`src/presentation/state/state_manager.py`)
-   **`GLOBAL_PERSIST_KEYS`**: Add new keys:
    -   `"working_copy_size_vertical"`
    -   `"working_copy_size_horizontal"`
-   **`init_session_state`**:
    -   Initialize these keys if missing, defaulting to `APP_CONFIG.preview_render_size`.

### 2. Layout Logic (`src/presentation/layouts/main_layout.py`)
-   **Imports**: Import `SessionContext` for type hinting and access.
-   **`render_layout_header`**:
    -   Update signature to accept `ctx: SessionContext`.
    -   **Determine Orientation**:
        -   Get `rotation` from `st.session_state.get("rotation", 0)`.
        -   Get `(h, w) = ctx.original_res`.
        -   If `(h, w) == (0, 0)` (no file loaded), skip sync or assume default.
        -   Apply rotation logic: if `rotation % 180 != 0` (i.e. 90, 270), swap `h` and `w`.
        -   `is_vertical = h > w`.
    -   **Sync State**:
        -   Identify target key: `target_key = "working_copy_size_vertical" if is_vertical else "working_copy_size_horizontal"`.
        -   Ensure `st.session_state.working_copy_size` matches `st.session_state[target_key]`.
    -   **Slider Interaction**:
        -   Define a callback `update_orientation_size()` that writes the new `st.session_state.working_copy_size` value back to `st.session_state[target_key]`.
        -   Pass `on_change=update_orientation_size` to `st.slider`.

### 3. Application Flow (`src/presentation/app.py`)
-   **Pass Context**: Update `render_layout_header()` call to pass the `ctx` object.
-   **Reactive Loading**:
    -   In the file loading block (`controller.handle_file_loading`), if it returns `True` (indicating a new file or color space change), call `st.rerun()`.
    -   **Reason**: `handle_file_loading` updates `ctx.original_res`. `render_layout_header` runs *before* loading. To ensure the slider reflects the *new* file's orientation immediately, we need to rerun the script with the updated context.

## Logic Summary
```python
# Orientation Calculation
h, w = ctx.original_res
if rotation in [90, 270, -90, -270]:
    h, w = w, h
is_vertical = h > w

# Sync Logic (Pseudocode)
target_key = "working_copy_size_vertical" if is_vertical else "working_copy_size_horizontal"
if st.session_state.working_copy_size != st.session_state[target_key]:
    st.session_state.working_copy_size = st.session_state[target_key]

def on_slider_change():
    st.session_state[target_key] = st.session_state.working_copy_size
```
