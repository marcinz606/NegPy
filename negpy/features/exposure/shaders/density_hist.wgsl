// Bins the normalized log-density ("val") signal from tex_norm over the crop ROI.

struct DensityHistUniforms {
    roi_offset: vec2<u32>,
    crop_dims: vec2<u32>,
    // The final bounds the normalization pass stretched with, to read the scan density back.
    floors: vec4<f32>,
    ceils: vec4<f32>,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> bins: array<atomic<u32>, 480>; // 120 * 4 (R, G, B, Luma)
@group(0) @binding(2) var<uniform> u: DensityHistUniforms;
// Scan-side RGB parade: each channel's scan density per crop column, 128 bins x 256 columns,
// indexed ch * 32768 + bin * 256 + col, bin 0 = the dense end of SCAN_DENSITY_RANGE. Mirrors
// density_parade() and WAVEFORM_BINS / WAVEFORM_COLS in features/exposure/analysis.py.
@group(0) @binding(3) var<storage, read_write> parade: array<atomic<u32>, 98304>;
// The linear scan before the unmix, and its per-column full-scale counts (ch * 256 + col).
// Mirrors scan_full_scale_columns().
@group(0) @binding(4) var geom_tex: texture_2d<f32>;
@group(0) @binding(5) var<storage, read_write> full_scale: array<atomic<u32>, 768>;

const LUMA_COEFFS = vec3<f32>(0.2126, 0.7152, 0.0722);

// Mirror DENSITY_HIST_BINS / DENSITY_HIST_RANGE in features/exposure/analysis.py.
const BIN_COUNT = 120.0;
const VAL_MIN = -0.1;
const VAL_SPAN = 1.2;
const PARADE_COLS = 256u;
const PARADE_BINS = 128.0;
// Mirror SCAN_DENSITY_RANGE: (hi, hi - lo).
const SCAN_D_MAX = 3.0;
const SCAN_D_SPAN = 3.0;
// Mirror EXPOSURE_CONSTANTS["scan_clip_level"].
const SCAN_CLIP_LEVEL = 0.99;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.crop_dims.x || gid.y >= u.crop_dims.y) {
        return;
    }
    let coords = vec2<i32>(i32(gid.x + u.roi_offset.x), i32(gid.y + u.roi_offset.y));
    let color = textureLoad(input_tex, coords, 0).rgb;
    let luma = dot(color, LUMA_COEFFS);

    let bin_r = u32(clamp((color.r - VAL_MIN) / VAL_SPAN * BIN_COUNT, 0.0, BIN_COUNT - 1.0));
    let bin_g = u32(clamp((color.g - VAL_MIN) / VAL_SPAN * BIN_COUNT, 0.0, BIN_COUNT - 1.0));
    let bin_b = u32(clamp((color.b - VAL_MIN) / VAL_SPAN * BIN_COUNT, 0.0, BIN_COUNT - 1.0));
    let bin_l = u32(clamp((luma - VAL_MIN) / VAL_SPAN * BIN_COUNT, 0.0, BIN_COUNT - 1.0));

    atomicAdd(&bins[bin_r], 1u);
    atomicAdd(&bins[120u + bin_g], 1u);
    atomicAdd(&bins[240u + bin_b], 1u);
    atomicAdd(&bins[360u + bin_l], 1u);

    let log_color = u.floors.xyz + color * (u.ceils.xyz - u.floors.xyz);
    let pos = (vec3<f32>(SCAN_D_MAX) + log_color) / SCAN_D_SPAN;
    let p_bins = vec3<u32>(clamp(floor(pos * PARADE_BINS), vec3<f32>(0.0), vec3<f32>(PARADE_BINS - 1.0)));
    let col = min(gid.x * PARADE_COLS / u.crop_dims.x, PARADE_COLS - 1u);
    atomicAdd(&parade[p_bins.r * PARADE_COLS + col], 1u);
    atomicAdd(&parade[32768u + p_bins.g * PARADE_COLS + col], 1u);
    atomicAdd(&parade[65536u + p_bins.b * PARADE_COLS + col], 1u);

    let raw = textureLoad(geom_tex, coords, 0).rgb;
    for (var ch = 0u; ch < 3u; ch = ch + 1u) {
        if (raw[ch] >= SCAN_CLIP_LEVEL) {
            atomicAdd(&full_scale[ch * PARADE_COLS + col], 1u);
        }
    }
}
