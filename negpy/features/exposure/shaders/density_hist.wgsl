// Bins the normalized log-density ("val") signal from tex_norm over the crop ROI.

struct DensityHistUniforms {
    roi_offset: vec2<u32>,
    crop_dims: vec2<u32>,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> bins: array<atomic<u32>, 480>; // 120 * 4 (R, G, B, Luma)
@group(0) @binding(2) var<uniform> u: DensityHistUniforms;

const LUMA_COEFFS = vec3<f32>(0.2126, 0.7152, 0.0722);

// Mirror DENSITY_HIST_BINS / DENSITY_HIST_RANGE in features/exposure/analysis.py.
const BIN_COUNT = 120.0;
const VAL_MIN = -0.1;
const VAL_SPAN = 1.2;

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
}
