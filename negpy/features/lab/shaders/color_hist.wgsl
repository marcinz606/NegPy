@group(0) @binding(0) var input_tex: texture_2d<f32>;
// 32 * 32 * 32 joint RGB bins, indexed (r * 32 + g) * 32 + b. Mirrors COLOR_HIST_BINS
// and color_histogram() in features/exposure/analysis.py.
@group(0) @binding(1) var<storage, read_write> bins: array<atomic<u32>, 32768>;

const BINS = 32.0;

// Working-space OETF (Adobe RGB: pure 563/256 gamma). Mirrors output_encode.wgsl.
fn oetf_encode(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    return pow(x, vec3<f32>(0.45470693));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    // Same encode as metrics.wgsl, so the joint bins and the marginal RGBL histogram
    // describe one image.
    let color = oetf_encode(textureLoad(input_tex, coords, 0).rgb);
    let q = vec3<u32>(clamp(floor(color * BINS), vec3<f32>(0.0), vec3<f32>(BINS - 1.0)));

    atomicAdd(&bins[(q.r * 32u + q.g) * 32u + q.b], 1u);
}
