// Auto Masking meter, pass 1: bins the boxed L* gradient of the midtones, as the edge mask in
// lab.wgsl takes it. Mirrors sharpen_grain in negpy_core/src/lab.rs.
struct LabUniforms {
    sharpen: f32,
    chroma_denoise: f32,
    saturation: f32,
    glow_amount: f32,
    halation_strength: f32,
    scale_factor: f32,
    sharpen_radius_px: f32,
    sharpen_masking: f32,
    sharpen_method: f32,
    skin_protection: f32,
    chroma_radius_px: f32,
    auto_masking: f32,
};

// USM: (blurred L*, original L*). RL: (deconvolved Y, observed Y).
@group(0) @binding(0) var sharpen_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> hist: array<atomic<u32>, 1024>;
@group(0) @binding(2) var<uniform> params: LabUniforms;

const GRAIN_BINS = 1024u;
const GRAIN_BIN_SCALE = 256.0;
const GRAIN_L_LO = 30.0;
const GRAIN_L_HI = 85.0;

var<workgroup> local_hist: array<atomic<u32>, 1024>;

fn lab_l_from_y(y_in: f32) -> f32 {
    var y = max(y_in, 0.0);
    if (y > 0.008856) { y = pow(y, 1.0 / 3.0); } else { y = (7.787 * y) + (16.0 / 116.0); }
    return (116.0 * y) - 16.0;
}

fn plane_l(p: vec2<i32>) -> f32 {
    let v = textureLoad(sharpen_tex, p, 0).y;
    if (params.sharpen_method > 0.5) { return lab_l_from_y(v); }
    return v;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    for (var i = li; i < GRAIN_BINS; i += 64u) { atomicStore(&local_hist[i], 0u); }
    workgroupBarrier();

    let dims = textureDimensions(sharpen_tex);
    if (gid.x < dims.x && gid.y < dims.y) {
        let coords = vec2<i32>(i32(gid.x), i32(gid.y));
        let l = plane_l(coords);
        if (l > GRAIN_L_LO && l < GRAIN_L_HI) {
            let hi = vec2<i32>(dims) - 1;
            var grad_box = 0.0;
            for (var j = -1; j <= 1; j++) {
                for (var i = -1; i <= 1; i++) {
                    let p = clamp(coords + vec2<i32>(i, j), vec2<i32>(0), hi);
                    let gx = (plane_l(clamp(p + vec2<i32>(1, 0), vec2<i32>(0), hi)) - plane_l(clamp(p - vec2<i32>(1, 0), vec2<i32>(0), hi))) * 0.5;
                    let gy = (plane_l(clamp(p + vec2<i32>(0, 1), vec2<i32>(0), hi)) - plane_l(clamp(p - vec2<i32>(0, 1), vec2<i32>(0), hi))) * 0.5;
                    grad_box += sqrt(gx * gx + gy * gy);
                }
            }
            atomicAdd(&local_hist[min(u32(grad_box / 9.0 * GRAIN_BIN_SCALE), GRAIN_BINS - 1u)], 1u);
        }
    }
    workgroupBarrier();

    for (var i = li; i < GRAIN_BINS; i += 64u) {
        let c = atomicLoad(&local_hist[i]);
        if (c > 0u) { atomicAdd(&hist[i], c); }
    }
}
