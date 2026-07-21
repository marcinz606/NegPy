// Vertical half of the separable sharpen blur — blurs the .x channel of the
// lab_sharpen_h.wgsl output with the same shared taps (sharpen_k buffer) and
// passes the original L* (.y) through for lab.wgsl's clamp/mask neighbourhood.
struct LabUniforms {
    sharpen: f32,
    chroma_denoise: f32,
    saturation: f32,
    vibrance: f32,
    glow_amount: f32,
    halation_strength: f32,
    scale_factor: f32,
    sharpen_radius_px: f32,
    sharpen_masking: f32,
    sharpen_method: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: LabUniforms;
@group(0) @binding(3) var<storage, read> kernel_w: array<f32>;

// cv2's default border mode (BORDER_REFLECT_101) — mirrors the CPU blur.
fn reflect_101(c: i32, n: i32) -> i32 {
    var v = c;
    if (v < 0) { v = -v; }
    if (v >= n) { v = 2 * (n - 1) - v; }
    return clamp(v, 0, n - 1);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));

    let r = i32(params.sharpen_radius_px);
    var acc = 0.0;
    for (var j = -r; j <= r; j++) {
        let sy = reflect_101(coords.y + j, i32(dims.y));
        acc += textureLoad(input_tex, vec2<i32>(coords.x, sy), 0).x * kernel_w[u32(j + r)];
    }

    let l_orig = textureLoad(input_tex, coords, 0).y;
    textureStore(output_tex, coords, vec4<f32>(acc, l_orig, 0.0, 0.0));
}
