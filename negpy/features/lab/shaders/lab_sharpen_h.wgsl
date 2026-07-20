// Horizontal half of the separable sharpen blur. Taps come from
// gaussian_kernel_1d in features/lab/logic.py, uploaded verbatim (sharpen_k
// buffer) — kernel support and weights are bit-identical to the CPU
// cv2.sepFilter2D call. Writes (h-blurred L*, original L*, 0, 0); the vertical
// pass (lab_sharpen_v.wgsl) finishes the blur and lab.wgsl consumes both.
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

// Working-space TRC (Adobe RGB 1998: pure 563/256 gamma) — mirrors oetf_decode
// in lab.wgsl.
fn oetf_decode(c: vec3<f32>) -> vec3<f32> {
    let e = max(c, vec3<f32>(0.0));
    return pow(e, vec3<f32>(2.19921875));
}

// CIELAB L* from linear Adobe RGB: L* depends only on Y, so just the Y row of
// the RGB->XYZ matrix (D65, Yn=1; matches rgb_to_lab in lab.wgsl / CPU kernel).
fn lab_l(rgb: vec3<f32>) -> f32 {
    var y = max(rgb.r, 0.0) * 0.2973769 + max(rgb.g, 0.0) * 0.6273491 + max(rgb.b, 0.0) * 0.0752741;
    if (y > 0.008856) { y = pow(y, 1.0 / 3.0); } else { y = (7.787 * y) + (16.0 / 116.0); }
    return (116.0 * y) - 16.0;
}

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
    for (var i = -r; i <= r; i++) {
        let sx = reflect_101(coords.x + i, i32(dims.x));
        let l = lab_l(oetf_decode(textureLoad(input_tex, vec2<i32>(sx, coords.y), 0).rgb));
        acc += l * kernel_w[u32(i + r)];
    }

    let l_center = lab_l(oetf_decode(textureLoad(input_tex, coords, 0).rgb));
    textureStore(output_tex, coords, vec4<f32>(acc, l_center, 0.0, 0.0));
}
