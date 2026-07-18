@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> histogram: array<atomic<u32>, 1024>; // 256 * 4 (R, G, B, Luma)

// Rec. 709 coefficients
const LUMA_COEFFS = vec3<f32>(0.2126, 0.7152, 0.0722);

// Working-space OETF (Adobe RGB: pure 563/256 gamma) — mirrors output_encode.wgsl.
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
    // Input is the scene-linear content texture (pre-output_encode) — encode so
    // the bins match the display-encoded base_positive the CPU histogram sees.
    let color = oetf_encode(textureLoad(input_tex, coords, 0).rgb);
    let luma = dot(color, LUMA_COEFFS);
    
    let bin_r = u32(clamp(color.r * 255.0, 0.0, 255.0));
    let bin_g = u32(clamp(color.g * 255.0, 0.0, 255.0));
    let bin_b = u32(clamp(color.b * 255.0, 0.0, 255.0));
    let bin_l = u32(clamp(luma * 255.0, 0.0, 255.0));

    // Atomic adds into global storage
    atomicAdd(&histogram[bin_r], 1u);
    atomicAdd(&histogram[256u + bin_g], 1u);
    atomicAdd(&histogram[512u + bin_b], 1u);
    atomicAdd(&histogram[768u + bin_l], 1u);
}