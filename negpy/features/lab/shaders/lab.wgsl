struct LabUniforms {
    crosstalk_row0: vec4<f32>,
    crosstalk_row1: vec4<f32>,
    crosstalk_row2: vec4<f32>,
    strength: f32,      // Now used as a flag/multiplier (1.0 or 0.0)
    sharpen: f32,
    pad: vec2<f32>,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: LabUniforms;

const gauss_kernel = array<f32, 25>(
    0.003765, 0.015019, 0.023792, 0.015019, 0.003765,
    0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
    0.023792, 0.094907, 0.150342, 0.094907, 0.023792,
    0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
    0.003765, 0.015019, 0.023792, 0.015019, 0.003765
);

const LUMA_COEFFS = vec3<f32>(0.2126, 0.7152, 0.0722);

fn to_perceptual(c: vec3<f32>) -> vec3<f32> {
    return pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
}

fn to_linear(c: vec3<f32>) -> vec3<f32> {
    return pow(max(c, vec3<f32>(0.0)), vec3<f32>(2.2));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    var color = textureLoad(input_tex, coords, 0).rgb;

    // 1. Spectral Crosstalk (Matrix multiplication in Density Space)
    if (params.strength > 0.0) {
        // Convert to Density (-log10)
        // WGSL log is ln, so we divide by ln(10) ~ 2.302585
        let epsilon = 1e-6;
        let dens = -log(max(color, vec3<f32>(epsilon))) / 2.302585;

        let m0 = params.crosstalk_row0.xyz;
        let m1 = params.crosstalk_row1.xyz;
        let m2 = params.crosstalk_row2.xyz;
        
        let mixed_r = dot(dens, m0);
        let mixed_g = dot(dens, m1);
        let mixed_b = dot(dens, m2);
        
        // Convert back from Density (10^-d)
        let mixed_dens = vec3<f32>(mixed_r, mixed_g, mixed_b);
        color = pow(vec3<f32>(10.0), -mixed_dens);
    }

    // 2. Sharpening (Perceptual Luma USM)
    if (params.sharpen > 0.0) {
        var blur_luma = 0.0;
        for (var j = -2; j <= 2; j++) {
            for (var i = -2; i <= 2; i++) {
                let sample_coords = clamp(coords + vec2<i32>(i, j), vec2<i32>(0), vec2<i32>(dims) - 1);
                let sample_color = textureLoad(input_tex, sample_coords, 0).rgb;
                let weight = gauss_kernel[(j + 2) * 5 + (i + 2)];
                blur_luma += dot(to_perceptual(sample_color), LUMA_COEFFS) * weight;
            }
        }
        
        let p_color = to_perceptual(color);
        let luma = dot(p_color, LUMA_COEFFS);
        let amount = params.sharpen * 2.5;
        let sharpened_luma = luma + (luma - blur_luma) * amount;
        let ratio = sharpened_luma / max(luma, 1e-6);
        color = to_linear(p_color * ratio);
    }

    textureStore(output_tex, coords, vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}