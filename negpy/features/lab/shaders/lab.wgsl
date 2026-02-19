struct LabUniforms {
    crosstalk_row0: vec4<f32>,
    crosstalk_row1: vec4<f32>,
    crosstalk_row2: vec4<f32>,
    strength: f32,
    sharpen: f32,
    chroma_denoise: f32,
    saturation: f32,
    vibrance: f32,
    pad: vec3<f32>,
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

fn rgb_to_lab(rgb: vec3<f32>) -> vec3<f32> {
    var r = rgb.r;
    var g = rgb.g;
    var b = rgb.b;

    if (r > 0.04045) { r = pow((r + 0.055) / 1.055, 2.4); } else { r = r / 12.92; }
    if (g > 0.04045) { g = pow((g + 0.055) / 1.055, 2.4); } else { g = g / 12.92; }
    if (b > 0.04045) { b = pow((b + 0.055) / 1.055, 2.4); } else { b = b / 12.92; }

    var x = r * 0.4124 + g * 0.3576 + b * 0.1805;
    var y = r * 0.2126 + g * 0.7152 + b * 0.0722;
    var z = r * 0.0193 + g * 0.1192 + b * 0.9505;

    x = x / 0.95047;
    y = y / 1.00000;
    z = z / 1.08883;

    if (x > 0.008856) { x = pow(x, 1.0/3.0); } else { x = (7.787 * x) + (16.0 / 116.0); }
    if (y > 0.008856) { y = pow(y, 1.0/3.0); } else { y = (7.787 * y) + (16.0 / 116.0); }
    if (z > 0.008856) { z = pow(z, 1.0/3.0); } else { z = (7.787 * z) + (16.0 / 116.0); }

    let l = (116.0 * y) - 16.0;
    let a = 500.0 * (x - y);
    let b_lab = 200.0 * (y - z);

    return vec3<f32>(l, a, b_lab);
}

fn lab_to_rgb(lab: vec3<f32>) -> vec3<f32> {
    var y = (lab.x + 16.0) / 116.0;
    var x = lab.y / 500.0 + y;
    var z = y - lab.z / 200.0;

    if (pow(x, 3.0) > 0.008856) { x = pow(x, 3.0); } else { x = (x - 16.0 / 116.0) / 7.787; }
    if (pow(y, 3.0) > 0.008856) { y = pow(y, 3.0); } else { y = (y - 16.0 / 116.0) / 7.787; }
    if (pow(z, 3.0) > 0.008856) { z = pow(z, 3.0); } else { z = (z - 16.0 / 116.0) / 7.787; }

    x = x * 0.95047;
    y = y * 1.00000;
    z = z * 1.08883;

    var r = x * 3.2406 + y * -1.5372 + z * -0.4986;
    var g = x * -0.9689 + y * 1.8758 + z * 0.0415;
    var b = x * 0.0557 + y * -0.2040 + z * 1.0570;

    if (r > 0.0031308) { r = 1.055 * pow(r, 1.0/2.4) - 0.055; } else { r = 12.92 * r; }
    if (g > 0.0031308) { g = 1.055 * pow(g, 1.0/2.4) - 0.055; } else { g = 12.92 * g; }
    if (b > 0.0031308) { b = 1.055 * pow(b, 1.0/2.4) - 0.055; } else { b = 12.92 * b; }

    return vec3<f32>(r, g, b);
}

fn rgb_to_hsv(c: vec3<f32>) -> vec3<f32> {
    let v = max(c.r, max(c.g, c.b));
    let m = min(c.r, min(c.g, c.b));
    let d = v - m;
    var h: f32;
    var s: f32;
    if (d == 0.0) { h = 0.0; }
    else if (v == c.r) { h = (c.g - c.b) / d; }
    else if (v == c.g) { h = (c.b - c.r) / d + 2.0; }
    else { h = (c.r - c.g) / d + 4.0; }
    h = fract(h / 6.0);
    if (v == 0.0) { s = 0.0; } else { s = d / v; }
    return vec3<f32>(h, s, v);
}

fn hsv_to_rgb(c: vec3<f32>) -> vec3<f32> {
    let h = c.x * 6.0;
    let s = c.y;
    let v = c.z;
    let i = floor(h);
    let f = h - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    let cond = i32(i) % 6;
    if (cond == 0) { return vec3<f32>(v, t, p); }
    else if (cond == 1) { return vec3<f32>(q, v, p); }
    else if (cond == 2) { return vec3<f32>(p, v, t); }
    else if (cond == 3) { return vec3<f32>(p, q, v); }
    else if (cond == 4) { return vec3<f32>(t, p, v); }
    else { return vec3<f32>(v, p, q); }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    var color = textureLoad(input_tex, coords, 0).rgb;

    // 1. Chroma Denoise
    if (params.chroma_denoise > 0.0) {
        let current_lab = rgb_to_lab(color);
        var blur_ab = vec2<f32>(0.0);
        for (var j = -2; j <= 2; j++) {
            for (var i = -2; i <= 2; i++) {
                let sample_coords = clamp(coords + vec2<i32>(i, j), vec2<i32>(0), vec2<i32>(dims) - 1);
                let sample_rgb = textureLoad(input_tex, sample_coords, 0).rgb;
                let sample_lab = rgb_to_lab(sample_rgb);
                let weight = gauss_kernel[(j + 2) * 5 + (i + 2)];
                blur_ab += sample_lab.yz * weight;
            }
        }
        color = lab_to_rgb(vec3<f32>(current_lab.x, blur_ab.x, blur_ab.y));
    }

    // 2. Spectral Crosstalk
    if (params.strength > 0.0) {
        let epsilon = 1e-6;
        let dens = -log(max(color, vec3<f32>(epsilon))) / 2.302585;
        let m0 = params.crosstalk_row0.xyz;
        let m1 = params.crosstalk_row1.xyz;
        let m2 = params.crosstalk_row2.xyz;
        let mixed_dens = vec3<f32>(dot(dens, m0), dot(dens, m1), dot(dens, m2));
        color = pow(vec3<f32>(10.0), -mixed_dens);
    }

    // 3. Vibrance
    if (params.vibrance != 1.0) {
        var lab = rgb_to_lab(color);
        let chroma = length(lab.yz);
        let muted_mask = clamp(1.0 - (chroma / 60.0), 0.0, 1.0);
        let boost = (params.vibrance - 1.0) * muted_mask;
        lab.y = lab.y * (1.0 + boost);
        lab.z = lab.z * (1.0 + boost);
        color = lab_to_rgb(lab);
    }

    // 4. Global Saturation
    if (params.saturation != 1.0) {
        var hsv = rgb_to_hsv(color);
        hsv.y = clamp(hsv.y * params.saturation, 0.0, 1.0);
        color = hsv_to_rgb(hsv);
    }

    // 5. Sharpening
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