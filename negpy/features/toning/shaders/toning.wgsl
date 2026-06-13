struct ToningUniforms {
    saturation: f32,
    selenium_strength: f32,
    sepia_strength: f32,
    gamma: f32,
    crop_offset: vec2<i32>,    // x, y offset in input texture
    is_bw: u32,                // 1 if B&W mode
    pad2: f32,
    shadow_tint_hue: f32,
    shadow_tint_strength: f32,
    highlight_tint_hue: f32,
    highlight_tint_strength: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: ToningUniforms;

fn rgb_to_lab(rgb: vec3<f32>) -> vec3<f32> {
    var r = rgb.r;
    var g = rgb.g;
    var b = rgb.b;

    if (r > 0.04045) { r = pow((r + 0.055) / 1.055, 2.4); } else { r = r / 12.92; }
    if (g > 0.04045) { g = pow((g + 0.055) / 1.055, 2.4); } else { g = g / 12.92; }
    if (b > 0.04045) { b = pow((b + 0.055) / 1.055, 2.4); } else { b = b / 12.92; }

    // Adobe RGB (1998) -> XYZ, D65 (working-space primaries; matches CPU rgb_to_lab_working).
    var x = r * 0.5767309 + g * 0.1855540 + b * 0.1881852;
    var y = r * 0.2973769 + g * 0.6273491 + b * 0.0752741;
    var z = r * 0.0270343 + g * 0.0706872 + b * 0.9911085;

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

    // XYZ -> Adobe RGB (1998), D65 (matches CPU lab_to_rgb_working).
    var r = x * 2.0413690 + y * -0.5649464 + z * -0.3446944;
    var g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560;
    var b = x * 0.0134474 + y * -0.1183897 + z * 1.0154096;

    if (r > 0.0031308) { r = 1.055 * pow(r, 1.0/2.4) - 0.055; } else { r = 12.92 * r; }
    if (g > 0.0031308) { g = 1.055 * pow(g, 1.0/2.4) - 0.055; } else { g = 12.92 * g; }
    if (b > 0.0031308) { b = 1.055 * pow(b, 1.0/2.4) - 0.055; } else { b = 12.92 * b; }

    return vec3<f32>(r, g, b);
}

fn hue_to_ab(hue_deg: f32, chroma: f32) -> vec2<f32> {
    let rad = hue_deg * 0.017453293;  // pi / 180
    return vec2<f32>(cos(rad), sin(rad)) * chroma;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let coords_out = vec2<i32>(i32(gid.x), i32(gid.y));
    let coords_in = coords_out + params.crop_offset;

    var color = textureLoad(input_tex, coords_in, 0).rgb;

    // 1. Process Mode (B&W)
    if (params.is_bw == 1u) {
        let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
        color = vec3<f32>(luma);
    }

    // 2. Chemical Toning (Selenium/Sepia) — B&W only
    let luma_toning = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));

    if (params.selenium_strength > 0.0) {
        let sel_m = clamp((1.0 - luma_toning) * (1.0 - luma_toning) * params.selenium_strength, 0.0, 1.0);
        color = mix(color, color * vec3<f32>(0.85, 0.75, 0.85), sel_m);
    }

    if (params.sepia_strength > 0.0) {
        let sep_m = exp(-pow(luma_toning - 0.6, 2.0) / 0.08) * params.sepia_strength;
        color = mix(color, color * vec3<f32>(1.1, 0.99, 0.825), sep_m);
    }

    // 3. Split Toning — all modes (color and B&W)
    if (params.shadow_tint_strength > 0.0 || params.highlight_tint_strength > 0.0) {
        var lab = rgb_to_lab(color);

        if (params.shadow_tint_strength > 0.0) {
            let s_mask = smoothstep(50.0, 0.0, lab.x);
            let ab = hue_to_ab(params.shadow_tint_hue, 20.0 * params.shadow_tint_strength * s_mask);
            lab.y += ab.x;
            lab.z += ab.y;
        }

        if (params.highlight_tint_strength > 0.0) {
            let h_mask = smoothstep(50.0, 100.0, lab.x);
            let ab = hue_to_ab(params.highlight_tint_hue, 20.0 * params.highlight_tint_strength * h_mask);
            lab.y += ab.x;
            lab.z += ab.y;
        }

        color = lab_to_rgb(lab);
    }

    textureStore(output_tex, coords_out, vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
