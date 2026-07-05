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
    // Linear Adobe RGB -> CIELAB (D65). Input is scene-linear (no sRGB decode).
    let r = max(rgb.r, 0.0);
    let g = max(rgb.g, 0.0);
    let b = max(rgb.b, 0.0);

    // ProPhoto RGB (ROMM) -> XYZ, D50 (working-space primaries; matches CPU rgb_to_lab_working).
    var x = r * 0.7976749 + g * 0.1351917 + b * 0.0313534;
    var y = r * 0.2880402 + g * 0.7118741 + b * 0.0000857;
    var z = r * 0.0000000 + g * 0.0000000 + b * 0.8252100;

    x = x / 0.96422;
    y = y / 1.00000;
    z = z / 0.82521;

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

    x = x * 0.96422;
    y = y * 1.00000;
    z = z * 0.82521;

    // XYZ -> ProPhoto RGB (ROMM), D50 (matches CPU lab_to_rgb_working). Returns
    // scene-linear (no encode).
    let r = x * 1.3459433 + y * -0.2556075 + z * -0.0511118;
    let g = x * -0.5445989 + y * 1.5081673 + z * 0.0205351;
    let b = x * 0.0000000 + y * 0.0000000 + z * 1.2118128;

    return max(vec3<f32>(r, g, b), vec3<f32>(0.0));
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

    // 2. Chemical Toning (Selenium/Sepia) — B&W only, density-driven on the linear
    // print; mirrors _apply_chemical_toning_jit / TONING_CONSTANTS: a density-
    // dependent fraction c of the silver converts, D' = D*(1-c) + c*D*gain.
    if (params.is_bw == 1u && (params.selenium_strength > 0.0 || params.sepia_strength > 0.0)) {
        let sel_gain = vec3<f32>(1.04, 1.10, 1.02);
        let sep_gain = vec3<f32>(0.82, 0.94, 1.12);
        var d = -log(clamp(color, vec3<f32>(1e-6), vec3<f32>(1.0))) / log(10.0);
        // Conversion caps at 1: all the silver is toned (slider > 1 = longer bath).
        if (params.selenium_strength > 0.0) {
            let c_sel = min(params.selenium_strength * pow(min(d / 2.0, vec3<f32>(1.0)), vec3<f32>(1.5)), vec3<f32>(1.0));
            d = d * (1.0 - c_sel) + c_sel * d * sel_gain;
        }
        if (params.sepia_strength > 0.0) {
            let c_sep = min(params.sepia_strength * pow(1.0 - min(d / 1.8, vec3<f32>(1.0)), vec3<f32>(2.0)), vec3<f32>(1.0));
            d = d * (1.0 - c_sep) + c_sep * d * sep_gain;
        }
        color = clamp(pow(vec3<f32>(10.0), -d), vec3<f32>(0.0), vec3<f32>(1.0));
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
