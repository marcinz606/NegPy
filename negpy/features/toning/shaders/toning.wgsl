struct ToningUniforms {
    saturation: f32,
    selenium_strength: f32,
    sepia_strength: f32,
    gamma: f32,
    crop_offset: vec2<i32>,    // x, y offset in input texture
    is_bw: u32,                // 1 if B&W mode
    gold_strength: f32,
    shadow_tint_hue: f32,
    shadow_tint_strength: f32,
    highlight_tint_hue: f32,
    highlight_tint_strength: f32,
    blue_strength: f32,
    copper_strength: f32,
    vanadium_strength: f32,
    alt_mode: u32,             // 0 none, 1 lith, 2 cyanotype
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: ToningUniforms;

fn rgb_to_lab(rgb: vec3<f32>) -> vec3<f32> {
    // Linear working RGB -> CIELAB (D65). Input is scene-linear (no TRC decode).
    let r = max(rgb.r, 0.0);
    let g = max(rgb.g, 0.0);
    let b = max(rgb.b, 0.0);

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

    // XYZ -> Adobe RGB (1998), D65 (matches CPU lab_to_rgb_working). Returns
    // scene-linear (no encode).
    let r = x * 2.0413690 + y * -0.5649464 + z * -0.3446944;
    let g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560;
    let b = x * 0.0134474 + y * -0.1183897 + z * 1.0154096;

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

    // 1. Process Mode (B&W). Redundant with the exposure pass, which already
    // collapsed to luma — except under an alternative process, where the lith or
    // cyanotype pass has since put real color on the print and collapsing would
    // throw it away.
    if (params.is_bw == 1u && params.alt_mode == 0u) {
        let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
        color = vec3<f32>(luma);
    }

    // 2. Chemical Toning (Selenium/Sepia/Gold/Blue/Copper/Vanadium) — B&W only
    // and never on a cyanotype, which holds no silver for the baths to react with:
    // silver-ledger model mirroring _apply_chemical_toning_jit / TONING_CONSTANTS:
    // all baths compete for one metallic-silver reservoir. Susceptibility c_i is
    // a pure function of the ORIGINAL density d0; bath order only decides who
    // claims silver first (f_i = a*c_i, a -= f_i). Gold is the lock-out
    // exception: it also plates the sulfide fraction with compounded covering
    // power (classic gold-over-sepia orange-red).
    if (params.is_bw == 1u && params.alt_mode != 2u && (params.selenium_strength > 0.0 || params.sepia_strength > 0.0 || params.gold_strength > 0.0 || params.blue_strength > 0.0 || params.copper_strength > 0.0 || params.vanadium_strength > 0.0)) {
        // Lith silver is fine, small-particle silver on the steep part of the
        // tone-vs-grain-size curve, so selenium and gold move it much further —
        // mirrors LITH_TONING_CONSTANTS. The other four baths are unchanged
        // (and the sidebar disables them under lith).
        let lith = params.alt_mode == 1u;
        let sel_gain = select(vec3<f32>(1.04, 1.10, 1.02), vec3<f32>(1.10, 1.24, 1.12), lith);
        let sep_gain = vec3<f32>(0.82, 0.94, 1.12);
        let gold_gain = select(vec3<f32>(1.08, 1.03, 1.00), vec3<f32>(1.16, 1.06, 0.96), lith);
        let gold_sepia_gain = vec3<f32>(0.80, 0.95, 1.20);
        let blue_gain = vec3<f32>(1.30, 1.00, 0.80);
        let copper_gain = vec3<f32>(0.72, 0.94, 1.18);
        let van_gain = vec3<f32>(1.12, 0.85, 1.03);
        let d3 = -log(clamp(color, vec3<f32>(1e-6), vec3<f32>(1.0))) / log(10.0);
        let d0 = (d3.x + d3.y + d3.z) / 3.0;
        // Conversion caps at 1: all the remaining silver is toned (slider > 1 = longer bath).
        let sel_d_ref = select(2.0, 1.2, lith);
        let sel_power = select(1.5, 1.0, lith);
        let c_sel = min(params.selenium_strength * pow(min(d0 / sel_d_ref, 1.0), sel_power), 1.0);
        let c_sep = min(params.sepia_strength * pow(1.0 - min(d0 / 1.8, 1.0), 2.0), 1.0);
        // On lith, gold "attacks all densities evenly" — the bleach-limited
        // shape is replaced by a flat conversion.
        let au_shape = select(pow(1.0 - min(d0 / 1.6, 1.0), 1.5), 1.0, lith);
        let c_au = min(params.gold_strength * au_shape, 1.0);
        let c_blue = min(params.blue_strength * pow(min(d0 / 0.9, 1.0), 0.85), 1.0);
        let c_cu = min(params.copper_strength * pow(min(d0 / 0.9, 1.0), 0.6), 1.0);
        let c_van = min(params.vanadium_strength * pow(1.0 - min(d0 / 1.8, 1.0), 1.2), 1.0);

        var a = 1.0;
        let f_sel = a * c_sel;
        a -= f_sel;
        var f_sep = a * c_sep;
        a -= f_sep;
        let f_au = a * c_au;
        a -= f_au;
        let f_ausp = f_sep * c_au;
        f_sep -= f_ausp;
        let f_blue = a * c_blue;
        a -= f_blue;
        let f_cu = a * c_cu;
        a -= f_cu;
        let f_van = a * c_van;
        a -= f_van;

        // d3, not d0: see _apply_chemical_toning_jit. Rebuilding from the mean
        // would discard color the print already carries.
        let d = d3 * (vec3<f32>(a)
            + f_sel * sel_gain
            + f_sep * sep_gain
            + f_au * gold_gain
            + f_ausp * sep_gain * gold_sepia_gain
            + f_blue * blue_gain
            + f_cu * copper_gain
            + f_van * van_gain);
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
