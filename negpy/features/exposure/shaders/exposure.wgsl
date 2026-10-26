struct ExposureUniforms {
    pivots: vec4<f32>,
    slopes: vec4<f32>,
    curvatures: vec4<f32>,
    cmy_offsets: vec4<f32>,
    shadow_cmy: vec4<f32>,
    highlight_cmy: vec4<f32>,
    // Per-channel knee widths (per_channel_widths, pre-clamped CPU-side): six
    // lanes in the ex-scalar toe/shoulder/midtone_gamma slots (those values ride
    // the vec4 w-lanes) and the former flare pad, keeping the 256B layout.
    toe_width_r: f32,
    toe_width_g: f32,
    toe_width_b: f32,
    shoulder_width_r: f32,
    // Zone Density ΔD shadow offset in the ex-d_min slot (the curve reads d_min_rgb).
    shadow_density: f32,
    d_max: f32,
    a_toe_base: f32,
    a_sh_base: f32,
    // Separation Damping (0 = off; toeshoulder_width_ref is the 2.5 literal below).
    sep_damping: f32,
    toe_height: f32,
    sh_height: f32,
    zone_center: f32,
    shoulder_width_g: f32,
    // Separation Damping's per-layer k, red (green/blue ride split_sh.w/split_hi.w);
    // was the ex-surround_gamma pad.
    sep_k_r: f32,
    mode: u32,
    v_star: f32,
    shoulder_width_b: f32,
    gamma_width: f32,
    use_dye: u32,
    // Black point compensation flag (0/1); was the 16B pad before d_min_rgb.
    bpc: f32,
    // Per-channel paper-white floor (base+fog incl. tint) in xyz; w carries the
    // Zone Density ΔD highlight offset (the block is full at 256B).
    d_min_rgb: vec4<f32>,
    // Row-normalized dye coupling rows (D_rgb = M * D_dye above base).
    dye_r: vec4<f32>,
    dye_g: vec4<f32>,
    dye_b: vec4<f32>,
    // Dodge/burn: xyz = per-channel normalized-space size of one EV stop
    // (local_ev_scale), w = enable flag (0 -> ev_tex is a dummy, skip it).
    ev_scale: vec4<f32>,
    // Split Grade per-channel zone contrast gains (split_grade_deltas); the two
    // w-lanes carry Separation Damping's green/blue k (red rides sep_k_r).
    // These rows push the block past 256B: exposure spans two UBO slots.
    split_sh: vec4<f32>,
    split_hi: vec4<f32>,
    // Hue Trim: x = rotation in radians; yzw = preflash fraction (0 = off), threshold
    // value and paper gamma (preflash_params). Costs no slot; 288B already spanned two.
    hue: vec4<f32>,
    // x = Contrast Mask stop cap (0 = off), yz = the printed frame's origin in rotated
    // pixels, w = the Diffusion mix (0 = off).
    mask: vec4<f32>,
    // The printed frame's span in rotated pixels in xy, shared by both planes; zw = the
    // Contrast Mask's code clamp (lo, hi).
    mask_span: vec4<f32>,
    // Contrast Mask (contrast_mask_coeffs): code = x * plane + y, z/w = the gain below/above zero.
    mask_tone: vec4<f32>,
    // Tone-limited masks (limited_mask_params): stops, ISO-R delta, key edges e0/e1 each.
    keyed: array<vec4<f32>, 4>,
    // x = limited mask count, y = frame ISO-R, zw = the ISO-R ladder's ends.
    key_meta: vec4<f32>,
    // Diffusion: xyz = each channel's log range (channel_density_ranges), w = the Hold Black
    // taper span (black_hold_taper, 0 = off).
    diff_range: vec4<f32>,
    // Print-stock LUT: xyz = e_max per dye (0 = no LUT), w = the black luminance BPC maps to 0.
    lut: vec4<f32>,
    // B&W collapse weights on the normalized log channels (collapse_weights); w is a pad.
    bw: vec4<f32>,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: ExposureUniforms;
// Per-pixel dodge/burn EV map, rasterised on the CPU (shared with the CPU path).
@group(0) @binding(3) var ev_tex: texture_2d<f32>;
// Analysis-grid planes: r = Contrast Mask, gba = Diffusion. Upscaled here, not uploaded at
// render size, so a slider costs a uniform write and no transfer.
@group(0) @binding(4) var mask_tex: texture_2d<f32>;
// Shape alpha of each tone-limited mask, one per channel (planes 2+ of compute_local_maps).
@group(0) @binding(5) var key_tex: texture_2d<f32>;
// The stock's LUT unrolled to 2D: node (i, j, k) at texel (i + N*j, k). A 1x1 dummy without one.
@group(0) @binding(6) var lut_tex: texture_2d<f32>;

fn lut_node(i: i32, j: i32, k: i32, n: i32) -> vec3<f32> {
    return textureLoad(lut_tex, vec2<i32>(i + n * j, k), 0).xyz;
}

// Tetrahedral interpolation; mirrors tetra_lookup in print_lut.py.
fn lut_tetra(e: vec3<f32>) -> vec3<f32> {
    let n = i32(textureDimensions(lut_tex).y);
    let p = clamp(e / params.lut.xyz, vec3<f32>(0.0), vec3<f32>(1.0)) * f32(n - 1);
    let i = min(vec3<i32>(p), vec3<i32>(n - 2));
    let f = p - vec3<f32>(i);
    let c000 = lut_node(i.x, i.y, i.z, n);
    let c111 = lut_node(i.x + 1, i.y + 1, i.z + 1, n);
    if (f.x >= f.y) {
        if (f.y >= f.z) {
            let c100 = lut_node(i.x + 1, i.y, i.z, n);
            let c110 = lut_node(i.x + 1, i.y + 1, i.z, n);
            return c000 + f.x * (c100 - c000) + f.y * (c110 - c100) + f.z * (c111 - c110);
        } else if (f.x >= f.z) {
            let c100 = lut_node(i.x + 1, i.y, i.z, n);
            let c101 = lut_node(i.x + 1, i.y, i.z + 1, n);
            return c000 + f.x * (c100 - c000) + f.z * (c101 - c100) + f.y * (c111 - c101);
        }
        let c001 = lut_node(i.x, i.y, i.z + 1, n);
        let c101 = lut_node(i.x + 1, i.y, i.z + 1, n);
        return c000 + f.z * (c001 - c000) + f.x * (c101 - c001) + f.y * (c111 - c101);
    }
    if (f.z >= f.y) {
        let c001 = lut_node(i.x, i.y, i.z + 1, n);
        let c011 = lut_node(i.x, i.y + 1, i.z + 1, n);
        return c000 + f.z * (c001 - c000) + f.y * (c011 - c001) + f.x * (c111 - c011);
    } else if (f.z >= f.x) {
        let c010 = lut_node(i.x, i.y + 1, i.z, n);
        let c011 = lut_node(i.x, i.y + 1, i.z + 1, n);
        return c000 + f.y * (c010 - c000) + f.z * (c011 - c010) + f.x * (c111 - c011);
    }
    let c010 = lut_node(i.x, i.y + 1, i.z, n);
    let c110 = lut_node(i.x + 1, i.y + 1, i.z, n);
    return c000 + f.y * (c010 - c000) + f.x * (c110 - c010) + f.z * (c111 - c110);
}

// The planes at this pixel. Mirrors plane_taps in exposure/logic.py.
fn plane_sample(coords: vec2<i32>) -> vec4<f32> {
    let dims = vec2<f32>(textureDimensions(mask_tex));
    let p = (vec2<f32>(coords) + vec2<f32>(0.5) - params.mask.yz) * dims / params.mask_span.xy - vec2<f32>(0.5);
    let lo = clamp(floor(p), vec2<f32>(0.0), dims - vec2<f32>(1.0));
    let hi = clamp(lo + vec2<f32>(1.0), vec2<f32>(0.0), dims - vec2<f32>(1.0));
    let f = clamp(p - lo, vec2<f32>(0.0), vec2<f32>(1.0));
    let i0 = vec2<i32>(lo);
    let i1 = vec2<i32>(hi);
    let top = mix(textureLoad(mask_tex, vec2<i32>(i0.x, i0.y), 0), textureLoad(mask_tex, vec2<i32>(i1.x, i0.y), 0), f.x);
    let bot = mix(textureLoad(mask_tex, vec2<i32>(i0.x, i1.y), 0), textureLoad(mask_tex, vec2<i32>(i1.x, i1.y), 0), f.x);
    return mix(top, bot, f.y);
}

// The Contrast Mask at this pixel, in stops. Mirrors contrast_mask_stops in negpy_core.
fn contrast_mask_stops(coords: vec2<i32>) -> f32 {
    let k = params.mask_tone;
    let d = clamp(k.x * plane_sample(coords).r + k.y, params.mask_span.z, params.mask_span.w);
    let t = select(-k.w * d * d, k.z * d * d, d < 0.0);
    return clamp(t, -params.mask.x, params.mask.x);
}

fn fast_sigmoid(x: f32) -> f32 {
    if (x >= 0.0) {
        return 1.0 / (1.0 + exp(-x));
    } else {
        let z = exp(x);
        return z / (1.0 + z);
    }
}

// Numerically stable softplus: log(1 + exp(x)). Antiderivative of the sigmoid.
fn softplus(x: f32) -> f32 {
    return max(x, 0.0) + log(1.0 + exp(-abs(x)));
}

// One pixel's effective dye-separation k; mirrors separation_damping_gain in
// exposure/logic.py. 0.35 mirrors separation_damping_ref_spread in models.py and
// the copy in transfer.wgsl -- change all three.
fn separation_damping_gain(k: f32, damping: f32, chroma: f32) -> f32 {
    if (k <= 0.0) {
        return 0.0;
    }
    let h = (0.35 - chroma) / (0.35 + chroma);
    return min(pow(k, (1.0 - damping) + damping * h), 3.0);
}

// A tone-limited mask's weight; mirrors tone_key_weight in exposure/logic.py. Written out
// rather than smoothstep(), which is undefined for e0 > e1 (a Highlights limit).
fn tone_key_weight(lum: f32, e0: f32, e1: f32) -> f32 {
    let t = clamp((lum - e0) / (e1 - e0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Mirrors logic.quadratic_core: held at the vertex where the quadratic would fold back.
fn quadratic_core(k: f32, pivot: f32, curv: f32, u_in: f32) -> f32 {
    var u = u_in;
    if (curv != 0.0) {
        let vertex = -k / (2.0 * curv);
        if ((curv > 0.0 && u < vertex) || (curv < 0.0 && u > vertex)) {
            u = vertex;
        }
    }
    return k * (u - pivot) + curv * u * u;
}

// Working-space OETF (Adobe RGB: pure 563/256 gamma); feeds the encoded
// perceptual region (clahe, retouch) before lab decodes back to linear.
fn oetf_encode(t: f32) -> f32 {
    let x = clamp(t, 0.0, 1.0);
    return pow(x, 0.45470693);
}

// Copy of lab.wgsl's rgb_to_lab/lab_to_rgb (Adobe RGB 1998, D65, scene-linear); WGSL has no includes,
// so a primaries or white-point change updates lab.wgsl, exposure.wgsl and transfer.wgsl.
fn hue_rgb_to_lab(rgb: vec3<f32>) -> vec3<f32> {
    let r = max(rgb.r, 0.0);
    let g = max(rgb.g, 0.0);
    let b = max(rgb.b, 0.0);

    var x = r * 0.5767309 + g * 0.1855540 + b * 0.1881852;
    var y = r * 0.2973769 + g * 0.6273491 + b * 0.0752741;
    var z = r * 0.0270343 + g * 0.0706872 + b * 0.9911085;

    x = x / 0.95047;
    y = y / 1.00000;
    z = z / 1.08883;

    if (x > 0.008856) { x = pow(x, 1.0/3.0); } else { x = (7.787 * x) + (16.0 / 116.0); }
    if (y > 0.008856) { y = pow(y, 1.0/3.0); } else { y = (7.787 * y) + (16.0 / 116.0); }
    if (z > 0.008856) { z = pow(z, 1.0/3.0); } else { z = (7.787 * z) + (16.0 / 116.0); }

    return vec3<f32>((116.0 * y) - 16.0, 500.0 * (x - y), 200.0 * (y - z));
}

fn hue_lab_to_rgb(lab: vec3<f32>) -> vec3<f32> {
    var y = (lab.x + 16.0) / 116.0;
    var x = lab.y / 500.0 + y;
    var z = y - lab.z / 200.0;

    if (pow(x, 3.0) > 0.008856) { x = pow(x, 3.0); } else { x = (x - 16.0 / 116.0) / 7.787; }
    if (pow(y, 3.0) > 0.008856) { y = pow(y, 3.0); } else { y = (y - 16.0 / 116.0) / 7.787; }
    if (pow(z, 3.0) > 0.008856) { z = pow(z, 3.0); } else { z = (z - 16.0 / 116.0) / 7.787; }

    x = x * 0.95047;
    y = y * 1.00000;
    z = z * 1.08883;

    let r = x * 2.0413690 + y * -0.5649464 + z * -0.3446944;
    let g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560;
    let b = x * 0.0134474 + y * -0.1183897 + z * 1.0154096;

    return max(vec3<f32>(r, g, b), vec3<f32>(0.0));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    var color = textureLoad(input_tex, coords, 0);

    // B&W: one weighted density BEFORE the curve, in matmul_3x3's fma order.
    if (params.mode == 1u) {
        let v = fma(color.b, params.bw.z, fma(color.g, params.bw.y, color.r * params.bw.x));
        color = vec4<f32>(v, v, v, color.a);
    }

    let eps = 1e-6;
    // Asymmetric H&D print curve (toe-linear-shoulder); mirrors the CPU print curve
    // (negpy_core print_curve.rs). toe -> shadow (paper-black) bound, shoulder ->
    // highlight (paper-white) bound. a_toe_base/a_sh_base carry shadow/highlight
    // sharpness; width sets gentleness, slider sets roll-off height.
    let toe_w3 = vec3<f32>(params.toe_width_r, params.toe_width_g, params.toe_width_b);
    let sh_w3 = vec3<f32>(params.shoulder_width_r, params.shoulder_width_g, params.shoulder_width_b);
    // 2.5 mirrors toeshoulder_width_ref in models.py — change together.
    let a_hl = params.a_sh_base * 2.5 / max(sh_w3, vec3<f32>(eps));
    let a_sh_base = params.a_toe_base * 2.5 / max(toe_w3, vec3<f32>(eps));
    // Per-channel toe/shoulder, pre-scaled CPU-side (per_channel_toe_shoulder).
    // The uniform block is full at 256B, so the vec4 w-lanes carry them.
    let toe3 = vec3<f32>(params.pivots.w, params.slopes.w, params.curvatures.w);
    let sh3 = vec3<f32>(params.cmy_offsets.w, params.shadow_cmy.w, params.highlight_cmy.w);
    // Per-channel Snap rides the dye-row w-lanes; scalar midtone_gamma is layout-only.
    let mg3 = vec3<f32>(params.dye_r.w, params.dye_g.w, params.dye_b.w);
    let toe_neg = toe3 < vec3<f32>(0.0);
    // Negative toe: tighten shadow roll-off (sharper knee) rather than extending
    // d_max_eff beyond paper black (perceptually near-zero effect above d_max).
    let a_sh = select(a_sh_base, a_sh_base * (1.0 - toe3 * 4.0), toe_neg);
    let d_min_rgb = params.d_min_rgb.xyz;
    let d_min_eff = max(d_min_rgb + sh3 * params.sh_height, vec3<f32>(0.0));
    let d_max_base = select(vec3<f32>(params.d_max) - toe3 * params.toe_height, vec3<f32>(params.d_max), toe_neg);
    let d_max_eff = max(d_max_base, d_min_eff + vec3<f32>(0.1));

    // .r = dodge/burn stops (positive = burn), .g = flash fraction, .b = ISO-R deltas, turned into a
    // slope multiplier here (local_grade_factor_map) so a Grade change re-uploads nothing.
    var ev = 0.0;
    var gfac = 1.0;
    var flash = params.hue.y;
    if (params.ev_scale.w != 0.0) {
        let local_maps = textureLoad(ev_tex, coords, 0);
        ev = local_maps.r;
        flash = flash + local_maps.g;
        var dr = local_maps.b;
        let n_key = i32(params.key_meta.x);
        if (n_key > 0) {
            // The unburned tone: no mask can move the pixels it selects. Mirrors the CPU
            // kernel; the grade sums in ISO-R space, then clamps once to the ladder.
            let lum = dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
            let alphas = textureLoad(key_tex, coords, 0);
            for (var k = 0; k < n_key; k++) {
                let e = params.keyed[k];
                let a = alphas[k] * tone_key_weight(lum, e.z, e.w);
                ev = ev + e.x * a;
                dr = dr + e.y * a;
            }
        }
        let r0 = params.key_meta.y;
        gfac = r0 / clamp(r0 + dr, params.key_meta.z, params.key_meta.w);
    }
    if (params.mask.x != 0.0) {
        ev = ev + contrast_mask_stops(coords);
    }

    // Diffusion mixes light, not density: the paper sees (1-d)·I + d·blur(I), with
    // I = 10^(val·r), r the channel's log range (diffusion_plane). Mirrors the CPU kernel.
    var diffused = vec3<f32>(0.0);
    if (params.mask.w > 0.0) {
        diffused = plane_sample(coords).gba;
    }

    var dens: vec3<f32>;

    for (var ch = 0; ch < 3; ch++) {
        var base = color[ch];
        if (params.mask.w > 0.0) {
            let r = params.diff_range[ch];
            base = log(mix(pow(10.0, base * r), pow(10.0, diffused[ch] * r), params.mask.w)) / (2.302585 * r);
        }
        let val = base + params.cmy_offsets[ch] + ev * params.ev_scale[ch];
        // Quadratic per-channel core (curvature 0 -> the original straight line).
        // gfac is the local grade: a slope rotation about this channel's pivot, so a
        // masked region's own midtone holds. Curvature stays global.
        var v = quadratic_core(params.slopes[ch] * gfac, params.pivots[ch], params.curvatures[ch], val);
        // Preflash, the frame's plus the masks': the flash exposure adds to the image
        // exposure (preflash_value).
        if (flash > 0.0) {
            v = params.hue.z + params.hue.w * log(pow(10.0, (v - params.hue.z) / params.hue.w) + flash) / 2.302585;
        }

        // Variable-gamma paper S-curve: extra local gamma at the midtone centre
        // (v_star), easing to zero toward toe/shoulder. Mirrors the CPU kernel.
        if (mg3[ch] != 0.0) {
            v = v + mg3[ch] * params.gamma_width * tanh((v - params.v_star) / params.gamma_width);
        }

        // Regional CMY: shadow weight rises with density, highlight falls.
        let w_sh = fast_sigmoid(3.0 * (v - params.zone_center));
        let w_hi = 1.0 - w_sh;
        v = v + params.shadow_cmy[ch] * w_sh + params.highlight_cmy[ch] * w_hi;

        // Split Grade: local contrast rotation about the zone centers, mid-
        // sparing. Own block before Zone Density (sequential stays monotone).
        let w_gsh = fast_sigmoid(4.0 * (v - (params.zone_center + 0.75)));
        let w_ghi = 1.0 - fast_sigmoid(4.0 * (v - (params.zone_center - 0.40)));
        v = v + params.split_sh[ch] * w_gsh * (v - (params.zone_center + 0.75)) + params.split_hi[ch] * w_ghi * (v - (params.zone_center - 0.40));

        // Zone Density (ΔD), mid-sparing weights; +0.75 / -0.40 / 4.0 mirror
        // the zone_density_* constants in models.py — change together.
        var w_zsh = fast_sigmoid(4.0 * (v - (params.zone_center + 0.75)));
        // Hold Black: a lift fades to nothing at paper black. Mirrors _black_hold.
        if (params.diff_range.w > 0.0 && params.shadow_density < 0.0) {
            let t = clamp((d_max_eff[ch] - v) / params.diff_range.w, 0.0, 1.0);
            w_zsh = w_zsh * t * t * (3.0 - 2.0 * t);
        }
        let w_zhi = 1.0 - fast_sigmoid(4.0 * (v - (params.zone_center - 0.40)));
        v = v + params.shadow_density * w_zsh + params.d_min_rgb.w * w_zhi;

        // Shoulder: smooth lower bound at paper white (highlights).
        let v1 = d_min_eff[ch] + softplus(a_hl[ch] * (v - d_min_eff[ch])) / a_hl[ch];
        // Toe: smooth upper bound at paper black (shadows).
        dens[ch] = d_max_eff[ch] - softplus(a_sh[ch] * (d_max_eff[ch] - v1)) / a_sh[ch];
    }

    // Dye unwanted absorptions: mix the densities above paper base.
    if (params.use_dye != 0u) {
        let e = dens - d_min_rgb;
        dens = d_min_rgb + vec3<f32>(
            dot(params.dye_r.xyz, e),
            dot(params.dye_g.xyz, e),
            dot(params.dye_b.xyz, e),
        );
    }

    // Separation Damping: chroma-selective dye separation, in place of the
    // frame-wide saturation matrix. Mirrors negpy_core print_curve.rs.
    if (params.sep_damping > 0.0) {
        let s = dens - d_min_rgb;
        let s_mean = (s.x + s.y + s.z) / 3.0;
        let d = vec3<f32>(s.x - s.y, s.y - s.z, s.x - s.z);
        let chroma = sqrt(dot(d, d) / 3.0);
        let k3 = vec3<f32>(params.sep_k_r, params.split_sh.w, params.split_hi.w);
        let kf = vec3<f32>(
            separation_damping_gain(k3.x, params.sep_damping, chroma),
            separation_damping_gain(k3.y, params.sep_damping, chroma),
            separation_damping_gain(k3.z, params.sep_damping, chroma),
        );
        dens = d_min_rgb + vec3<f32>(s_mean) + kf * (s - vec3<f32>(s_mean));
    }

    var transmittance = pow(vec3<f32>(10.0), -dens);
    if (params.lut.x > 0.0) {
        transmittance = pow(vec3<f32>(10.0), -d_min_rgb) * lut_tetra(dens - d_min_rgb);
        if (params.bpc != 0.0) {
            transmittance = (transmittance - vec3<f32>(params.lut.w)) / (1.0 - params.lut.w);
        }
    } else if (params.bpc != 0.0) {
        // BPC: physical paper black -> display 0; mirrors the CPU kernel prologue
        // (negative toe raises the clip point). oetf_encode clamps the tail to 0.
        let db = vec3<f32>(params.d_max) + select(vec3<f32>(0.0), toe3 * params.toe_height, toe_neg);
        let tb = pow(vec3<f32>(10.0), -db);
        transmittance = (transmittance - tb) / (vec3<f32>(1.0) - tb);
    }

    // B&W: re-collapse after the curve — per-channel trims must not tint a
    // B&W print. Mirrors the CPU post-curve collapse in exposure/processor.py.
    if (params.mode == 1u) {
        let l = dot(transmittance, vec3<f32>(0.2126, 0.7152, 0.0722));
        transmittance = vec3<f32>(l, l, l);
    }

    // Before the encode: the CPU rotates the same scene-linear buffer
    // (features/process/hue.py), which is what holds parity.
    if (params.hue.x != 0.0) {
        let lab = hue_rgb_to_lab(transmittance);
        let c = cos(params.hue.x);
        let s = sin(params.hue.x);
        let rotated = vec3<f32>(lab.x, lab.y * c - lab.z * s, lab.y * s + lab.z * c);
        transmittance = clamp(hue_lab_to_rgb(rotated), vec3<f32>(0.0), vec3<f32>(1.0));
    }

    let res = vec3<f32>(
        oetf_encode(transmittance.x),
        oetf_encode(transmittance.y),
        oetf_encode(transmittance.z),
    );

    textureStore(output_tex, coords, vec4<f32>(clamp(res, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
