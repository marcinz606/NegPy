struct RetouchUniforms {
    dust_threshold: f32,
    dust_size: f32,
    num_regions: u32,
    enabled_auto: u32,
    global_offset: vec2<i32>,
    full_dims: vec2<i32>,
    scale_factor: f32,
    ir_enabled: u32,
    ir_threshold: f32,
    ir_inpaint_radius: f32,
};

// Capsule-chain heal region: polyline points [pt_start, pt_start+pt_count) and
// ordered boundary-loop samples [bnd_start, bnd_start+bnd_count) index into
// heal_pts (global pixel coords). src_off is the clone-source offset in pixels.
struct HealRegion {
    pt_start: u32,
    pt_count: u32,
    bnd_start: u32,
    bnd_count: u32,
    radius: f32,
    pad0: f32,
    src_off: vec2<f32>,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: RetouchUniforms;
@group(0) @binding(3) var<storage, read> heal_regions: array<HealRegion>;
@group(0) @binding(4) var ir_tex: texture_2d<f32>;
@group(0) @binding(5) var<storage, read> heal_pts: array<vec2<f32>>;

fn hash(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Clone-sample dust guard (mirrors _sample_clean_jit): if the pixel's luma
// exceeds its 3x3 luma-median neighbour by CLONE_GUARD_LUMA it's a speck —
// return the median-luma pixel (a real pixel, grain preserved) instead, so
// dust in the source patch or on the boundary is never recloned.
// Ceiling: specks wider than ~2px fill the window and pass through; the
// source-offset scoring on the CPU avoids those upfront.
const CLONE_GUARD_LUMA: f32 = 0.06;

fn sample_clean(gp: vec2<f32>, idims: vec2<i32>) -> vec3<f32> {
    let gi = clamp(vec2<i32>(floor(gp)) - params.global_offset, vec2<i32>(0), idims - 1);
    var lums: array<f32, 9>;
    var cols: array<vec3<f32>, 9>;
    var n = 0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let sc = clamp(gi + vec2<i32>(dx, dy), vec2<i32>(0), idims - 1);
            let v = textureLoad(input_tex, sc, 0).rgb;
            cols[n] = v;
            lums[n] = dot(v, vec3<f32>(0.2126, 0.7152, 0.0722));
            n++;
        }
    }
    for (var i = 0; i <= 4; i++) {
        var mi = i;
        for (var j = i + 1; j < 9; j++) {
            if (lums[j] < lums[mi]) { mi = j; }
        }
        let tl = lums[i]; lums[i] = lums[mi]; lums[mi] = tl;
        let tc = cols[i]; cols[i] = cols[mi]; cols[mi] = tc;
    }
    let v = textureLoad(input_tex, gi, 0).rgb;
    if (dot(v, vec3<f32>(0.2126, 0.7152, 0.0722)) - lums[4] > CLONE_GUARD_LUMA) {
        return cols[4];
    }
    return v;
}

// 5x5 variant for the directly-cloned source sample — catches specks up to
// ~4px that slip through the 3x3 window. Mirrors _sample_clean5_jit.
fn sample_clean5(gp: vec2<f32>, idims: vec2<i32>) -> vec3<f32> {
    let gi = clamp(vec2<i32>(floor(gp)) - params.global_offset, vec2<i32>(0), idims - 1);
    var lums: array<f32, 25>;
    var cols: array<vec3<f32>, 25>;
    var n = 0;
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let sc = clamp(gi + vec2<i32>(dx, dy), vec2<i32>(0), idims - 1);
            let v = textureLoad(input_tex, sc, 0).rgb;
            cols[n] = v;
            lums[n] = dot(v, vec3<f32>(0.2126, 0.7152, 0.0722));
            n++;
        }
    }
    for (var i = 0; i <= 12; i++) {
        var mi = i;
        for (var j = i + 1; j < 25; j++) {
            if (lums[j] < lums[mi]) { mi = j; }
        }
        let tl = lums[i]; lums[i] = lums[mi]; lums[mi] = tl;
        let tc = cols[i]; cols[i] = cols[mi]; cols[mi] = tc;
    }
    let v = textureLoad(input_tex, gi, 0).rgb;
    if (dot(v, vec3<f32>(0.2126, 0.7152, 0.0722)) - lums[12] > CLONE_GUARD_LUMA) {
        return cols[12];
    }
    return v;
}

fn dist_to_seg(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let ab2 = dot(ab, ab);
    var t = 0.0;
    if (ab2 > 1e-12) { t = clamp(dot(p - a, ab) / ab2, 0.0, 1.0); }
    return distance(p, a + t * ab);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let idims = vec2<i32>(dims);
    let global_coords = vec2<f32>(f32(coords.x + params.global_offset.x) + 0.5,
                                  f32(coords.y + params.global_offset.y) + 0.5);
    let global_uv = global_coords / vec2<f32>(f32(params.full_dims.x), f32(params.full_dims.y));

    let original = textureLoad(input_tex, coords, 0).rgb;
    var res = original;

    if (params.enabled_auto == 1u) {
        let base_s = max(1.0, params.dust_size);
        let scale = max(1.0, params.scale_factor);

        let v_rad = i32(max(3.0, base_s * 3.0 * scale));
        var luma_sum = 0.0; var luma_sq_sum = 0.0;
        let step_v = max(1, v_rad / 4);
        var samples_v = 0.0;
        for (var j = -v_rad; j <= v_rad; j += step_v) {
            for (var i = -v_rad; i <= v_rad; i += step_v) {
                let s = textureLoad(input_tex, clamp(coords + vec2<i32>(i, j), vec2<i32>(0), idims - 1), 0).rgb;
                let l = dot(s, vec3<f32>(0.2126, 0.7152, 0.0722));
                luma_sum += l; luma_sq_sum += l * l; samples_v += 1.0;
            }
        }
        let mean = luma_sum / samples_v;
        let luma_std = sqrt(max(0.0, (luma_sq_sum / samples_v) - (mean * mean)));

        let w_rad = i32(max(7.0, base_s * 4.0 * scale));
        var w_luma_sum = 0.0; var w_luma_sq_sum = 0.0;
        let step_w = max(1, w_rad / 6);
        var samples_w = 0.0;
        for (var j = -w_rad; j <= w_rad; j += step_w) {
            for (var i = -w_rad; i <= w_rad; i += step_w) {
                let s = textureLoad(input_tex, clamp(coords + vec2<i32>(i, j), vec2<i32>(0), idims - 1), 0).rgb;
                let l = dot(s, vec3<f32>(0.2126, 0.7152, 0.0722));
                w_luma_sum += l; w_luma_sq_sum += l * l; samples_w += 1.0;
            }
        }
        let w_std = sqrt(max(0.0, (w_luma_sq_sum / samples_w) - (pow(w_luma_sum / samples_w, 2.0))));

        let luma = dot(original, vec3<f32>(0.2126, 0.7152, 0.0722));
        let local_s = max(0.005, luma_std);
        let z_score = (luma - mean) / local_s;
        let wide_penalty = pow(max(0.0, w_std - 0.02), 3.0) * 800.0;
        let thresh = (params.dust_threshold * 0.4) + (local_s * 1.0) + wide_penalty;

        var min_d2 = 1000.0;
        var c_x = 0.0; var c_y = 0.0; var c_n = 0.0;
        let exp_rad = i32(clamp(params.dust_size * 0.4 * scale, 1.0, 16.0));
        let p_rad = exp_rad + i32(3.0 * scale);

        for (var yoff = -exp_rad; yoff <= exp_rad; yoff++) {
            for (var xoff = -exp_rad; xoff <= exp_rad; xoff++) {
                let nc = clamp(coords + vec2<i32>(xoff, yoff), vec2<i32>(0), idims - 1);
                let ns = textureLoad(input_tex, nc, 0).rgb;
                let nl = dot(ns, vec3<f32>(0.2126, 0.7152, 0.0722));
                let n_diff = nl - mean;

                if (n_diff > thresh && nl > 0.15 && (nl - mean) / local_s > 3.0) {
                    let is_strong = n_diff > (thresh * 2.5) || n_diff > 0.25;
                    var is_max = true;
                    for (var my = -1; my <= 1; my++) {
                        for (var mx = -1; mx <= 1; mx++) {
                            if (mx == 0 && my == 0) { continue; }
                            let sc = clamp(nc + vec2<i32>(mx, my), vec2<i32>(0), idims - 1);
                            let sl = dot(textureLoad(input_tex, sc, 0).rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
                            if (sl >= nl) { is_max = false; break; }
                        }
                        if (!is_max) { break; }
                    }
                    if (is_max || is_strong) {
                        let d2 = f32(xoff*xoff + yoff*yoff);
                        if (d2 < min_d2) { min_d2 = d2; }
                        c_x += f32(nc.x); c_y += f32(nc.y); c_n += 1.0;
                    }
                }
            }
        }

        if (min_d2 < 1000.0) {
            let dist = sqrt(min_d2);
            var feather = 1.0 - (dist / f32(exp_rad + 1));
            feather = clamp(feather, 0.0, 1.0);
            feather = feather * feather * (3.0 - 2.0 * feather);

            // Guarded reflection copy: sample outward from the defect centroid so
            // neighbouring pixels copy neighbouring sources (grain continuity).
            var u = vec2<f32>(f32(coords.x), f32(coords.y)) - vec2<f32>(c_x, c_y) / c_n;
            let ul = length(u);
            if (ul < 1e-3) {
                let ang = hash(vec2<f32>(f32(coords.x + params.global_offset.x), f32(coords.y + params.global_offset.y))) * 6.28318530718;
                u = vec2<f32>(cos(ang), sin(ang));
            } else {
                u = u / ul;
            }

            // Guard rotations (0°, ±45°, ±90°) — order mirrored in the CPU engine.
            let g_cos = array<f32, 5>(1.0, 0.70710678, 0.70710678, 0.0, 0.0);
            let g_sin = array<f32, 5>(0.0, 0.70710678, -0.70710678, 1.0, -1.0);
            var found = false;
            var healed_val = vec3<f32>(0.0);
            for (var k = 0; k < 5; k++) {
                let dir = vec2<f32>(u.x * g_cos[k] - u.y * g_sin[k],
                                    u.x * g_sin[k] + u.y * g_cos[k]);
                let sp = clamp(vec2<i32>(round(vec2<f32>(coords) + dir * f32(p_rad))), vec2<i32>(0), idims - 1);
                let sv = textureLoad(input_tex, sp, 0).rgb;
                let sl = dot(sv, vec3<f32>(0.2126, 0.7152, 0.0722));
                // Guard: source must not be a defect itself.
                if (sl - mean <= params.dust_threshold * 0.4) {
                    healed_val = sv;
                    found = true;
                    break;
                }
            }

            if (!found) {
                // Fallback: 8-point trimmed sampling (previous fill — worst case unchanged).
                var s_r = array<f32, 8>(); var s_g = array<f32, 8>(); var s_b = array<f32, 8>(); var s_l = array<f32, 8>();
                let dxs = array<i32, 8>(-p_rad, p_rad, 0, 0, -p_rad, -p_rad, p_rad, p_rad);
                let dys = array<i32, 8>(0, 0, -p_rad, p_rad, -p_rad, p_rad, -p_rad, p_rad);

                for (var i = 0; i < 8; i++) {
                    let pix = textureLoad(input_tex, clamp(coords + vec2<i32>(dxs[i], dys[i]), vec2<i32>(0), idims - 1), 0).rgb;
                    s_r[i] = pix.r; s_g[i] = pix.g; s_b[i] = pix.b;
                    s_l[i] = dot(pix, vec3<f32>(0.2126, 0.7152, 0.0722));
                }

                for (var i = 0; i < 7; i++) {
                    for (var j = i + 1; j < 8; j++) {
                        if (s_l[i] > s_l[j]) {
                            let tl = s_l[i]; s_l[i] = s_l[j]; s_l[j] = tl;
                            let tr = s_r[i]; s_r[i] = s_r[j]; s_r[j] = tr;
                            let tg = s_g[i]; s_g[i] = s_g[j]; s_g[j] = tg;
                            let tb = s_b[i]; s_b[i] = s_b[j]; s_b[j] = tb;
                        }
                    }
                }

                healed_val = vec3<f32>(
                    (s_r[2] + s_r[3] + s_r[4] + s_r[5]) / 4.0,
                    (s_g[2] + s_g[3] + s_g[4] + s_g[5]) / 4.0,
                    (s_b[2] + s_b[3] + s_b[4] + s_b[5]) / 4.0
                );
            }

            res = mix(original, healed_val, feather);
        }
    }

    if (params.ir_enabled == 1u) {
        let ir_scale = max(1.0, params.scale_factor);
        let ir_exp_rad = i32(clamp(params.ir_inpaint_radius * ir_scale, 1.0, 16.0));
        let ir_p_rad = ir_exp_rad + i32(max(2.0, 3.0 * ir_scale));

        var ir_min_d2 = 1.0e9;
        var ir_cx = 0.0; var ir_cy = 0.0; var ir_cn = 0.0;
        for (var yoff = -ir_exp_rad; yoff <= ir_exp_rad; yoff++) {
            for (var xoff = -ir_exp_rad; xoff <= ir_exp_rad; xoff++) {
                let nc = clamp(coords + vec2<i32>(xoff, yoff), vec2<i32>(0), idims - 1);
                if (textureLoad(ir_tex, nc, 0).r < params.ir_threshold) {
                    let d2 = f32(xoff*xoff + yoff*yoff);
                    if (d2 < ir_min_d2) { ir_min_d2 = d2; }
                    ir_cx += f32(nc.x); ir_cy += f32(nc.y); ir_cn += 1.0;
                }
            }
        }

        if (ir_min_d2 < f32(ir_exp_rad * ir_exp_rad + 1)) {
            let dist = sqrt(ir_min_d2);
            var ir_feather = clamp(1.0 - dist / f32(ir_exp_rad + 1), 0.0, 1.0);
            ir_feather = ir_feather * ir_feather * (3.0 - 2.0 * ir_feather);

            var u = vec2<f32>(f32(coords.x), f32(coords.y)) - vec2<f32>(ir_cx, ir_cy) / ir_cn;
            let ul = length(u);
            if (ul < 1e-3) {
                let ang = hash(vec2<f32>(f32(coords.x + params.global_offset.x), f32(coords.y + params.global_offset.y))) * 6.28318530718;
                u = vec2<f32>(cos(ang), sin(ang));
            } else {
                u = u / ul;
            }

            let g_cos = array<f32, 5>(1.0, 0.70710678, 0.70710678, 0.0, 0.0);
            let g_sin = array<f32, 5>(0.0, 0.70710678, -0.70710678, 1.0, -1.0);
            var found = false;
            var ir_healed = vec3<f32>(0.0);
            for (var k = 0; k < 5; k++) {
                let dir = vec2<f32>(u.x * g_cos[k] - u.y * g_sin[k],
                                    u.x * g_sin[k] + u.y * g_cos[k]);
                let sp = clamp(vec2<i32>(round(vec2<f32>(coords) + dir * f32(ir_p_rad))), vec2<i32>(0), idims - 1);
                if (textureLoad(ir_tex, sp, 0).r >= params.ir_threshold) {
                    ir_healed = textureLoad(input_tex, sp, 0).rgb;
                    found = true;
                    break;
                }
            }

            if (!found) {
                var ir_sr = array<f32, 8>(); var ir_sg = array<f32, 8>(); var ir_sb = array<f32, 8>(); var ir_sl = array<f32, 8>();
                let ir_dxs = array<i32, 8>(-ir_p_rad, ir_p_rad, 0, 0, -ir_p_rad, -ir_p_rad, ir_p_rad, ir_p_rad);
                let ir_dys = array<i32, 8>(0, 0, -ir_p_rad, ir_p_rad, -ir_p_rad, ir_p_rad, -ir_p_rad, ir_p_rad);
                for (var i = 0; i < 8; i++) {
                    let sc = clamp(coords + vec2<i32>(ir_dxs[i], ir_dys[i]), vec2<i32>(0), idims - 1);
                    let pix = textureLoad(input_tex, sc, 0).rgb;
                    ir_sr[i] = pix.r; ir_sg[i] = pix.g; ir_sb[i] = pix.b;
                    ir_sl[i] = dot(pix, vec3<f32>(0.2126, 0.7152, 0.0722));
                }
                for (var i = 0; i < 7; i++) {
                    for (var j = i + 1; j < 8; j++) {
                        if (ir_sl[i] > ir_sl[j]) {
                            let tl = ir_sl[i]; ir_sl[i] = ir_sl[j]; ir_sl[j] = tl;
                            let tr = ir_sr[i]; ir_sr[i] = ir_sr[j]; ir_sr[j] = tr;
                            let tg = ir_sg[i]; ir_sg[i] = ir_sg[j]; ir_sg[j] = tg;
                            let tb = ir_sb[i]; ir_sb[i] = ir_sb[j]; ir_sb[j] = tb;
                        }
                    }
                }
                ir_healed = vec3<f32>(
                    (ir_sr[2] + ir_sr[3] + ir_sr[4] + ir_sr[5]) / 4.0,
                    (ir_sg[2] + ir_sg[3] + ir_sg[4] + ir_sg[5]) / 4.0,
                    (ir_sb[2] + ir_sb[3] + ir_sb[4] + ir_sb[5]) / 4.0,
                );
            }
            res = mix(res, ir_healed, ir_feather);
        }
    }

    // Manual heals: mean-value-coordinates membrane clone (Georgiev healing
    // brush). out = src_patch + MVC-interpolated boundary difference — copied
    // pixels carry real grain, the membrane matches the rim seamlessly.
    for (var ri = 0u; ri < params.num_regions; ri++) {
        let reg = heal_regions[ri];
        if (reg.bnd_count < 3u || reg.bnd_count > 64u || reg.pt_count < 1u) { continue; }

        let p = global_coords;
        var d = 1e18;
        if (reg.pt_count == 1u) {
            d = distance(p, heal_pts[reg.pt_start]);
        } else {
            for (var s = 0u; s + 1u < reg.pt_count; s++) {
                d = min(d, dist_to_seg(p, heal_pts[reg.pt_start + s], heal_pts[reg.pt_start + s + 1u]));
            }
        }
        if (d >= reg.radius) { continue; }

        let n = reg.bnd_count;
        var vxs: array<f32, 64>; var vys: array<f32, 64>; var vls: array<f32, 64>;
        var diffs: array<vec3<f32>, 64>;
        var on_sample = -1;
        for (var i = 0u; i < n; i++) {
            let b = heal_pts[reg.bnd_start + i];
            diffs[i] = sample_clean(b, idims) - sample_clean(b + reg.src_off, idims);
            let v = b - p;
            let l = length(v);
            vxs[i] = v.x; vys[i] = v.y; vls[i] = l;
            if (l < 1e-4) { on_sample = i32(i); }
        }

        var mem = vec3<f32>(0.0);
        if (on_sample >= 0) {
            mem = diffs[on_sample];
        } else {
            var tans: array<f32, 64>;
            for (var i = 0u; i < n; i++) {
                var j = i + 1u;
                if (j == n) { j = 0u; }
                var cr = vxs[i] * vys[j] - vys[i] * vxs[j];
                if (abs(cr) < 1e-9) { cr = 1e-9; }
                tans[i] = (vls[i] * vls[j] - (vxs[i] * vxs[j] + vys[i] * vys[j])) / cr;
            }
            var wsum = 0.0;
            for (var i = 0u; i < n; i++) {
                var prev = n - 1u;
                if (i > 0u) { prev = i - 1u; }
                let wi = (tans[prev] + tans[i]) / vls[i];
                wsum += wi;
                mem += wi * diffs[i];
            }
            if (abs(wsum) < 1e-12) { continue; }
            mem /= wsum;
        }

        let healed = sample_clean5(p + reg.src_off, idims) + mem;
        // 1.5px feather at the rim hides boundary-sampling aliasing.
        let t = clamp((d - (reg.radius - 1.5)) / 1.5, 0.0, 1.0);
        var alpha = 1.0 - t * t * (3.0 - 2.0 * t);
        // Dust gate: heal only pixels brighter than the membrane-predicted
        // clean value — the brush is a search area, not a clone stamp.
        let gate = smoothstep(0.04, 0.12, dot(res, vec3<f32>(0.2126, 0.7152, 0.0722)) - dot(healed, vec3<f32>(0.2126, 0.7152, 0.0722)));
        alpha *= gate;
        res = mix(res, healed, alpha);
    }

    textureStore(output_tex, coords, vec4<f32>(res, 1.0));
}
