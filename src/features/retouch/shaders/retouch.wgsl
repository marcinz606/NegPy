struct RetouchUniforms {
    dust_threshold: f32,
    dust_size: f32,
    num_manual_spots: u32,
    enabled_auto: u32,
    global_offset: vec2<i32>,
    full_dims: vec2<i32>,
    scale_factor: f32,
};

struct ManualSpot {
    pos: vec2<f32>,
    radius: f32,
    pad: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: RetouchUniforms;
@group(0) @binding(3) var<storage, read> manual_spots: array<ManualSpot>;

fn hash(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn get_noise(p: vec2<f32>) -> f32 {
    return hash(p) * 2.0 - 1.0;
}

fn median3x3(coords: vec2<i32>, dims: vec2<i32>) -> vec3<f32> {
    var r = array<f32, 9>();
    var g = array<f32, 9>();
    var b = array<f32, 9>();
    var idx = 0;
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            let sample = textureLoad(input_tex, clamp(coords + vec2<i32>(i, j), vec2<i32>(0), dims - 1), 0).rgb;
            r[idx] = sample.r; g[idx] = sample.g; b[idx] = sample.b;
            idx++;
        }
    }
    for (var i = 0; i < 9; i++) {
        for (var j = i + 1; j < 9; j++) {
            if (r[i] > r[j]) { let t = r[i]; r[i] = r[j]; r[j] = t; }
            if (g[i] > g[j]) { let t = g[i]; g[i] = g[j]; g[j] = t; }
            if (b[i] > b[j]) { let t = b[i]; b[i] = b[j]; b[j] = t; }
        }
    }
    return vec3<f32>(r[4], g[4], b[4]);
}

fn median5x5(coords: vec2<i32>, dims: vec2<i32>) -> vec3<f32> {
    var r = array<f32, 25>();
    var g = array<f32, 25>();
    var b = array<f32, 25>();
    var idx = 0;
    for (var j = -2; j <= 2; j++) {
        for (var i = -2; i <= 2; i++) {
            let sample = textureLoad(input_tex, clamp(coords + vec2<i32>(i, j), vec2<i32>(0), dims - 1), 0).rgb;
            r[idx] = sample.r; g[idx] = sample.g; b[idx] = sample.b;
            idx++;
        }
    }
    for (var i = 0; i <= 12; i++) {
        var min_idx_r = i;
        var min_idx_g = i;
        var min_idx_b = i;
        for (var j = i + 1; j < 25; j++) {
            if (r[j] < r[min_idx_r]) { min_idx_r = j; }
            if (g[j] < g[min_idx_g]) { min_idx_g = j; }
            if (b[j] < b[min_idx_b]) { min_idx_b = j; }
        }
        let tr = r[i]; r[i] = r[min_idx_r]; r[min_idx_r] = tr;
        let tg = g[i]; g[i] = g[min_idx_g]; g[min_idx_g] = tg;
        let tb = b[i]; b[i] = b[min_idx_b]; b[min_idx_b] = tb;
    }
    return vec3<f32>(r[12], g[12], b[12]);
}

fn median7x7(coords: vec2<i32>, dims: vec2<i32>) -> vec3<f32> {
    var luma = array<f32, 49>();
    var colors = array<vec3<f32>, 49>();
    var idx = 0;
    for (var j = -3; j <= 3; j++) {
        for (var i = -3; i <= 3; i++) {
            let s = textureLoad(input_tex, clamp(coords + vec2<i32>(i, j), vec2<i32>(0), dims - 1), 0).rgb;
            colors[idx] = s;
            luma[idx] = dot(s, vec3<f32>(0.2126, 0.7152, 0.0722));
            idx++;
        }
    }
    for (var i = 0; i <= 24; i++) {
        var min_idx = i;
        for (var j = i + 1; j < 49; j++) {
            if (luma[j] < luma[min_idx]) { min_idx = j; }
        }
        let tl = luma[i]; luma[i] = luma[min_idx]; luma[min_idx] = tl;
        let tc = colors[i]; colors[i] = colors[min_idx]; colors[min_idx] = tc;
    }
    return colors[24];
}

fn min3x3(coords: vec2<i32>, dims: vec2<i32>) -> vec3<f32> {
    var m = vec3<f32>(1.0, 1.0, 1.0);
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            let s = textureLoad(input_tex, clamp(coords + vec2<i32>(i, j), vec2<i32>(0), dims - 1), 0).rgb;
            m = min(m, s);
        }
    }
    return m;
}

fn min5x5(coords: vec2<i32>, dims: vec2<i32>) -> vec3<f32> {
    var m = vec3<f32>(1.0, 1.0, 1.0);
    for (var j = -2; j <= 2; j++) {
        for (var i = -2; i <= 2; i++) {
            let s = textureLoad(input_tex, clamp(coords + vec2<i32>(i, j), vec2<i32>(0), dims - 1), 0).rgb;
            m = min(m, s);
        }
    }
    return m;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let global_coords = vec2<f32>(f32(coords.x + params.global_offset.x) + 0.5, 
                                  f32(coords.y + params.global_offset.y) + 0.5);
    let global_uv = global_coords / vec2<f32>(f32(params.full_dims.x), f32(params.full_dims.y));
    
    let original = textureLoad(input_tex, coords, 0).rgb;
    var res = original;

    if (params.enabled_auto == 1u) {
        let base_s = max(1.0, params.dust_size);
        let scale = max(1.0, params.scale_factor);
        
        // 1. Resolution-Scaled Statistics Windows (Increased multipliers for large specks)
        let v_rad = i32(max(3.0, base_s * 3.0 * scale)); 
        var luma_sum = 0.0;
        var luma_sq_sum = 0.0;
        
        let step_v = max(1, v_rad / 4);
        var samples_v = 0.0;
        for (var j = -v_rad; j <= v_rad; j += step_v) {
            for (var i = -v_rad; i <= v_rad; i += step_v) {
                let s = textureLoad(input_tex, clamp(coords + vec2<i32>(i, j), vec2<i32>(0), vec2<i32>(dims) - 1), 0).rgb;
                let l = dot(s, vec3<f32>(0.2126, 0.7152, 0.0722));
                luma_sum += l;
                luma_sq_sum += l * l;
                samples_v += 1.0;
            }
        }
        let mean = luma_sum / samples_v;
        let luma_std = sqrt(max(0.0, (luma_sq_sum / samples_v) - (mean * mean)));

        let w_rad = i32(max(7.0, base_s * 4.0 * scale)); 
        var w_luma_sum = 0.0;
        var w_luma_sq_sum = 0.0;
        let step_w = max(1, w_rad / 6);
        var samples_w = 0.0;
        for (var j = -w_rad; j <= w_rad; j += step_w) {
            for (var i = -w_rad; i <= w_rad; i += step_w) {
                let s = textureLoad(input_tex, clamp(coords + vec2<i32>(i, j), vec2<i32>(0), vec2<i32>(dims) - 1), 0).rgb;
                let l = dot(s, vec3<f32>(0.2126, 0.7152, 0.0722));
                w_luma_sum += l;
                w_luma_sq_sum += l * l;
                samples_w += 1.0;
            }
        }
        let w_mean = w_luma_sum / samples_w;
        let w_std = sqrt(max(0.0, (w_luma_sq_sum / samples_w) - (w_mean * w_mean)));
        
        // 2. Multi-Scale Reference Selection
        var ref_val = vec3<f32>(0.0);
        if (scale > 1.5) {
            ref_val = min5x5(coords, vec2<i32>(dims));
            let r_off = i32(base_s * 3.0 * scale);
            ref_val = min(ref_val, textureLoad(input_tex, clamp(coords + vec2<i32>(-r_off, -r_off), vec2<i32>(0), vec2<i32>(dims) - 1), 0).rgb);
            ref_val = min(ref_val, textureLoad(input_tex, clamp(coords + vec2<i32>(r_off, r_off), vec2<i32>(0), vec2<i32>(dims) - 1), 0).rgb);
        } else {
            if (params.dust_size >= 2.5) {
                ref_val = median7x7(coords, vec2<i32>(dims));
            } else if (params.dust_size >= 1.5) {
                ref_val = median5x5(coords, vec2<i32>(dims));
            } else {
                let d_rad = i32(base_s * 3.0);
                ref_val = textureLoad(input_tex, clamp(coords + vec2<i32>(-d_rad, 0), vec2<i32>(0), vec2<i32>(dims) - 1), 0).rgb;
                ref_val = min(ref_val, textureLoad(input_tex, clamp(coords + vec2<i32>(d_rad, 0), vec2<i32>(0), vec2<i32>(dims) - 1), 0).rgb);
                ref_val = min(ref_val, median3x3(coords, vec2<i32>(dims)));
            }
        }

        // 3. Simple, Aggressive Detection
        let diff = original - ref_val;
        let max_pos_diff = max(diff.r, max(diff.g, diff.b));
        let luma = dot(original, vec3<f32>(0.2126, 0.7152, 0.0722));
        
        let local_s = max(0.005, luma_std);
        let z_score = (luma - mean) / local_s;
        
        let w_s = max(0.0, w_std - 0.02);
        let wide_penalty = (w_s * w_s * w_s) * 800.0;
        
        let thresh = (params.dust_threshold * 0.4) + (local_s * 1.0) + wide_penalty;

        // 4. Hit Detection with Strict Peak and Strong-Signal Bypass
        var strength = 0.0;
        let idims = vec2<i32>(dims);
        // Expansion radius scales with dust_size to handle large artifacts
        let exp_rad = i32(clamp(params.dust_size * 0.25 * params.scale_factor, 1.0, 6.0));
        
        for (var yoff = -exp_rad; yoff <= exp_rad; yoff++) {
            for (var xoff = -exp_rad; xoff <= exp_rad; xoff++) {
                let nc = clamp(coords + vec2<i32>(xoff, yoff), vec2<i32>(0), idims - 1);
                let ns = textureLoad(input_tex, nc, 0).rgb;
                let nl = dot(ns, vec3<f32>(0.2126, 0.7152, 0.0722));
                
                let n_diff = ns - ref_val; 
                let n_max_diff = max(n_diff.r, max(n_diff.g, n_diff.b));
                
                // Only treat as a hit if it's a statistically significant peak
                if (n_max_diff > thresh && nl > 0.15 && z_score > 3.0) {
                    let is_strong = n_max_diff > (thresh * 2.5) || n_max_diff > 0.25;
                    
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
                        strength = 1.0;
                    }
                }
            }
        }

        if (strength > 0.0) {
            let luma_mod = 4.0 * mean * (1.0 - mean);
            let grain = get_noise(global_uv * 1000.0) * 0.003 * luma_mod;
            res = mix(original, ref_val + vec3<f32>(grain), strength);
        }
    }

    for (var i = 0u; i < params.num_manual_spots; i++) {
        let spot = manual_spots[i];
        let d = distance(global_uv, spot.pos);
        if (d < spot.radius) {
            let pi = 3.14159265;
            let full_f = vec2<f32>(f32(params.full_dims.x), f32(params.full_dims.y));

            let delta = global_uv - spot.pos;
            let pixel_angle = atan2(delta.y, delta.x);

            let seed = global_coords + f32(i) * 7.77;
            var heal = vec3<f32>(0.0);
            for(var s = 0.0; s < 3.0; s += 1.0) {
                let jitter = (hash(seed + s * 0.555) - 0.5) * (pi * 0.2);
                let p_off = vec2<f32>(cos(pixel_angle + jitter), sin(pixel_angle + jitter)) * (spot.radius * 0.95);
                let pc = vec2<i32>((spot.pos + p_off) * full_f) - params.global_offset;
                heal += min3x3(pc, vec2<i32>(dims));
            }
            let healed_val = heal / 3.0;
            
            let current_luma = dot(res, vec3<f32>(0.2126, 0.7152, 0.0722));
            let heal_luma = dot(healed_val, vec3<f32>(0.2126, 0.7152, 0.0722));
            
            let luma_mask = smoothstep(0.04, 0.12, current_luma - heal_luma);

            let feather = smoothstep(spot.radius, spot.radius * 0.8, d);
            res = mix(res, healed_val, feather * luma_mask);
        }
    }

    textureStore(output_tex, coords, vec4<f32>(res, 1.0));
}

            