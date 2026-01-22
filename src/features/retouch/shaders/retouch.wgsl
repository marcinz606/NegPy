struct RetouchUniforms {
    dust_threshold: f32,
    dust_size: f32,
    num_manual_spots: u32,
    enabled_auto: u32,
    global_offset: vec2<i32>,
    full_dims: vec2<i32>,
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
        // 1. Local Statistics (matching CPU 15x15 blur window)
        var luma_sum = 0.0;
        var luma_sq_sum = 0.0;
        let v_rad = 7; 
        let v_count = 225.0;
        
        for (var j = -v_rad; j <= v_rad; j++) {
            for (var i = -v_rad; i <= v_rad; i++) {
                let s = textureLoad(input_tex, clamp(coords + vec2<i32>(i, j), vec2<i32>(0), vec2<i32>(dims) - 1), 0).rgb;
                let l = dot(s, vec3<f32>(0.2126, 0.7152, 0.0722));
                luma_sum += l;
                luma_sq_sum += l * l;
            }
        }
        let mean = luma_sum / v_count;
        let luma_std = sqrt(max(0.0, (luma_sq_sum / v_count) - (mean * mean)));
        
        // 2. Exact CPU Flatness Logic
        let flatness = clamp(1.0 - (luma_std / 0.08), 0.0, 1.0);
        let flatness_weight = sqrt(flatness);
        let brightness = clamp(mean, 0.0, 1.0);
        let highlight_sens = clamp((brightness - 0.4) * 1.5, 0.0, 1.0);

        // 3. Adaptive Threshold (Exact CPU formulas)
        let detail_boost = (1.0 - flatness) * 0.05;
        let sens_factor = (1.0 - 0.98 * flatness_weight) * (1.0 - 0.5 * highlight_sens);
        
        // Final Threshold Calculation
        let final_thresh = params.dust_threshold * sens_factor + detail_boost;

        // 4. Detection and Healing
        if (luma_std <= 0.2) { // Strict CPU variance cap
            var median = vec3<f32>(0.0);
            if (params.dust_size > 2.0) {
                median = median5x5(coords, vec2<i32>(dims));
            } else {
                median = median3x3(coords, vec2<i32>(dims));
            }

            let diff = original - median;
            let max_pos_diff = max(diff.r, max(diff.g, diff.b));

            if (max_pos_diff > final_thresh) {
                // Approximate CPU mask feathering
                let strength = smoothstep(final_thresh, final_thresh * 1.2, max_pos_diff);
                
                // Keep grain matching for better blending than pure median
                let luma_mod = 5.0 * mean * (1.0 - mean);
                let grain = get_noise(global_uv * 1000.0) * 0.015 * luma_mod;
                let healed = median + vec3<f32>(grain);
                
                res = mix(original, healed, strength);
            }
        }
    }

    // Manual Healing Section
    for (var i = 0u; i < params.num_manual_spots; i++) {
        let spot = manual_spots[i];
        let d = distance(global_uv, spot.pos);
        if (d < spot.radius) {
            var heal = vec3<f32>(0.0);
            let search_radius = spot.radius * 1.1;
            let pi = 3.14159265;
            for(var step = 0.0; step < 16.0; step += 1.0) {
                let angle = step * (pi / 8.0);
                let offset_uv = vec2<f32>(cos(angle), sin(angle)) * search_radius;
                let sample_uv = spot.pos + offset_uv;
                let sample_global_coords = sample_uv * vec2<f32>(f32(params.full_dims.x), f32(params.full_dims.y));
                let sample_tile_coords = vec2<i32>(sample_global_coords) - params.global_offset;
                heal += textureLoad(input_tex, clamp(sample_tile_coords, vec2<i32>(0), vec2<i32>(dims) - 1), 0).rgb;
            }
            res = heal / 16.0;
            let luma_grain = dot(res, vec3<f32>(0.2126, 0.7152, 0.0722));
            let luma_mod = 5.0 * luma_grain * (1.0 - luma_grain);
            let grain = get_noise(global_uv * 1000.0) * 0.015 * luma_mod; 
            res = res + vec3<f32>(grain);
            let feather = smoothstep(spot.radius, spot.radius * 0.75, d);
            res = mix(original, res, feather);
            break; 
        }
    }

    textureStore(output_tex, coords, vec4<f32>(res, 1.0));
}