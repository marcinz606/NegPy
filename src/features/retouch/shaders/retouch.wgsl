struct RetouchUniforms {
    dust_threshold: f32,
    dust_size: f32,
    num_manual_spots: u32,
    enabled_auto: u32,
};

struct ManualSpot {
    pos: vec2<f32>,    // Normalized (0..1)
    radius: f32,       // Normalized radius
    pad: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: RetouchUniforms;
@group(0) @binding(3) var<storage, read> manual_spots: array<ManualSpot>;

// Simple PCG-based noise for grain synthesis
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

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let uv = vec2<f32>(f32(coords.x) + 0.5, f32(coords.y) + 0.5) / vec2<f32>(f32(dims.x), f32(dims.y));
    let original = textureLoad(input_tex, coords, 0).rgb;
    var res = original;

    // 1. Automatic Dust Removal
    if (params.enabled_auto == 1u) {
        let median = median3x3(coords, vec2<i32>(dims));
        let diff = abs(original - median);
        let max_diff = max(diff.r, max(diff.g, diff.b));
        let luma = dot(original, vec3<f32>(0.2126, 0.7152, 0.0722));
        let thresh = params.dust_threshold * (1.0 + luma * 0.5);
        if (max_diff > thresh) { res = median; }
    }

    // 2. Manual Healing with Grain Synthesis
    for (var i = 0u; i < params.num_manual_spots; i++) {
        let spot = manual_spots[i];
        let d = distance(uv, spot.pos);
        
        if (d < spot.radius) {
            // Advanced Multi-point Healing (16 directions)
            var heal = vec3<f32>(0.0);
            let search_radius = spot.radius * 1.1;
            let pi = 3.14159265;
            
            for(var step = 0.0; step < 16.0; step += 1.0) {
                let angle = step * (pi / 8.0);
                let offset = vec2<f32>(cos(angle), sin(angle)) * search_radius;
                let sample_uv = spot.pos + offset;
                let sample_coords = vec2<i32>(sample_uv * vec2<f32>(f32(dims.x), f32(dims.y)));
                heal += textureLoad(input_tex, clamp(sample_coords, vec2<i32>(0), vec2<i32>(dims) - 1), 0).rgb;
            }
            res = heal / 16.0;

            // Grain Synthesis (match surrounding film grain feel)
            let luma_grain = dot(res, vec3<f32>(0.2126, 0.7152, 0.0722));
            // Modulation matches CPU logic: 5.0 * lum * (1.0 - lum)
            let luma_mod = 5.0 * luma_grain * (1.0 - luma_grain);
            // We use a small noise scale to match scan granularity
            let grain = get_noise(uv * 1000.0) * 0.015 * luma_mod; 
            res = res + vec3<f32>(grain);

            let feather = smoothstep(spot.radius, spot.radius * 0.75, d);
            res = mix(original, res, feather);
            break; 
        }
    }

    textureStore(output_tex, coords, vec4<f32>(res, 1.0));
}