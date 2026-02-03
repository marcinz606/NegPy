struct ToningUniforms {
    saturation: f32,
    selenium_strength: f32,
    sepia_strength: f32,
    gamma: f32,         // Applied at the very end
    tint: vec4<f32>,    // rgb + dmax_boost
    crop_offset: vec2<i32>, // x, y offset in input texture
    is_bw: u32,         // 1 if B&W mode
    pad2: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: ToningUniforms;

fn rgb2hsv(c: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    let p = mix(vec4<f32>(c.bg, K.wz), vec4<f32>(c.gb, K.xy), step(c.b, c.g));
    let q = mix(vec4<f32>(p.xyw, c.r), vec4<f32>(c.r, p.yzx), step(p.x, c.r));
    let d = q.x - min(q.w, q.y);
    let e = 1.0e-10;
    return vec3<f32>(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), c.y);
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

    // 2. Saturation (Only if not B&W or if we want to saturate toning colors? 
    // Usually B&W implies saturation slider is ignored or works differently. 
    // In our app, let's follow CPU: desaturate first.)
    if (params.is_bw == 0u && params.saturation != 1.0) {
        var hsv = rgb2hsv(color);
        hsv.y = clamp(hsv.y * params.saturation, 0.0, 1.0);
        color = hsv2rgb(hsv);
    }

    // 3. Chemical Toning (Selenium/Sepia)
    let luma_toning = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    
    if (params.selenium_strength > 0.0) {
        let sel_m = clamp((1.0 - luma_toning) * (1.0 - luma_toning) * params.selenium_strength, 0.0, 1.0);
        color = mix(color, color * vec3<f32>(0.85, 0.75, 0.85), sel_m);
    }

    if (params.sepia_strength > 0.0) {
        let sep_m = exp(-pow(luma_toning - 0.6, 2.0) / 0.08) * params.sepia_strength;
        color = mix(color, color * vec3<f32>(1.1, 0.99, 0.825), sep_m);
    }

    // 4. Paper Tint / Dmax
    color = color * params.tint.rgb;
    if (params.tint.a != 1.0) {
        color = pow(color, vec3<f32>(params.tint.a));
    }

    // 5. Final Gamma Correction
    if (params.gamma > 0.0) {
        color = pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / params.gamma));
    }

    textureStore(output_tex, coords_out, vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}