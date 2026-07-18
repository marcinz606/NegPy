struct ClaheUniforms {
    strength: f32,
    clip_limit: f32,
    global_offset: vec2<i32>,
    full_dims: vec2<i32>,
    pad: vec2<f32>,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<storage, read> cdfs: array<f32>;
@group(0) @binding(3) var<uniform> params: ClaheUniforms;

// Working-space TRC (Adobe RGB) — input/output texels stay OETF-encoded
// (retouch and lab downstream decode on load).
fn oetf_encode(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    return pow(x, vec3<f32>(0.45470693));
}

fn oetf_decode(c: vec3<f32>) -> vec3<f32> {
    let e = max(c, vec3<f32>(0.0));
    return pow(e, vec3<f32>(2.19921875));
}

fn rgb_to_lab(rgb: vec3<f32>) -> vec3<f32> {
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

    // XYZ -> Adobe RGB (1998), D65. Returns scene-linear (no encode).
    let r = x * 2.0413690 + y * -0.5649464 + z * -0.3446944;
    let g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560;
    let b = x * 0.0134474 + y * -0.1183897 + z * 1.0154096;

    return max(vec3<f32>(r, g, b), vec3<f32>(0.0));
}

fn get_cdf_val(tile_x: u32, tile_y: u32, bin: u32) -> f32 {
    let tx = clamp(tile_x, 0u, 7u);
    let ty = clamp(tile_y, 0u, 7u);
    let tile_idx = ty * 8u + tx;
    return cdfs[tile_idx * 256u + bin];
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let color = textureLoad(input_tex, coords, 0).rgb;

    let lab = rgb_to_lab(oetf_decode(color));
    let bin = u32(clamp(lab.x / 100.0 * 255.0, 0.0, 255.0));

    let global_pos = vec2<f32>(f32(coords.x + params.global_offset.x), f32(coords.y + params.global_offset.y));
    let full_fdims = vec2<f32>(f32(params.full_dims.x), f32(params.full_dims.y));
    
    let tile_pos = (global_pos / full_fdims) * 8.0 - 0.5;
    
    let t_floor = vec2<i32>(floor(tile_pos));
    let t_ceil = t_floor + vec2<i32>(1, 1);
    let raw_frac = tile_pos - floor(tile_pos);
    let frac = raw_frac * raw_frac * (3.0 - 2.0 * raw_frac);

    let v00 = get_cdf_val(u32(max(t_floor.x, 0)), u32(max(t_floor.y, 0)), bin);
    let v10 = get_cdf_val(u32(min(t_ceil.x, 7)),  u32(max(t_floor.y, 0)), bin);
    let v01 = get_cdf_val(u32(max(t_floor.x, 0)), u32(min(t_ceil.y, 7)),  bin);
    let v11 = get_cdf_val(u32(min(t_ceil.x, 7)),  u32(min(t_ceil.y, 7)),  bin);

    let cdf_luma = mix(mix(v00, v10, frac.x), mix(v01, v11, frac.x), frac.y);
    let l_new = mix(lab.x, cdf_luma * 100.0, params.strength);
    let rgb = lab_to_rgb(vec3<f32>(l_new, lab.y, lab.z));

    textureStore(output_tex, coords, vec4<f32>(oetf_encode(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0))), 1.0));
}
