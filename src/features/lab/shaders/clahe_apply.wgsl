struct ClaheUniforms {
    strength: f32,
    clip_limit: f32,
    pad: vec2<f32>,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<storage, read> cdfs: array<f32>;
@group(0) @binding(3) var<uniform> params: ClaheUniforms;

fn to_perceptual(c: vec3<f32>) -> vec3<f32> {
    return pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
}

fn to_linear(c: vec3<f32>) -> vec3<f32> {
    return pow(max(c, vec3<f32>(0.0)), vec3<f32>(2.2));
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
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let color = textureLoad(input_tex, coords, 0).rgb;
    
    // 1. Transform to perceptual space
    let p_color = to_perceptual(color);
    let luma = dot(p_color, vec3<f32>(0.2126, 0.7152, 0.0722));
    let bin = u32(clamp(luma * 255.0, 0.0, 255.0));

    // 2. Bilinear interpolation between tiles
    let fdims = vec2<f32>(f32(dims.x), f32(dims.y));
    let tile_pos = (vec2<f32>(f32(gid.x), f32(gid.y)) / fdims) * 8.0 - 0.5;
    
    let t_floor = vec2<i32>(floor(tile_pos));
    let t_ceil = t_floor + vec2<i32>(1, 1);
    let frac = tile_pos - floor(tile_pos);

    let v00 = get_cdf_val(u32(max(t_floor.x, 0)), u32(max(t_floor.y, 0)), bin);
    let v10 = get_cdf_val(u32(min(t_ceil.x, 7)),  u32(max(t_floor.y, 0)), bin);
    let v01 = get_cdf_val(u32(max(t_floor.x, 0)), u32(min(t_ceil.y, 7)),  bin);
    let v11 = get_cdf_val(u32(min(t_ceil.x, 7)),  u32(min(t_ceil.y, 7)),  bin);

    let cdf_luma = mix(mix(v00, v10, frac.x), mix(v01, v11, frac.x), frac.y);
    
    // 3. Blend and convert back to linear
    let final_luma = mix(luma, cdf_luma, params.strength);
    let ratio = final_luma / max(luma, 1e-6);
    
    let final_p_color = p_color * ratio;
    let final_linear = to_linear(final_p_color);
    
    textureStore(output_tex, coords, vec4<f32>(clamp(final_linear, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
