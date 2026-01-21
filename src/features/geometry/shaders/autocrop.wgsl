@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> row_sums: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> col_sums: array<atomic<u32>>;

const LUMA_COEFFS = vec3<f32>(0.2126, 0.7152, 0.0722);

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let color = textureLoad(input_tex, coords, 0).rgb;
    
    // Perceptual luma for more accurate border detection
    let p_color = pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
    let luma = dot(p_color, LUMA_COEFFS);
    
    // Use fixed-point scaling for atomic compatibility (0.0..1.0 -> 0..1000000)
    let val = u32(clamp(luma * 1000000.0, 0.0, 1000000.0));
    
    atomicAdd(&row_sums[gid.y], val);
    atomicAdd(&col_sums[gid.x], val);
}
