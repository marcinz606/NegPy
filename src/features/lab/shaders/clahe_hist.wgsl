@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> histograms: array<u32>; // 8x8 tiles * 256 bins

var<workgroup> local_hist: array<atomic<u32>, 256>;

fn to_perceptual(c: vec3<f32>) -> vec3<f32> {
    return pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
}

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    if (lid < 256u) {
        atomicStore(&local_hist[lid], 0u);
    }
    workgroupBarrier();

    let dims = textureDimensions(input_tex);
    let tile_size = (dims + vec2<u32>(7u)) / 8u;
    
    let x_start = wid.x * tile_size.x;
    let y_start = wid.y * tile_size.y;
    let x_end = min(x_start + tile_size.x, dims.x);
    let y_end = min(y_start + tile_size.y, dims.y);

    for (var y = y_start + (lid / 16u); y < y_end; y += 16u) {
        for (var x = x_start + (lid % 16u); x < x_end; x += 16u) {
            let color = textureLoad(input_tex, vec2<i32>(i32(x), i32(y)), 0).rgb;
            // Calculate luma in perceptual space for histogram
            let p_color = to_perceptual(color);
            let luma = dot(p_color, vec3<f32>(0.2126, 0.7152, 0.0722));
            let bin = u32(clamp(luma * 255.0, 0.0, 255.0));
            atomicAdd(&local_hist[bin], 1u);
        }
    }
    workgroupBarrier();

    if (lid < 256u) {
        let tile_idx = wid.y * 8u + wid.x;
        histograms[tile_idx * 256u + lid] = atomicLoad(&local_hist[lid]);
    }
}