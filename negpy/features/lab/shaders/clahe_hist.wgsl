@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> histograms: array<u32>; // 8x8 tiles * 256 bins

var<workgroup> local_hist: array<atomic<u32>, 256>;

// Working-space TRC (ProPhoto ROMM) — input texels arrive OETF-encoded.
fn oetf_decode(c: vec3<f32>) -> vec3<f32> {
    let e = max(c, vec3<f32>(0.0));
    return select(pow(e, vec3<f32>(1.8)), e / 16.0, e < vec3<f32>(0.03125));
}

// CIELAB L* (Y row of the ProPhoto->XYZ matrix only — a*/b* unused here).
// Must mirror rgb_to_lab in clahe_apply.wgsl / lab.wgsl op-for-op so hist
// and apply agree on bins.
fn lab_l(rgb: vec3<f32>) -> f32 {
    let r = max(rgb.r, 0.0);
    let g = max(rgb.g, 0.0);
    let b = max(rgb.b, 0.0);
    var y = r * 0.2880402 + g * 0.7118741 + b * 0.0000857;
    if (y > 0.008856) { y = pow(y, 1.0/3.0); } else { y = (7.787 * y) + (16.0 / 116.0); }
    return (116.0 * y) - 16.0;
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
            let l = lab_l(oetf_decode(color));
            let bin = u32(clamp(l / 100.0 * 255.0, 0.0, 255.0));
            atomicAdd(&local_hist[bin], 1u);
        }
    }
    workgroupBarrier();

    if (lid < 256u) {
        let tile_idx = wid.y * 8u + wid.x;
        histograms[tile_idx * 256u + lid] = atomicLoad(&local_hist[lid]);
    }
}