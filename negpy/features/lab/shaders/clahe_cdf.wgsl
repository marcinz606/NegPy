struct ClaheUniforms {
    strength: f32,
    clip_limit: f32,
    global_offset: vec2<i32>,
    full_dims: vec2<i32>,
    pad: vec2<f32>,
};

@group(0) @binding(0) var<storage, read> histograms: array<u32>;
@group(0) @binding(1) var<storage, read_write> cdfs: array<f32>;
@group(0) @binding(2) var<uniform> params: ClaheUniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tile_idx = wid.y * 8u + wid.x;
    let offset = tile_idx * 256u;

    if (lid == 0u) {
        var total: u32 = 0u;
        for (var i = 0u; i < 256u; i++) {
            total += histograms[offset + i];
        }

        let limit = max(1u, u32(params.clip_limit * f32(total) / 256.0));
        var excess: u32 = 0u;
        for (var i = 0u; i < 256u; i++) {
            let count = histograms[offset + i];
            if (count > limit) {
                excess += (count - limit);
            }
        }

        let inc = excess / 256u;
        let rem = excess % 256u;
        var count_acc: u32 = 0u;
        for (var i = 0u; i < 256u; i++) {
            var count = min(histograms[offset + i], limit) + inc;
            if (i < rem) { count++; }
            
            count_acc += count;
            cdfs[offset + i] = f32(count_acc) / f32(total);
        }
    }
}