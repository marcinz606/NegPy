struct NormUniforms {
    floors: vec4<f32>,
    ceils: vec4<f32>,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: NormUniforms;

fn log10_vec(v: vec3<f32>) -> vec3<f32> {
    return log(v) * 0.43429448190325182765; // 1.0 / log(10.0)
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let color = textureLoad(input_tex, coords, 0).rgb;
    
    // 1. Log10 Conversion
    let epsilon = 1e-6;
    let log_color = log10_vec(max(color, vec3<f32>(epsilon)));
    
    // 2. Normalization (Stretch to 0.0-1.0)
    var res: vec3<f32>;
    for (var ch = 0; ch < 3; ch++) {
        let f = params.floors[ch];
        let c = params.ceils[ch];
        let norm = (log_color[ch] - f) / max(c - f, epsilon);
        res[ch] = clamp(norm, 0.0, 1.0);
    }

    textureStore(output_tex, coords, vec4<f32>(res, 1.0));
}
