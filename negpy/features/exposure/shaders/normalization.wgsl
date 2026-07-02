struct NormUniforms {
    floors: vec4<f32>,
    ceils: vec4<f32>,
    mode: u32,
    normalize_flag: u32,
    wp_offset: f32,
    bp_offset: f32,
    // Capture-side dye-unmix rows (effective blended+row-normalized matrix,
    // computed CPU-side); identity rows when the unmix is off. Applied to the
    // raw negative log densities before the stretch — mirrors the CPU path.
    unmix0: vec4<f32>,
    unmix1: vec4<f32>,
    unmix2: vec4<f32>,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: NormUniforms;

fn log10_vec(v: vec3<f32>) -> vec3<f32> {
    return log(v) * 0.43429448190325182765;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    var color = textureLoad(input_tex, coords, 0).rgb;
    
    let is_e6 = params.mode == 2u;

    let epsilon = 1e-6;
    var log_color = log10_vec(max(color, vec3<f32>(epsilon)));
    log_color = vec3<f32>(
        dot(params.unmix0.xyz, log_color),
        dot(params.unmix1.xyz, log_color),
        dot(params.unmix2.xyz, log_color),
    );

    var res: vec3<f32>;

    for (var ch = 0; ch < 3; ch++) {
        let f = params.floors[ch] + params.wp_offset;
        let c = params.ceils[ch] + params.bp_offset;
        
        let delta = c - f;
        var denom = delta;
        if (abs(delta) < epsilon) {
            if (delta >= 0.0) { denom = epsilon; }
            else { denom = -epsilon; }
        }

        let norm = (log_color[ch] - f) / denom;
        res[ch] = norm;
    }

    textureStore(output_tex, coords, vec4<f32>(res, 1.0));
}
