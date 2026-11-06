struct NormUniforms {
    floors: vec4<f32>,
    ceils: vec4<f32>,
    mode: u32,
    // is_transfer_path() decided CPU-side: the shader must not re-derive it, or the two
    // engines drift the moment that rule grows a term (Positive already added one).
    transfer_flag: u32,
    wp_offset: f32,
    bp_offset: f32,
    // Log-density matrix rows, computed CPU-side (`unmix_with_fade`): the row-normalized
    // crosstalk unmix, times the fade restoration on a slide, which is not row-normalized.
    // Identity rows when neither applies. Applied before the stretch, as on the CPU.
    // The w lanes carry Remove Stain's log offset, added after the 3x3; 0.0 when it is off.
    unmix0: vec4<f32>,
    unmix1: vec4<f32>,
    unmix2: vec4<f32>,
    // Working-space-from-camera rows for the transparency transfer (identity when the
    // source carries no camera matrix). Applied in LINEAR, before the log — the print
    // path never uses them.
    cam0: vec4<f32>,
    cam1: vec4<f32>,
    cam2: vec4<f32>,
    // A camera's or a Positive source's primaries, applied in LINEAR after the density matrix so
    // the unmix, fade and stain offset act on its own channels (density_stage). post0.w is 1.0
    // when it applies; the cam rows are then identity, and white-balance gains ride the unmix w.
    post0: vec4<f32>,
    post1: vec4<f32>,
    post2: vec4<f32>,
    // The density anchor, on when w is 1.0: each channel is capped at it before the unmix,
    // and the amount the pixel is above it in every channel is added after, as a neutral.
    anchor: vec4<f32>,
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
    
    // The transparency transfer: camera primaries -> working space before the log.
    // Only that conversion is branch-specific.
    let is_transfer = params.transfer_flag == 1u;

    let epsilon = 1e-6;
    var lin = color;
    if (is_transfer) {
        lin = vec3<f32>(
            dot(params.cam0.xyz, color),
            dot(params.cam1.xyz, color),
            dot(params.cam2.xyz, color),
        );
    }
    var log_color = log10_vec(max(lin, vec3<f32>(epsilon)));
    // Unmix rides BOTH paths — the transfer honours a rig-calibrated matrix like it
    // honours Hue Trim. The CPU gates it by film process and packs identity rows when it
    // does not apply, so there is nothing to branch on here (and branching here is
    // exactly how this silently stopped working on the GPU once before).
    var excess = 0.0;
    if (params.anchor.w > 0.5) {
        let over = log_color - params.anchor.xyz;
        excess = max(min(over.x, min(over.y, over.z)), 0.0);
        log_color = min(log_color, params.anchor.xyz);
    }
    log_color = vec3<f32>(
        dot(params.unmix0.xyz, log_color) + excess + params.unmix0.w,
        dot(params.unmix1.xyz, log_color) + excess + params.unmix1.w,
        dot(params.unmix2.xyz, log_color) + excess + params.unmix2.w,
    );
    if (params.post0.w > 0.5) {
        let unmixed = pow(vec3<f32>(10.0), log_color);
        log_color = log10_vec(max(vec3<f32>(
            dot(params.post0.xyz, unmixed),
            dot(params.post1.xyz, unmixed),
            dot(params.post2.xyz, unmixed),
        ), vec3<f32>(epsilon)));
    }

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
