struct ExposureUniforms {
    pivots: vec4<f32>,
    slopes: vec4<f32>,
    cmy_offsets: vec4<f32>,
    shadow_cmy: vec4<f32>,
    highlight_cmy: vec4<f32>,
    toe: f32,
    toe_width: f32,
    toe_hardness: f32,
    shoulder: f32,
    shoulder_width: f32,
    shoulder_hardness: f32,
    shadows: f32,
    highlights: f32,
    d_max: f32,
    gamma: f32,
    mode: u32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
    pad3: vec4<f32>,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: ExposureUniforms;

fn fast_sigmoid(x: f32) -> f32 {
    if (x >= 0.0) {
        return 1.0 / (1.0 + exp(-x));
    } else {
        let z = exp(x);
        return z / (1.0 + z);
    }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let color = textureLoad(input_tex, coords, 0);
    
    var res: vec3<f32>;

    for (var ch = 0; ch < 3; ch++) {
        let val = color[ch] + params.cmy_offsets[ch];
        let diff = val - params.pivots[ch];
        let epsilon = 1e-6;

        let s_center = (1.0 - params.pivots[ch]) * 0.9;
        let h_center = (0.0 - params.pivots[ch]) * 0.9;

        let s_mask = exp(-( pow(diff - s_center, 2.0) ) / 0.15);
        let shadow_density_offset = params.shadows * s_mask * 0.3;
        let shadow_color_offset = params.shadow_cmy[ch] * s_mask;

        let h_mask = exp(-( pow(diff - h_center, 2.0) ) / 0.15);
        let highlight_density_offset = params.highlights * h_mask * 0.3;
        let highlight_color_offset = params.highlight_cmy[ch] * h_mask;

        let diff_adj = diff + shadow_color_offset + highlight_color_offset - shadow_density_offset - highlight_density_offset;

        let sw_val = params.shoulder_width * (diff_adj / max(params.pivots[ch], epsilon));
        let w_s = fast_sigmoid(sw_val);
        let prot_s = pow(4.0 * pow(w_s - 0.5, 2.0), params.shoulder_hardness);
        let damp_shoulder = params.shoulder * (1.0 - w_s) * prot_s;

        let tw_val = params.toe_width * (diff_adj / max(1.0 - params.pivots[ch], epsilon));
        let w_t = fast_sigmoid(tw_val);
        let prot_t = pow(4.0 * pow(w_t - 0.5, 2.0), params.toe_hardness);
        let damp_toe = params.toe * w_t * prot_t;

        var k_mod = 1.0 - damp_toe - damp_shoulder;
        k_mod = clamp(k_mod, 0.1, 2.0);

        var slope = params.slopes[ch];
        let density = params.d_max * fast_sigmoid(slope * diff_adj * k_mod);
        
        let transmittance = pow(10.0, -density);
        res[ch] = pow(max(transmittance, 0.0), 1.0 / params.gamma);
    }

    textureStore(output_tex, coords, vec4<f32>(clamp(res, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}