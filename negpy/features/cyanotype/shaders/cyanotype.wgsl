struct CyanoUniforms {
    path_a: vec4<f32>,         // a* at u = 0.00 / 0.15 / 0.55 / 1.00
    path_b: vec4<f32>,         // b* at the same four knots
    over: f32,                 // 0.301 * exposure stops
    scale: f32,                // exposure scale in log D, floored at 0.1
    bleach: f32,
    tannin: f32,
    d_max: f32,
    brown_a: f32,              // (a*, b*) of iron tannate at full density
    brown_b: f32,
    pad: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: CyanoUniforms;

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

    // XYZ -> Adobe RGB (1998), D65 (matches CPU lab_to_rgb_working). Returns
    // scene-linear (no encode).
    let r = x * 2.0413690 + y * -0.5649464 + z * -0.3446944;
    let g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560;
    let b = x * 0.0134474 + y * -0.1183897 + z * 1.0154096;

    return max(vec3<f32>(r, g, b), vec3<f32>(0.0));
}

// np.interp over the fixed knots in CYANOTYPE_CONSTANTS["path_u"], clamped outside.
fn path_lerp(u: f32, v: vec4<f32>) -> f32 {
    if (u <= 0.00) { return v.x; }
    if (u < 0.15) { return mix(v.x, v.y, u / 0.15); }
    if (u < 0.55) { return mix(v.y, v.z, (u - 0.15) / 0.40); }
    if (u < 1.00) { return mix(v.z, v.w, (u - 0.55) / 0.45); }
    return v.w;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let rgb = clamp(textureLoad(input_tex, coords, 0).rgb, vec3<f32>(1e-6), vec3<f32>(1.0));
    let luma = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));

    // Mirrors apply_cyanotype. mid_compress / bleach_floor / tannin_restore /
    // tannin_dmax_gain mirror CYANOTYPE_CONSTANTS.
    let base = -log(clamp(luma, 1e-6, 1.0)) / log(10.0);
    let t = clamp((base + params.over) / params.scale, 0.0, 1.0);

    let m = 0.45;
    let v = 2.0 * t - 1.0;
    let u0 = (1.0 - m) * t + m * 0.5 * (1.0 + v * abs(v));

    let u_b = u0 * (1.0 - params.bleach * (1.0 - 0.15 * u0));
    let u = u_b + params.tannin * (u0 * 1.05 - u_b);
    let dens = params.d_max * (1.0 + params.tannin * 0.15) * u;

    let frac = clamp(u, 0.0, 1.0);
    let a_blue = path_lerp(frac, params.path_a);
    let b_blue = path_lerp(frac, params.path_b);
    let a_star = a_blue + params.tannin * (params.brown_a * frac - a_blue);
    let b_star = b_blue + params.tannin * (params.brown_b * frac - b_blue);

    let grey = pow(10.0, -dens);
    var fy = 0.0;
    if (grey > 0.008856) { fy = pow(grey, 1.0 / 3.0); } else { fy = 7.787 * grey + 16.0 / 116.0; }
    let color = lab_to_rgb(vec3<f32>(116.0 * fy - 16.0, a_star, b_star));

    textureStore(output_tex, coords, vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
