struct LithUniforms {
    path_a: vec4<f32>,         // a* at u = 0.10 / 0.35 / 0.65 / 1.00
    path_b: vec4<f32>,         // b* at the same four knots
    over: f32,                 // 0.301 * exposure stops
    over_foot: f32,            // over * foot_veil
    knee: f32,                 // knee density, from the snatch point
    width: f32,                // knee width, from abruptness
    d_max: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: LithUniforms;

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

// np.interp over the fixed knots in LITH_CONSTANTS["path_u"], clamped outside.
fn path_lerp(u: f32, v: vec4<f32>) -> f32 {
    if (u <= 0.10) { return v.x; }
    if (u < 0.35) { return mix(v.x, v.y, (u - 0.10) / 0.25); }
    if (u < 0.65) { return mix(v.y, v.z, (u - 0.35) / 0.30); }
    if (u < 1.00) { return mix(v.z, v.w, (u - 0.65) / 0.35); }
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

    // Mirrors apply_lith: an exposure-only highlight branch plus an infectious
    // knee. foot_max / foot_rate mirror LITH_CONSTANTS.
    let base = -log(clamp(luma, 1e-6, 1.0)) / log(10.0);
    let d0 = base + params.over;

    let foot_max = 0.70;
    let foot_rate = 0.60;
    let d_foot = base + params.over_foot;
    let d_h = foot_max * (1.0 - exp(-foot_rate * d_foot / foot_max));

    let g = 1.0 / (1.0 + exp(-clamp((d0 - params.knee) / params.width, -30.0, 30.0)));
    let dens = d_h + (params.d_max - d_h) * g;

    let grey = pow(10.0, -dens);
    let u = clamp(dens / params.d_max, 0.0, 1.0);
    var fy = 0.0;
    if (grey > 0.008856) { fy = pow(grey, 1.0 / 3.0); } else { fy = 7.787 * grey + 16.0 / 116.0; }
    let color = lab_to_rgb(vec3<f32>(116.0 * fy - 16.0, path_lerp(u, params.path_a), path_lerp(u, params.path_b)));

    textureStore(output_tex, coords, vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
