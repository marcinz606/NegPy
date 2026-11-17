struct VignetteUniforms {
    strength: f32,
    feather: f32,
    roundness: f32,
    crop_w: f32,
    crop_h: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: VignetteUniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    var color = textureLoad(input_tex, coords, 0).rgb;

    let cx = (params.crop_w - 1.0) * 0.5;
    let cy = (params.crop_h - 1.0) * 0.5;
    let max_dist = max(length(vec2<f32>(cx, cy)), 1.0);
    let dx = f32(coords.x) - cx;
    let dy = f32(coords.y) - cy;
    let rx = max(cx, 1.0);
    let ry = max(cy, 1.0);

    let d_circle = length(vec2<f32>(dx, dy)) / max_dist;
    let d_ellipse = sqrt((dx / rx) * (dx / rx) + (dy / ry) * (dy / ry));
    let dist = d_circle * (1.0 - params.roundness) + d_ellipse * params.roundness;

    let inner = 1.0 - params.feather;
    let span = max(1e-6, params.feather);
    let t = clamp((dist - inner) / span, 0.0, 1.0);
    let factor = 0.5 * (1.0 - cos(t * 3.14159265));
    let gain = exp2(params.strength * factor);

    color = color * gain;
    textureStore(output_tex, coords, vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
