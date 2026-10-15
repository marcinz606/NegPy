// Sabattier, pass 3 of 3: the vertical blur of the mask (the bromide), then the fold.
// Mirrors apply_sabattier. With params.line == 0 the mask is never read.
struct SabattierUniforms {
    re: f32,
    fold_width: f32,
    edge_width: f32,
    strength: f32,
    radius: f32,
    line: f32,
    d_max: f32,
    pad: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var mask_tex: texture_2d<f32>;
@group(0) @binding(2) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform> params: SabattierUniforms;
@group(0) @binding(4) var<storage, read> kernel_w: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec2<i32>(textureDimensions(output_tex));
    if (gid.x >= u32(dims.x) || gid.y >= u32(dims.y)) {
        return;
    }
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let rgb = clamp(textureLoad(input_tex, coords, 0).rgb, vec3<f32>(1e-6), vec3<f32>(1.0));
    let luma = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let dens = -log(clamp(luma, 1e-6, 1.0)) / log(10.0);

    var bromide = 0.0;
    if (params.line > 0.0) {
        let r = i32(params.radius);
        for (var i = -r; i <= r; i++) {
            let y = clamp(coords.y + i, 0, dims.y - 1);
            bromide = bromide + kernel_w[u32(i + r)] * textureLoad(mask_tex, vec2<i32>(coords.x, y), 0).r;
        }
    }

    let w = params.fold_width;
    let shortfall = w * log(1.0 + exp(clamp((params.re - dens) / w, -30.0, 30.0)));
    let reversal = params.strength * shortfall * (1.0 - params.line * bromide);
    let grey = pow(10.0, -min(dens + reversal, params.d_max));
    textureStore(output_tex, coords, vec4<f32>(clamp(vec3<f32>(grey), vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
