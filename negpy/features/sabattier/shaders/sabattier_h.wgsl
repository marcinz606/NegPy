// Sabattier, pass 2 of 3: the developed-silver mask blurred horizontally. The taps are
// gaussian_kernel_1d's, uploaded verbatim (sabattier_k buffer), so support and weights
// match the CPU's sepFilter2D; clamped coordinates are its BORDER_REPLICATE.
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
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: SabattierUniforms;
@group(0) @binding(3) var<storage, read> kernel_w: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec2<i32>(textureDimensions(output_tex));
    if (gid.x >= u32(dims.x) || gid.y >= u32(dims.y)) {
        return;
    }
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let r = i32(params.radius);
    var acc = 0.0;
    for (var i = -r; i <= r; i++) {
        let x = clamp(coords.x + i, 0, dims.x - 1);
        acc = acc + kernel_w[u32(i + r)] * textureLoad(input_tex, vec2<i32>(x, coords.y), 0).r;
    }
    textureStore(output_tex, coords, vec4<f32>(acc, 0.0, 0.0, 1.0));
}
