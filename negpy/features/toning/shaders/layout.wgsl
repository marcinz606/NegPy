struct LayoutUniforms {
    bg_color: vec4<f32>,
    offset: vec2<i32>,
    content_dims: vec2<i32>, // Size of image on paper (px)
    source_dims: vec2<i32>,  // Size of incoming texture (px)
    scale: f32,              // scale factor: paper_content / source
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: LayoutUniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dims = textureDimensions(output_tex);
    if (gid.x >= out_dims.x || gid.y >= out_dims.y) {
        return;
    }

    let coords = vec2<i32>(i32(gid.x), i32(gid.y));

    // Check if within content area
    let local_x = f32(coords.x - params.offset.x);
    let local_y = f32(coords.y - params.offset.y);

    if (local_x >= 0.0 && local_x < f32(params.content_dims.x) &&
        local_y >= 0.0 && local_y < f32(params.content_dims.y)) {

        // Shrinking averages each output pixel's source footprint, as INTER_AREA does on
        // the CPU; enlarging interpolates between pixel centres.
        let inv = vec2<f32>(
            f32(params.source_dims.x) / f32(params.content_dims.x),
            f32(params.source_dims.y) / f32(params.content_dims.y),
        );
        let last = params.source_dims - vec2<i32>(1, 1);
        var color = vec4<f32>(0.0);
        if (inv.x > 1.0 || inv.y > 1.0) {
            let lo = vec2<f32>(local_x, local_y) * inv;
            let hi = lo + inv;
            var total = 0.0;
            for (var iy = i32(floor(lo.y)); f32(iy) < hi.y; iy++) {
                let wy = min(f32(iy + 1), hi.y) - max(f32(iy), lo.y);
                for (var ix = i32(floor(lo.x)); f32(ix) < hi.x; ix++) {
                    let wx = min(f32(ix + 1), hi.x) - max(f32(ix), lo.x);
                    let p = clamp(vec2<i32>(ix, iy), vec2<i32>(0, 0), last);
                    color += textureLoad(input_tex, p, 0) * (wx * wy);
                    total += wx * wy;
                }
            }
            color /= max(total, 1e-8);
        } else {
            let src = max((vec2<f32>(local_x, local_y) + 0.5) * inv - 0.5, vec2<f32>(0.0));
            let p1 = min(vec2<i32>(floor(src)), last);
            let p2 = min(p1 + vec2<i32>(1, 1), last);
            let f = src - vec2<f32>(p1);
            color = mix(
                mix(textureLoad(input_tex, p1, 0), textureLoad(input_tex, vec2<i32>(p2.x, p1.y), 0), f.x),
                mix(textureLoad(input_tex, vec2<i32>(p1.x, p2.y), 0), textureLoad(input_tex, p2, 0), f.x),
                f.y,
            );
        }

        textureStore(output_tex, coords, color);
    } else {
        textureStore(output_tex, coords, params.bg_color);
    }
}
