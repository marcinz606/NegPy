struct LayoutUniforms {
    bg_color: vec4<f32>,
    offset: vec2<i32>,
    content_dims: vec2<i32>, // Size of image on paper (px)
    source_dims: vec2<i32>,  // Size of incoming texture (px)
    scale: f32,              // scale factor: paper_content / source
    keyline_px: i32,         // keyline thickness (px), 0 = off
    keyline_gap_px: i32,     // gap between image edge and keyline (px)
    corner_style: i32,       // 0 = square, 1 = rounded, 2 = deckle
    corner_param_px: f32,    // radius (rounded) or scallop amplitude (deckle)
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: LayoutUniforms;

const TAU: f32 = 6.28318530718;

// 0..1 scallop along a normalized edge coordinate; mirrored in print.py.
fn deckle_profile(s: f32, phase: f32) -> f32 {
    return 0.5 + 0.3 * sin(TAU * 5.0 * s + 0.7 + phase) + 0.2 * sin(TAU * 13.0 * s + 2.9 + phase);
}

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

        // Map output pixel to source coordinates
        // Using bilinear interpolation
        let src_x = local_x / params.scale;
        let src_y = local_y / params.scale;

        let x1 = i32(floor(src_x));
        let y1 = i32(floor(src_y));
        let x2 = min(x1 + 1, params.source_dims.x - 1);
        let y2 = min(y1 + 1, params.source_dims.y - 1);

        let fx = src_x - f32(x1);
        let fy = src_y - f32(y1);

        let c11 = textureLoad(input_tex, vec2<i32>(x1, y1), 0);
        let c21 = textureLoad(input_tex, vec2<i32>(x2, y1), 0);
        let c12 = textureLoad(input_tex, vec2<i32>(x1, y2), 0);
        let c22 = textureLoad(input_tex, vec2<i32>(x2, y2), 0);

        var color = mix(
            mix(c11, c21, fx),
            mix(c12, c22, fx),
            fy
        );

        // Corner styling: alpha toward the mat colour, matching print.py.
        var alpha = 1.0;
        let cw = f32(params.content_dims.x);
        let ch = f32(params.content_dims.y);
        if (params.corner_style == 1) {
            let r = params.corner_param_px;
            let ax = max(r - min(local_x, cw - 1.0 - local_x), 0.0);
            let ay = max(r - min(local_y, ch - 1.0 - local_y), 0.0);
            if (ax > 0.0 && ay > 0.0) {
                alpha = clamp(r - sqrt(ax * ax + ay * ay) + 0.5, 0.0, 1.0);
            }
        } else if (params.corner_style == 2) {
            let amp = params.corner_param_px;
            let a_t = clamp(local_y - amp * deckle_profile(local_x / cw, 0.0) + 0.5, 0.0, 1.0);
            let a_b = clamp((ch - 1.0 - local_y) - amp * deckle_profile(local_x / cw, 3.4) + 0.5, 0.0, 1.0);
            let a_l = clamp(local_x - amp * deckle_profile(local_y / ch, 5.1) + 0.5, 0.0, 1.0);
            let a_r = clamp((cw - 1.0 - local_x) - amp * deckle_profile(local_y / ch, 1.7) + 0.5, 0.0, 1.0);
            alpha = a_t * a_b * a_l * a_r;
        }
        color = mix(params.bg_color, color, alpha);

        textureStore(output_tex, coords, color);
    } else {
        // Chebyshev distance outside the content rect; keyline is a
        // rectangular ring gap..gap+width away from the image edge.
        let lxi = coords.x - params.offset.x;
        let lyi = coords.y - params.offset.y;
        let d = max(
            max(-lxi, lxi - (params.content_dims.x - 1)),
            max(-lyi, lyi - (params.content_dims.y - 1)),
        );
        if (params.keyline_px > 0 && d > params.keyline_gap_px && d <= params.keyline_gap_px + params.keyline_px) {
            textureStore(output_tex, coords, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        } else {
            textureStore(output_tex, coords, params.bg_color);
        }
    }
}
