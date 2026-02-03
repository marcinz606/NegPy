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
        
        let color = mix(
            mix(c11, c21, fx),
            mix(c12, c22, fx),
            fy
        );
        
        textureStore(output_tex, coords, color);
    } else {
        textureStore(output_tex, coords, params.bg_color);
    }
}