struct GeometryUniforms {
    rotation: i32,      // 0, 1, 2, 3 (multipled by 90 deg)
    fine_rotation: f32, // degrees
    flip_h: i32,        // 0 or 1
    flip_v: i32,        // 0 or 1
    pad: vec4<f32>,     // Padding
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: GeometryUniforms;

fn safeLoad(tex: texture_2d<f32>, coords: vec2<i32>, dims: vec2<i32>) -> vec4<f32> {
    let x = clamp(coords.x, 0, dims.x - 1);
    let y = clamp(coords.y, 0, dims.y - 1);
    return textureLoad(tex, vec2<i32>(x, y), 0);
}

fn textureSampleBilinear(tex: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    let udims = textureDimensions(tex);
    let dims = vec2<i32>(i32(udims.x), i32(udims.y));
    let fdims = vec2<f32>(f32(dims.x), f32(dims.y));
    
    let pixel = uv * fdims - 0.5;
    let c00 = floor(pixel);
    let c11 = c00 + 1.0;
    
    let t = pixel - c00;
    let w00 = (1.0 - t.x) * (1.0 - t.y);
    let w10 = t.x * (1.0 - t.y);
    let w01 = (1.0 - t.x) * t.y;
    let w11 = t.x * t.y;
    
    let i00 = vec2<i32>(i32(c00.x), i32(c00.y));
    let i10 = vec2<i32>(i32(c11.x), i32(c00.y));
    let i01 = vec2<i32>(i32(c00.x), i32(c11.y));
    let i11 = vec2<i32>(i32(c11.x), i32(c11.y));
    
    let v00 = safeLoad(tex, i00, dims);
    let v10 = safeLoad(tex, i10, dims);
    let v01 = safeLoad(tex, i01, dims);
    let v11 = safeLoad(tex, i11, dims);
    
    return v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dims = textureDimensions(output_tex);
    if (gid.x >= out_dims.x || gid.y >= out_dims.y) {
        return;
    }

    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let out_uv = vec2<f32>(f32(coords.x) + 0.5, f32(coords.y) + 0.5) / vec2<f32>(f32(out_dims.x), f32(out_dims.y));

    // 1. Center UV
    var uv = out_uv - 0.5;

    // 2. Inverse Fine Rotation (Must be first in sampling path)
    let out_aspect = f32(out_dims.x) / f32(out_dims.y);
    if (params.fine_rotation != 0.0) {
        let rad = radians(params.fine_rotation);
        let c = cos(rad);
        let s = sin(rad);
        let corrected_x = uv.x * out_aspect;
        let rx = corrected_x * c - uv.y * s;
        let ry = corrected_x * s + uv.y * c;
        uv.x = rx / out_aspect;
        uv.y = ry;
    }

    // 3. Inverse Flip
    if (params.flip_h == 1) { uv.x = -uv.x; }
    if (params.flip_v == 1) { uv.y = -uv.y; }

    // 4. Inverse 90-degree steps
    let k = params.rotation % 4;
    var temp_uv = uv;
    if (k == 1) {
        temp_uv.x = -uv.y;
        temp_uv.y = uv.x;
    } else if (k == 2) {
        temp_uv.x = -uv.x;
        temp_uv.y = -uv.y;
    } else if (k == 3) {
        temp_uv.x = uv.y;
        temp_uv.y = -uv.x;
    }
    uv = temp_uv;

    // 5. Un-center and sample
    uv = uv + 0.5;

    let color = textureSampleBilinear(input_tex, uv);
    textureStore(output_tex, coords, color);
}
