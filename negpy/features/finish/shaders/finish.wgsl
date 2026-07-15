struct FinishUniforms {
    vignette_stops: f32,
    vignette_size: f32,
    vignette_roundness: f32,
    full_crop_w: f32,
    full_crop_h: f32,
    tile_off_x: f32,
    tile_off_y: f32,
    carrier_width_px: f32,   // 0 = off
    carrier_rough: f32,
    diffusion_amount: f32,   // 0 = off
    scale_factor: f32,       // render scale; diffusion radius tracks it like the CPU path
    _pad0: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: FinishUniforms;
// (4, 512) edge jitter profiles from carrier_profiles() — top, bottom, left, right.
@group(0) @binding(3) var<storage, read> carrier_prof: array<f32>;

const CARRIER_SAMPLES: i32 = 512;
const CARRIER_JITTER: f32 = 0.6;
// Penumbra fraction of the rebate width; mirrors CARRIER_SOFT in logic.py.
const CARRIER_SOFT: f32 = 0.35;
// Diffusion blur radius mirrors DIFFUSION_RADIUS in logic.py (base_r at scale 1).
const DIFFUSION_RADIUS: f32 = 20.0;

// 64-tap Fibonacci spiral — uniform area coverage, smooth Gaussian approximation.
// Points lie in the unit disk; scale by the desired pixel radius when sampling.
// Mirrors FIBONACCI_64 in lab.wgsl.
const FIBONACCI_64 = array<vec2<f32>, 64>(
    vec2<f32>(0.088388, 0.000000),
    vec2<f32>(-0.112886, 0.103413),
    vec2<f32>(0.017279, -0.196886),
    vec2<f32>(0.142286, 0.185586),
    vec2<f32>(-0.261112, -0.046187),
    vec2<f32>(0.247348, -0.157342),
    vec2<f32>(-0.082733, 0.307763),
    vec2<f32>(-0.157781, -0.303797),
    vec2<f32>(0.342321, 0.125015),
    vec2<f32>(-0.356128, 0.147004),
    vec2<f32>(0.171677, -0.366864),
    vec2<f32>(0.126865, 0.404466),
    vec2<f32>(-0.382373, -0.221593),
    vec2<f32>(0.448567, -0.098616),
    vec2<f32>(-0.273753, 0.389386),
    vec2<f32>(-0.063243, -0.488045),
    vec2<f32>(0.388252, 0.327220),
    vec2<f32>(-0.522466, 0.021606),
    vec2<f32>(0.381099, -0.379244),
    vec2<f32>(-0.025497, 0.551396),
    vec2<f32>(-0.362617, -0.434536),
    vec2<f32>(0.574425, 0.077288),
    vec2<f32>(-0.486709, 0.338640),
    vec2<f32>(0.132997, -0.591185),
    vec2<f32>(0.307615, 0.536829),
    vec2<f32>(-0.601358, -0.191850),
    vec2<f32>(0.584143, -0.269889),
    vec2<f32>(-0.253065, 0.604686),
    vec2<f32>(-0.225855, -0.627935),
    vec2<f32>(0.600976, 0.315856),
    vec2<f32>(-0.667533, 0.175960),
    vec2<f32>(0.379431, -0.590102),
    vec2<f32>(0.120699, 0.702313),
    vec2<f32>(-0.572008, -0.442995),
    vec2<f32>(0.731702, -0.060620),
    vec2<f32>(-0.505760, 0.546712),
    vec2<f32>(0.003684, -0.755181),
    vec2<f32>(0.514305, 0.566946),
    vec2<f32>(-0.772295, -0.071576),
    vec2<f32>(0.625787, -0.474950),
    vec2<f32>(-0.142381, 0.782650),
    vec2<f32>(-0.428884, -0.681539),
    vec2<f32>(0.785920, 0.215388),
    vec2<f32>(-0.733486, 0.376413),
    vec2<f32>(0.289862, -0.781852),
    vec2<f32>(0.317911, 0.780942),
    vec2<f32>(-0.770264, -0.365042),
    vec2<f32>(0.823263, -0.253821),
    vec2<f32>(-0.440157, 0.751049),
    vec2<f32>(-0.184643, -0.859851),
    vec2<f32>(0.724177, 0.514422),
    vec2<f32>(-0.890157, 0.110939),
    vec2<f32>(0.587054, -0.689695),
    vec2<f32>(0.033320, 0.913689),
    vec2<f32>(-0.647727, -0.657276),
    vec2<f32>(0.930014, 0.047552),
    vec2<f32>(-0.724323, 0.598472),
    vec2<f32>(0.130975, -0.938767),
    vec2<f32>(0.542205, 0.787449),
    vec2<f32>(-0.939649, -0.216211),
    vec2<f32>(0.845937, -0.479274),
    vec2<f32>(-0.302492, 0.932436),
    vec2<f32>(-0.410097, -0.899101),
    vec2<f32>(0.916976, 0.389028)
);
// Sum of exp(-2*r²) over all 64 Fibonacci samples — used to normalize the
// accumulator the same way a Gaussian convolution kernel is normalized (sum=1).
const BLOOM_GAUSS_SUM = 27.668145;

fn carrier_edge_width(edge: i32, s: f32) -> f32 {
    let idx = min(i32(s * f32(CARRIER_SAMPLES)), CARRIER_SAMPLES - 1);
    let jitter = carrier_prof[edge * CARRIER_SAMPLES + idx];
    return params.carrier_width_px * (1.0 + CARRIER_JITTER * params.carrier_rough * jitter);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    var color = textureLoad(input_tex, coords, 0).rgb;

    // Enlarger diffusion first (raw taps + raw center), before the per-pixel
    // burn/carrier multiplies — mirrors the CPU stage order in processor.py.
    if (params.diffusion_amount > 0.0) {
        // Radius tracks the render scale exactly like apply_diffusion's base_r.
        let radius = max(3.0, floor(DIFFUSION_RADIUS * params.scale_factor));
        var accum = vec3<f32>(0.0);
        for (var tap = 0; tap < 64; tap++) {
            let offset = FIBONACCI_64[tap];
            let s_coord = clamp(coords + vec2<i32>(offset * radius), vec2<i32>(0), vec2<i32>(dims) - 1);
            let samp = textureLoad(input_tex, s_coord, 0).rgb;
            let r = length(offset);
            let w = exp(-r * r * 2.0);
            accum += samp * w;
        }
        let blurred = accum / BLOOM_GAUSS_SUM;
        // Darken-only: dense negative areas bloom, highlights never lift.
        color = color + params.diffusion_amount * (min(blurred, color) - color);
    }

    let full = vec2<f32>(params.full_crop_w, params.full_crop_h);
    let center = full * 0.5;
    let max_dist = length(center);
    let px = vec2<f32>(f32(coords.x) + params.tile_off_x, f32(coords.y) + params.tile_off_y);
    let off = abs(px - center);
    let d_radial = length(off) / max(max_dist, 1.0);
    let d_rect = max(off.x / max(center.x, 1.0), off.y / max(center.y, 1.0));
    let d = mix(d_radial, d_rect, params.vignette_roundness);

    // Remap: size=0 → vignette at edges, size=1 → covers entire image
    let midpoint = 1.0 - params.vignette_size;
    let t = clamp((d - midpoint) / max(1e-6, 1.0 - midpoint), 0.0, 1.0);

    // Smooth cosine falloff
    let factor = 0.5 * (1.0 - cos(t * 3.14159265));

    if (params.vignette_stops != 0.0) {
        color = color * exp2(-params.vignette_stops * factor);
    }

    // Filed-carrier rebate: multiply toward black inside the jittered frame,
    // mirroring apply_carrier() in logic.py.
    if (params.carrier_width_px > 0.0) {
        let soft = max(1.0, params.carrier_width_px * CARRIER_SOFT);
        let sx = (px.x + 0.5) / full.x;
        let sy = (px.y + 0.5) / full.y;
        let a_t = clamp((px.y - carrier_edge_width(0, sx)) / soft + 0.5, 0.0, 1.0);
        let a_b = clamp(((full.y - 1.0 - px.y) - carrier_edge_width(1, sx)) / soft + 0.5, 0.0, 1.0);
        let a_l = clamp((px.x - carrier_edge_width(2, sy)) / soft + 0.5, 0.0, 1.0);
        let a_r = clamp(((full.x - 1.0 - px.x) - carrier_edge_width(3, sy)) / soft + 0.5, 0.0, 1.0);
        color = color * (a_t * a_b * a_l * a_r);
    }

    textureStore(output_tex, coords, vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
