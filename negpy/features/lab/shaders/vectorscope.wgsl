@group(0) @binding(0) var input_tex: texture_2d<f32>;
// 180 hue bins x 64 log-chroma bins of CIELAB a*b*, indexed hue * 64 + chroma. Mirrors the
// SCOPE_* constants and vectorscope_histogram() in features/exposure/analysis.py.
@group(0) @binding(1) var<storage, read_write> bins: array<atomic<u32>, 11520>;

const HUE_BINS = 180.0;
const CHROMA_BINS = 64.0;
const MAX_CHROMA = 60.0;
const LOG_KNEE = 4.0;
const PI = 3.14159265358979;

// Linear working RGB -> CIELAB (D65). Mirrors rgb_to_lab in lab.wgsl.
fn rgb_to_lab(rgb: vec3<f32>) -> vec3<f32> {
    var x = rgb.r * 0.5767309 + rgb.g * 0.1855540 + rgb.b * 0.1881852;
    var y = rgb.r * 0.2973769 + rgb.g * 0.6273491 + rgb.b * 0.0752741;
    var z = rgb.r * 0.0270343 + rgb.g * 0.0706872 + rgb.b * 0.9911085;

    x = x / 0.95047;
    z = z / 1.08883;

    if (x > 0.008856) { x = pow(x, 1.0/3.0); } else { x = (7.787 * x) + (16.0 / 116.0); }
    if (y > 0.008856) { y = pow(y, 1.0/3.0); } else { y = (7.787 * y) + (16.0 / 116.0); }
    if (z > 0.008856) { z = pow(z, 1.0/3.0); } else { z = (7.787 * z) + (16.0 / 116.0); }

    return vec3<f32>(116.0 * y - 16.0, 500.0 * (x - y), 200.0 * (y - z));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    // Clamped like the encoded output the CPU bins, so out-of-range light lands on the same bin.
    let lin = clamp(textureLoad(input_tex, vec2<i32>(i32(gid.x), i32(gid.y)), 0).rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let lab = rgb_to_lab(lin);
    let chroma = length(lab.yz);
    let radius = log(1.0 + chroma / LOG_KNEE) / log(1.0 + MAX_CHROMA / LOG_KNEE);
    let c_bin = u32(clamp(floor(radius * CHROMA_BINS), 0.0, CHROMA_BINS - 1.0));
    let h_bin = min(u32(floor((atan2(lab.z, lab.y) + PI) / (2.0 * PI) * HUE_BINS)), u32(HUE_BINS) - 1u);

    atomicAdd(&bins[h_bin * u32(CHROMA_BINS) + c_bin], 1u);
}
