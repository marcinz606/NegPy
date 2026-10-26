// Auto Masking meter, pass 2: the 30th percentile of the gradient histogram, as the masking
// lab.wgsl applies (word 0) and the auto share the panel shows (word 1). Mirrors sharpen_grain in
// negpy_core/src/lab.rs and auto_masking / effective_masking in logic.py.
struct LabUniforms {
    sharpen: f32,
    chroma_denoise: f32,
    saturation: f32,
    glow_amount: f32,
    halation_strength: f32,
    scale_factor: f32,
    sharpen_radius_px: f32,
    sharpen_masking: f32,
    sharpen_method: f32,
    skin_protection: f32,
    chroma_radius_px: f32,
    auto_masking: f32,
};

@group(0) @binding(0) var<storage, read> hist: array<u32, 1024>;
@group(0) @binding(1) var<storage, read_write> auto_mask: array<f32, 2>;
@group(0) @binding(2) var<uniform> params: LabUniforms;

const GRAIN_BINS = 1024u;
const GRAIN_BIN_SCALE = 256.0;
const AUTO_MASKING_GAIN = 0.27;
const NEUTRAL_MASKING = 0.1;

@compute @workgroup_size(1)
fn main() {
    var total = 0u;
    for (var i = 0u; i < GRAIN_BINS; i++) { total += hist[i]; }
    var share = NEUTRAL_MASKING;
    if (total > 0u) {
        let rank = total * 3u / 10u;
        var cum = 0u;
        var bin = 0u;
        for (; bin < GRAIN_BINS; bin++) {
            cum += hist[bin];
            if (cum > rank) { break; }
        }
        share = min(AUTO_MASKING_GAIN * ((f32(bin) + 0.5) / GRAIN_BIN_SCALE), 1.0);
    }
    auto_mask[0] = clamp(share + params.sharpen_masking - NEUTRAL_MASKING, 0.0, 1.0);
    auto_mask[1] = share;
}
