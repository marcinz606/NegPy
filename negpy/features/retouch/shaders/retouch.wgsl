struct RetouchUniforms {
    num_regions: u32,
    _pad: u32,
    global_offset: vec2<i32>,
};

// Capsule-chain heal region: polyline points [pt_start, pt_start+pt_count) and
// ordered boundary-loop samples [bnd_start, bnd_start+bnd_count) index into
// heal_pts (global pixel coords). src_off is the clone-source offset in pixels.
// gate: 1 = bright-only dust gate, 0 = unconditional clone (IR, dark-dust modes).
struct HealRegion {
    pt_start: u32,
    pt_count: u32,
    bnd_start: u32,
    bnd_count: u32,
    radius: f32,
    gate: f32,
    src_off: vec2<f32>,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: RetouchUniforms;
@group(0) @binding(3) var<storage, read> heal_regions: array<HealRegion>;
@group(0) @binding(4) var<storage, read> heal_pts: array<vec2<f32>>;

// Clone-sample dust guard (mirrors _sample_clean_jit): if the pixel's luma
// exceeds its 3x3 luma-median neighbour by CLONE_GUARD_LUMA it's a speck —
// return the median-luma pixel (a real pixel, grain preserved) instead, so
// dust in the source patch or on the boundary is never recloned.
// Ceiling: specks wider than ~2px fill the window and pass through; the
// source-offset scoring on the CPU avoids those upfront.
const CLONE_GUARD_LUMA: f32 = 0.06;

fn sample_clean(gp: vec2<f32>, idims: vec2<i32>) -> vec3<f32> {
    let gi = clamp(vec2<i32>(floor(gp)) - params.global_offset, vec2<i32>(0), idims - 1);
    var lums: array<f32, 9>;
    var cols: array<vec3<f32>, 9>;
    var n = 0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let sc = clamp(gi + vec2<i32>(dx, dy), vec2<i32>(0), idims - 1);
            let v = textureLoad(input_tex, sc, 0).rgb;
            cols[n] = v;
            lums[n] = dot(v, vec3<f32>(0.2126, 0.7152, 0.0722));
            n++;
        }
    }
    for (var i = 0; i <= 4; i++) {
        var mi = i;
        for (var j = i + 1; j < 9; j++) {
            if (lums[j] < lums[mi]) { mi = j; }
        }
        let tl = lums[i]; lums[i] = lums[mi]; lums[mi] = tl;
        let tc = cols[i]; cols[i] = cols[mi]; cols[mi] = tc;
    }
    let v = textureLoad(input_tex, gi, 0).rgb;
    if (dot(v, vec3<f32>(0.2126, 0.7152, 0.0722)) - lums[4] > CLONE_GUARD_LUMA) {
        return cols[4];
    }
    return v;
}

// 5x5 variant for the directly-cloned source sample — catches specks up to
// ~4px that slip through the 3x3 window. Mirrors _sample_clean5_jit.
fn sample_clean5(gp: vec2<f32>, idims: vec2<i32>) -> vec3<f32> {
    let gi = clamp(vec2<i32>(floor(gp)) - params.global_offset, vec2<i32>(0), idims - 1);
    var lums: array<f32, 25>;
    var cols: array<vec3<f32>, 25>;
    var n = 0;
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let sc = clamp(gi + vec2<i32>(dx, dy), vec2<i32>(0), idims - 1);
            let v = textureLoad(input_tex, sc, 0).rgb;
            cols[n] = v;
            lums[n] = dot(v, vec3<f32>(0.2126, 0.7152, 0.0722));
            n++;
        }
    }
    for (var i = 0; i <= 12; i++) {
        var mi = i;
        for (var j = i + 1; j < 25; j++) {
            if (lums[j] < lums[mi]) { mi = j; }
        }
        let tl = lums[i]; lums[i] = lums[mi]; lums[mi] = tl;
        let tc = cols[i]; cols[i] = cols[mi]; cols[mi] = tc;
    }
    let v = textureLoad(input_tex, gi, 0).rgb;
    if (dot(v, vec3<f32>(0.2126, 0.7152, 0.0722)) - lums[12] > CLONE_GUARD_LUMA) {
        return cols[12];
    }
    return v;
}

fn dist_to_seg(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let ab2 = dot(ab, ab);
    var t = 0.0;
    if (ab2 > 1e-12) { t = clamp(dot(p - a, ab) / ab2, 0.0, 1.0); }
    return distance(p, a + t * ab);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let idims = vec2<i32>(dims);
    let global_coords = vec2<f32>(f32(coords.x + params.global_offset.x) + 0.5,
                                  f32(coords.y + params.global_offset.y) + 0.5);

    let original = textureLoad(input_tex, coords, 0).rgb;
    var res = original;

    // Heal regions (manual + synthesized auto/IR; detection is CPU-side):
    // mean-value-coordinates membrane clone (Georgiev healing brush).
    // out = src_patch + MVC boundary difference — copied pixels carry real grain.
    for (var ri = 0u; ri < params.num_regions; ri++) {
        let reg = heal_regions[ri];
        if (reg.bnd_count < 3u || reg.bnd_count > 64u || reg.pt_count < 1u) { continue; }

        let p = global_coords;
        var d = 1e18;
        if (reg.pt_count == 1u) {
            d = distance(p, heal_pts[reg.pt_start]);
        } else {
            for (var s = 0u; s + 1u < reg.pt_count; s++) {
                d = min(d, dist_to_seg(p, heal_pts[reg.pt_start + s], heal_pts[reg.pt_start + s + 1u]));
            }
        }
        if (d >= reg.radius) { continue; }

        let n = reg.bnd_count;
        var vxs: array<f32, 64>; var vys: array<f32, 64>; var vls: array<f32, 64>;
        var diffs: array<vec3<f32>, 64>;
        var on_sample = -1;
        for (var i = 0u; i < n; i++) {
            let b = heal_pts[reg.bnd_start + i];
            diffs[i] = sample_clean(b, idims) - sample_clean(b + reg.src_off, idims);
            let v = b - p;
            let l = length(v);
            vxs[i] = v.x; vys[i] = v.y; vls[i] = l;
            if (l < 1e-4) { on_sample = i32(i); }
        }

        var mem = vec3<f32>(0.0);
        if (on_sample >= 0) {
            mem = diffs[on_sample];
        } else {
            var tans: array<f32, 64>;
            for (var i = 0u; i < n; i++) {
                var j = i + 1u;
                if (j == n) { j = 0u; }
                var cr = vxs[i] * vys[j] - vys[i] * vxs[j];
                if (abs(cr) < 1e-9) { cr = 1e-9; }
                tans[i] = (vls[i] * vls[j] - (vxs[i] * vxs[j] + vys[i] * vys[j])) / cr;
            }
            var wsum = 0.0;
            for (var i = 0u; i < n; i++) {
                var prev = n - 1u;
                if (i > 0u) { prev = i - 1u; }
                let wi = (tans[prev] + tans[i]) / vls[i];
                wsum += wi;
                mem += wi * diffs[i];
            }
            if (abs(wsum) < 1e-12) { continue; }
            mem /= wsum;
        }

        let healed = sample_clean5(p + reg.src_off, idims) + mem;
        // Rim feather; mirrors logic.py _RIM_FEATHER_FRAC (+ _RIM_FEATHER_UNGATED for
        // gate=0). Ungated synthesized clones get a softer 0.4·radius edge.
        let fth = max(mix(0.4, 0.25, reg.gate) * reg.radius, 1.5);
        let t = clamp((d - (reg.radius - fth)) / fth, 0.0, 1.0);
        var alpha = 1.0 - t * t * (3.0 - 2.0 * t);
        // Dust gate: heal only pixels brighter than the membrane-predicted
        // clean value; gate=0 regions clone unconditionally.
        let g = smoothstep(0.04, 0.12, dot(res, vec3<f32>(0.2126, 0.7152, 0.0722)) - dot(healed, vec3<f32>(0.2126, 0.7152, 0.0722)));
        alpha *= mix(1.0, g, reg.gate);
        res = mix(res, healed, alpha);
    }

    textureStore(output_tex, coords, vec4<f32>(res, 1.0));
}
