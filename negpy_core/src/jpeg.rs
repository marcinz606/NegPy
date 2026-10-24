//! Baseline 4:4:4 JPEG of uint8 RGB or gray, as Pillow writes it through libjpeg-turbo with
//! `subsampling=0, optimize=True`: the quality-scaled Annex K tables, fixed-point YCbCr, the
//! islow DCT and optimal Huffman tables. Restart markers split the scan into strips that encode
//! in parallel; they change the bytes but not one decoded pixel. Without them the bytes are
//! Pillow's.

// The tables index several arrays by symbol or by code length in step.
#![allow(clippy::needless_range_loop)]

use multiversion::multiversion;
use rayon::prelude::*;

const NATURAL: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

const STD_LUMA: [u32; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

const STD_CHROMA: [u32; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

pub struct Options<'a> {
    pub quality: i32,
    pub icc: Option<&'a [u8]>,
    pub dpi: Option<(u16, u16)>,
    /// MCU rows per restart interval; 0 for one interval, None to pick one for the cores.
    pub restart_rows: Option<usize>,
}

/// `jpeg_set_quality(q, force_baseline=TRUE)`'s table, in natural order.
fn quant_table(basic: &[u32; 64], quality: i32) -> [u32; 64] {
    let q = quality.clamp(1, 100);
    let scale = if q < 50 { 5000 / q } else { 200 - 2 * q } as i64;
    basic.map(|b| ((b as i64 * scale + 50) / 100).clamp(1, 255) as u32)
}

/// jccolor.c's fixed-point YCbCr weights. Its tables are these products, so a row computes the
/// same sums in vector lanes.
const fn fix(x: f64) -> i32 {
    (x * 65536.0 + 0.5) as i32
}
const HALF: i32 = 1 << 15;
const CBCR: i32 = (128 << 16) + HALF - 1;
const Y_W: [i32; 3] = [fix(0.29900), fix(0.58700), fix(0.11400)];
const CB_W: [i32; 3] = [-fix(0.16874), -fix(0.33126), fix(0.5)];
const CR_W: [i32; 3] = [fix(0.5), -fix(0.41869), -fix(0.08131)];

/// One RGB row to Y, Cb and Cr rows of the padded width, the last pixel repeated into the pad.
#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2",
    "aarch64+neon"
))]
fn ycc_row(src: &[u8], [y, cb, cr]: [&mut [u8]; 3]) {
    let w = src.len() / 3;
    let px = src.as_chunks::<3>().0;
    let sum = |wt: [i32; 3], p: &[u8; 3], off: i32| {
        ((wt[0] * p[0] as i32 + wt[1] * p[1] as i32 + wt[2] * p[2] as i32 + off) >> 16) as u8
    };
    for (((y, cb), cr), p) in y.iter_mut().zip(cb.iter_mut()).zip(cr.iter_mut()).zip(px) {
        *y = sum(Y_W, p, HALF);
        *cb = sum(CB_W, p, CBCR);
        *cr = sum(CR_W, p, CBCR);
    }
    for plane in [y, cb, cr] {
        let last = plane[w - 1];
        plane[w..].fill(last);
    }
}

/// jfdctint.c, `jpeg_fdct_islow`, on a block given transposed (`t[col][row]`), so each pass
/// runs the eight rows or columns side by side; the result is in natural order, scaled by 8.
#[inline(always)]
fn fdct_islow(t: &mut [[i32; 8]; 8]) {
    for j in 0..8 {
        fdct_pass::<true>(t, j);
    }
    let mut u = [[0i32; 8]; 8];
    for (r, row) in u.iter_mut().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            *v = t[c][r];
        }
    }
    for j in 0..8 {
        fdct_pass::<false>(&mut u, j);
    }
    *t = u;
}

/// One lane of the islow DCT: the row pass (`FIRST`) or the column pass.
#[inline(always)]
fn fdct_pass<const FIRST: bool>(d: &mut [[i32; 8]; 8], j: usize) {
    const CB: i32 = 13;
    const P1: i32 = 2;
    let descale = |x: i32, n: i32| (x + (1 << (n - 1))) >> n;
    let odd = if FIRST { CB - P1 } else { CB + P1 };
    let (tmp0, tmp7) = (d[0][j] + d[7][j], d[0][j] - d[7][j]);
    let (tmp1, tmp6) = (d[1][j] + d[6][j], d[1][j] - d[6][j]);
    let (tmp2, tmp5) = (d[2][j] + d[5][j], d[2][j] - d[5][j]);
    let (tmp3, tmp4) = (d[3][j] + d[4][j], d[3][j] - d[4][j]);
    let (tmp10, tmp13) = (tmp0 + tmp3, tmp0 - tmp3);
    let (tmp11, tmp12) = (tmp1 + tmp2, tmp1 - tmp2);
    if FIRST {
        d[0][j] = (tmp10 + tmp11) << P1;
        d[4][j] = (tmp10 - tmp11) << P1;
    } else {
        d[0][j] = descale(tmp10 + tmp11, P1);
        d[4][j] = descale(tmp10 - tmp11, P1);
    }
    let z1 = (tmp12 + tmp13) * 4433;
    d[2][j] = descale(z1 + tmp13 * 6270, odd);
    d[6][j] = descale(z1 + tmp12 * -15137, odd);
    let (z1, z2, z3, z4) = (tmp4 + tmp7, tmp5 + tmp6, tmp4 + tmp6, tmp5 + tmp7);
    let z5 = (z3 + z4) * 9633;
    let (tmp4, tmp5, tmp6, tmp7) = (tmp4 * 2446, tmp5 * 16819, tmp6 * 25172, tmp7 * 12299);
    let (z1, z2) = (z1 * -7373, z2 * -20995);
    let (z3, z4) = (z3 * -16069 + z5, z4 * -3196 + z5);
    d[7][j] = descale(tmp4 + z1 + z3, odd);
    d[5][j] = descale(tmp5 + z2 + z4, odd);
    d[3][j] = descale(tmp6 + z2 + z3, odd);
    d[1][j] = descale(tmp7 + z1 + z4, odd);
}

/// libjpeg's rounded quantization, `(|v| + d / 2) / d` with the sign put back, for the islow
/// divisor `d = 8 q`. The division is a multiply by ceil(2^32 / d): exact while |v| + d / 2
/// stays under 2^16 and d under 2^12, which every 8-bit DCT output and baseline table keeps.
struct Divisors {
    half: [u32; 64],
    recip: [u64; 64],
}

impl Divisors {
    fn new(q: &[u32; 64]) -> Self {
        Divisors {
            half: q.map(|q| (q << 3) >> 1),
            recip: q.map(|q| (1u64 << 32).div_ceil((q << 3) as u64)),
        }
    }

    #[inline(always)]
    fn quantize(&self, block: &mut [[i32; 8]; 8]) {
        for ((v, &h), &r) in block
            .as_flattened_mut()
            .iter_mut()
            .zip(&self.half)
            .zip(&self.recip)
        {
            let m = (((v.unsigned_abs() + h) as u64 * r) >> 32) as i32;
            *v = if *v < 0 { -m } else { m };
        }
    }
}

/// The quantized blocks of one MCU row, in scan order (block column, then component), from
/// the component planes of its eight padded rows.
#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2",
    "aarch64+neon"
))]
fn dct_row(planes: &[Vec<u8>], pw: usize, divisors: &[Divisors; 2], out: &mut [[[i32; 8]; 8]]) {
    let cn = planes.len();
    for (i, block) in out.iter_mut().enumerate() {
        let (bx, ci) = (i / cn, i % cn);
        let plane = &planes[ci];
        for (r, row) in plane.chunks_exact(pw).enumerate() {
            for (c, &v) in row[bx * 8..bx * 8 + 8].iter().enumerate() {
                block[c][r] = v as i32 - 128;
            }
        }
        fdct_islow(block);
        divisors[ci.min(1)].quantize(block);
    }
}

/// The quantized coefficients of every block of MCU rows `rows`, handed to `f` in scan order
/// (component index, block in natural order). Rows and columns past the image repeat its edge.
fn for_each_block(
    px: &[u8],
    (h, w, cn): (usize, usize, usize),
    rows: std::ops::Range<usize>,
    divisors: &[Divisors; 2],
    mut f: impl FnMut(usize, &[i32]),
) {
    let pw = w.div_ceil(8) * 8;
    let mut planes = vec![vec![0u8; 8 * pw]; cn];
    let mut blocks = vec![[[0i32; 8]; 8]; pw / 8 * cn];
    for mcu_row in rows {
        for r in 0..8 {
            let y = (mcu_row * 8 + r).min(h - 1);
            let src = &px[y * w * cn..(y + 1) * w * cn];
            if cn == 3 {
                let [a, b, c] = &mut planes[..] else {
                    unreachable!()
                };
                let span = r * pw..(r + 1) * pw;
                ycc_row(
                    src,
                    [&mut a[span.clone()], &mut b[span.clone()], &mut c[span]],
                );
            } else {
                let dst = &mut planes[0][r * pw..(r + 1) * pw];
                dst[..w].copy_from_slice(src);
                let last = src[w - 1];
                dst[w..].fill(last);
            }
        }
        dct_row(&planes, pw, divisors, &mut blocks);
        for (i, b) in blocks.iter().enumerate() {
            f(i % cn, b.as_flattened());
        }
    }
}

fn nbits(v: i32) -> u32 {
    32 - v.unsigned_abs().leading_zeros()
}

/// Symbol counts per table: [dc0, ac0, dc1, ac1].
type Counts = [[i64; 257]; 4];

fn count_block(counts: &mut Counts, t: usize, block: &[i32], last_dc: &mut i32) {
    counts[2 * t][nbits(block[0] - *last_dc) as usize] += 1;
    *last_dc = block[0];
    let mut run = 0;
    for &n in &NATURAL[1..] {
        let v = block[n];
        if v == 0 {
            run += 1;
            continue;
        }
        while run > 15 {
            counts[2 * t + 1][0xF0] += 1;
            run -= 16;
        }
        counts[2 * t + 1][(run << 4) + nbits(v) as usize] += 1;
        run = 0;
    }
    if run > 0 {
        counts[2 * t + 1][0] += 1;
    }
}

/// jchuff.c's `jpeg_gen_optimal_table`: (bits[1..=16], huffval).
fn optimal_table(freq: &[i64; 257]) -> ([u8; 17], Vec<u8>) {
    const MAX_CLEN: usize = 32;
    let mut freq = *freq;
    let mut bits = [0u8; MAX_CLEN + 1];
    let mut codesize = [0usize; 257];
    let mut others = [-1i64; 257];
    freq[256] = 1;
    loop {
        let mut c1 = -1i64;
        let mut v = 1_000_000_000i64;
        for (i, &f) in freq.iter().enumerate() {
            if f != 0 && f <= v {
                v = f;
                c1 = i as i64;
            }
        }
        let mut c2 = -1i64;
        v = 1_000_000_000;
        for (i, &f) in freq.iter().enumerate() {
            if f != 0 && f <= v && i as i64 != c1 {
                v = f;
                c2 = i as i64;
            }
        }
        if c2 < 0 {
            break;
        }
        let (mut a, mut b) = (c1 as usize, c2 as usize);
        freq[a] += freq[b];
        freq[b] = 0;
        codesize[a] += 1;
        while others[a] >= 0 {
            a = others[a] as usize;
            codesize[a] += 1;
        }
        others[a] = b as i64;
        codesize[b] += 1;
        while others[b] >= 0 {
            b = others[b] as usize;
            codesize[b] += 1;
        }
    }
    for &s in &codesize {
        if s != 0 {
            bits[s] += 1;
        }
    }
    let mut i = MAX_CLEN;
    while i > 16 {
        while bits[i] > 0 {
            let mut j = i - 2;
            while bits[j] == 0 {
                j -= 1;
            }
            bits[i] -= 2;
            bits[i - 1] += 1;
            bits[j + 1] += 2;
            bits[j] -= 1;
        }
        i -= 1;
    }
    while bits[i] == 0 {
        i -= 1;
    }
    bits[i] -= 1;
    let mut huffval = Vec::new();
    for len in 1..=MAX_CLEN {
        for (sym, &s) in codesize.iter().enumerate().take(256) {
            if s == len {
                huffval.push(sym as u8);
            }
        }
    }
    let mut out = [0u8; 17];
    out.copy_from_slice(&bits[..17]);
    (out, huffval)
}

/// Canonical codes (code, length) by symbol, as `jpeg_make_c_derived_tbl` assigns them.
fn derived(bits: &[u8; 17], huffval: &[u8]) -> [(u32, u32); 256] {
    let mut table = [(0u32, 0u32); 256];
    let (mut code, mut k) = (0u32, 0usize);
    for len in 1..=16 {
        for _ in 0..bits[len] {
            table[huffval[k] as usize] = (code, len as u32);
            code += 1;
            k += 1;
        }
        code <<= 1;
    }
    table
}

struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    n: u32,
}

impl BitWriter {
    fn put(&mut self, code: u32, size: u32) {
        self.acc = (self.acc << size) | code as u64;
        self.n += size;
        while self.n >= 8 {
            self.n -= 8;
            let b = (self.acc >> self.n) as u8;
            self.out.push(b);
            if b == 0xFF {
                self.out.push(0);
            }
        }
        self.acc &= (1u64 << self.n) - 1;
    }

    /// Pads the last byte with ones, as libjpeg's `flush_bits` does.
    fn flush(&mut self) {
        if self.n > 0 {
            self.put((1 << (8 - self.n)) - 1, 8 - self.n);
        }
    }
}

type Codes = [[(u32, u32); 256]; 4];

fn encode_block(bw: &mut BitWriter, codes: &Codes, t: usize, block: &[i32], last_dc: &mut i32) {
    let (dc, ac) = (&codes[2 * t], &codes[2 * t + 1]);
    let emit = |bw: &mut BitWriter, v: i32, n: u32| {
        if n > 0 {
            let bits = if v < 0 { v - 1 } else { v };
            bw.put(bits as u32 & ((1 << n) - 1), n);
        }
    };
    let diff = block[0] - *last_dc;
    *last_dc = block[0];
    let n = nbits(diff);
    bw.put(dc[n as usize].0, dc[n as usize].1);
    emit(bw, diff, n);
    let mut run = 0;
    for &k in &NATURAL[1..] {
        let v = block[k];
        if v == 0 {
            run += 1;
            continue;
        }
        while run > 15 {
            bw.put(ac[0xF0].0, ac[0xF0].1);
            run -= 16;
        }
        let n = nbits(v);
        let s = (run << 4) + n as usize;
        bw.put(ac[s].0, ac[s].1);
        emit(bw, v, n);
        run = 0;
    }
    if run > 0 {
        bw.put(ac[0].0, ac[0].1);
    }
}

fn marker(out: &mut Vec<u8>, m: u8, body: &[u8]) {
    out.extend_from_slice(&[0xFF, m]);
    out.extend_from_slice(&((body.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(body);
}

/// The JPEG of an HxW (`cn` 1) or HxWx3 uint8 image. None for an empty image or one too wide
/// for a restart interval.
pub fn encode(px: &[u8], (h, w, cn): (usize, usize, usize), o: &Options) -> Option<Vec<u8>> {
    if h == 0 || w == 0 || h > 65535 || w > 65535 || !(cn == 1 || cn == 3) || px.len() != h * w * cn
    {
        return None;
    }
    let quant = [
        quant_table(&STD_LUMA, o.quality),
        quant_table(&STD_CHROMA, o.quality),
    ];
    let divisors = [Divisors::new(&quant[0]), Divisors::new(&quant[1])];
    let (mcu_rows, mcu_cols) = (h.div_ceil(8), w.div_ceil(8));
    let rows = match o.restart_rows {
        Some(0) => mcu_rows,
        Some(r) => r.max(1),
        None => mcu_rows.div_ceil(4 * rayon::current_num_threads()).max(1),
    };
    // The restart interval counts MCUs in 16 bits.
    let rows = if rows < mcu_rows {
        rows.min(65535 / mcu_cols).max(1)
    } else {
        rows
    };
    let restart = rows < mcu_rows;
    let strips: Vec<std::ops::Range<usize>> = (0..mcu_rows)
        .step_by(rows)
        .map(|s| s..(s + rows).min(mcu_rows))
        .collect();
    let table = |ci: usize| ci.min(1);

    let counts = strips
        .par_iter()
        .map(|r| {
            let mut counts: Counts = [[0; 257]; 4];
            let mut last = [0i32; 3];
            for_each_block(px, (h, w, cn), r.clone(), &divisors, |ci, b| {
                count_block(&mut counts, table(ci), b, &mut last[ci]);
            });
            counts
        })
        .reduce(
            || [[0; 257]; 4],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(&b) {
                    x.iter_mut().zip(y).for_each(|(p, q)| *p += q);
                }
                a
            },
        );
    let tables: Vec<([u8; 17], Vec<u8>)> = (0..if cn == 3 { 4 } else { 2 })
        .map(|t| optimal_table(&counts[t]))
        .collect();
    let mut codes: Codes = [[(0, 0); 256]; 4];
    for (c, (bits, vals)) in codes.iter_mut().zip(&tables) {
        *c = derived(bits, vals);
    }

    let scans: Vec<Vec<u8>> = strips
        .par_iter()
        .map(|r| {
            let mut bw = BitWriter {
                out: Vec::with_capacity(r.len() * mcu_cols * cn * 32),
                acc: 0,
                n: 0,
            };
            let mut last = [0i32; 3];
            for_each_block(px, (h, w, cn), r.clone(), &divisors, |ci, b| {
                encode_block(&mut bw, &codes, table(ci), b, &mut last[ci]);
            });
            bw.flush();
            bw.out
        })
        .collect();

    let mut out = vec![0xFF, 0xD8];
    let (unit, dx, dy) = o.dpi.map_or((0u8, 1u16, 1u16), |(x, y)| (1, x, y));
    let mut jfif = b"JFIF\0\x01\x01".to_vec();
    jfif.push(unit);
    jfif.extend_from_slice(&dx.to_be_bytes());
    jfif.extend_from_slice(&dy.to_be_bytes());
    jfif.extend_from_slice(&[0, 0]);
    marker(&mut out, 0xE0, &jfif);
    if let Some(icc) = o.icc.filter(|i| !i.is_empty()) {
        let chunks: Vec<&[u8]> = icc.chunks(65533 - 14).collect();
        for (i, chunk) in chunks.iter().enumerate() {
            let mut body = b"ICC_PROFILE\0".to_vec();
            body.extend_from_slice(&[i as u8 + 1, chunks.len() as u8]);
            body.extend_from_slice(chunk);
            marker(&mut out, 0xE2, &body);
        }
    }
    for (t, q) in quant.iter().enumerate().take(if cn == 3 { 2 } else { 1 }) {
        let mut body = vec![t as u8];
        body.extend(NATURAL.iter().map(|&n| q[n] as u8));
        marker(&mut out, 0xDB, &body);
    }
    let mut sof = vec![8];
    sof.extend_from_slice(&(h as u16).to_be_bytes());
    sof.extend_from_slice(&(w as u16).to_be_bytes());
    sof.push(cn as u8);
    for ci in 0..cn {
        sof.extend_from_slice(&[ci as u8 + 1, 0x11, table(ci) as u8]);
    }
    marker(&mut out, 0xC0, &sof);
    for (t, (bits, vals)) in tables.iter().enumerate() {
        let mut body = vec![(((t % 2) << 4) | (t / 2)) as u8];
        body.extend_from_slice(&bits[1..]);
        body.extend_from_slice(vals);
        marker(&mut out, 0xC4, &body);
    }
    if restart {
        marker(&mut out, 0xDD, &((rows * mcu_cols) as u16).to_be_bytes());
    }
    let mut sos = vec![cn as u8];
    for ci in 0..cn {
        let t = table(ci) as u8;
        sos.extend_from_slice(&[ci as u8 + 1, t << 4 | t]);
    }
    sos.extend_from_slice(&[0, 63, 0]);
    marker(&mut out, 0xDA, &sos);
    let total: usize = scans.iter().map(Vec::len).sum();
    out.reserve(total + 2 * scans.len() + 2);
    for (i, s) in scans.iter().enumerate() {
        if i > 0 {
            out.extend_from_slice(&[0xFF, 0xD0 + ((i - 1) % 8) as u8]);
        }
        out.extend_from_slice(s);
    }
    out.extend_from_slice(&[0xFF, 0xD9]);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_90_tables_match_libjpeg() {
        let q = quant_table(&STD_LUMA, 90);
        assert_eq!(&q[..8], &[3, 2, 2, 3, 5, 8, 10, 12]);
        assert_eq!(quant_table(&STD_CHROMA, 90)[0], 3);
        assert_eq!(quant_table(&STD_LUMA, 1)[0], 255);
    }

    #[test]
    fn a_flat_block_has_only_dc() {
        let mut b = [[10i32; 8]; 8];
        fdct_islow(&mut b);
        let b = b.as_flattened();
        assert_eq!(b[0], 10 * 64);
        assert!(b[1..].iter().all(|&v| v == 0));
    }
}
