//! A PNG encoder that filters and deflates bands of rows in parallel. Each band is an
//! independent deflate run ended by a sync flush, so the bands join into one zlib stream; the
//! Adler-32 of the whole image is combined from the bands' own.

use miniz_oxide::deflate::core::{
    CompressorOxide, TDEFLFlush, TDEFLStatus, compress_to_output, create_comp_flags_from_zip_params,
};
use rayon::prelude::*;

/// Rows per band. Each band restarts the deflate window, which costs a few bytes per band.
const BAND_ROWS: usize = 64;
const ADLER_BASE: u32 = 65521;

/// zlib's adler32_combine: the checksum of A then B from theirs and B's length.
fn adler32_combine(a: u32, b: u32, len_b: usize) -> u32 {
    let rem = (len_b % ADLER_BASE as usize) as u32;
    let mut sum1 = a & 0xffff;
    let mut sum2 = (rem * sum1) % ADLER_BASE;
    sum1 += (b & 0xffff) + ADLER_BASE - 1;
    sum2 += ((a >> 16) & 0xffff) + ((b >> 16) & 0xffff) + ADLER_BASE - rem;
    if sum1 >= ADLER_BASE {
        sum1 -= ADLER_BASE;
    }
    if sum1 >= ADLER_BASE {
        sum1 -= ADLER_BASE;
    }
    if sum2 >= ADLER_BASE << 1 {
        sum2 -= ADLER_BASE << 1;
    }
    if sum2 >= ADLER_BASE {
        sum2 -= ADLER_BASE;
    }
    sum1 | (sum2 << 16)
}

#[inline(always)]
fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let (ia, ib, ic) = (a as i16, b as i16, c as i16);
    let p = ia + ib - ic;
    let (pa, pb, pc) = ((p - ia).abs(), (p - ib).abs(), (p - ic).abs());
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// One row with its filter byte, by the minimum sum of absolute differences over the five
/// PNG filters (the heuristic libpng suggests).
fn filter_row(
    row: &[u8],
    prev: Option<&[u8]>,
    bpp: usize,
    out: &mut Vec<u8>,
    scratch: &mut [Vec<u8>; 5],
) {
    let n = row.len();
    for s in scratch.iter_mut() {
        s.clear();
    }
    for i in 0..n {
        let x = row[i];
        let a = if i >= bpp { row[i - bpp] } else { 0 };
        let b = prev.map_or(0, |p| p[i]);
        let c = if i >= bpp {
            prev.map_or(0, |p| p[i - bpp])
        } else {
            0
        };
        scratch[0].push(x);
        scratch[1].push(x.wrapping_sub(a));
        scratch[2].push(x.wrapping_sub(b));
        scratch[3].push(x.wrapping_sub(((a as u16 + b as u16) / 2) as u8));
        scratch[4].push(x.wrapping_sub(paeth(a, b, c)));
    }
    let cost = |s: &Vec<u8>| {
        s.iter()
            .map(|&v| (v as i8).unsigned_abs() as u64)
            .sum::<u64>()
    };
    let best = (0..5).min_by_key(|&k| cost(&scratch[k])).unwrap_or(0);
    out.push(best as u8);
    out.extend_from_slice(&scratch[best]);
}

fn chunk(out: &mut Vec<u8>, tag: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(tag);
    out.extend_from_slice(data);
    out.extend_from_slice(&crc32fast::hash(&out[start..]).to_be_bytes());
}

/// A PNG of `h` rows of `row_bytes` big-endian sample bytes each (`row(y)` gives row y), with
/// `channels` samples of `depth` bits per pixel, deflated at zlib `level` (0 to 9).
pub fn encode(
    h: usize,
    w: usize,
    channels: usize,
    depth: u8,
    level: u8,
    row: impl Fn(usize, &mut Vec<u8>) + Sync,
) -> Vec<u8> {
    let bpp = channels * depth as usize / 8;
    let row_bytes = w * bpp;
    let flags = create_comp_flags_from_zip_params(level.min(9) as i32, 0, 0);
    let bands: Vec<(Vec<u8>, u32, usize)> = (0..h.div_ceil(BAND_ROWS))
        .into_par_iter()
        .map(|band| {
            let (y0, y1) = (band * BAND_ROWS, ((band + 1) * BAND_ROWS).min(h));
            let mut raw = Vec::with_capacity((y1 - y0) * (row_bytes + 1));
            let mut scratch: [Vec<u8>; 5] = std::array::from_fn(|_| Vec::with_capacity(row_bytes));
            let (mut prev, mut cur) =
                (Vec::with_capacity(row_bytes), Vec::with_capacity(row_bytes));
            if y0 > 0 {
                row(y0 - 1, &mut prev);
            }
            for y in y0..y1 {
                cur.clear();
                row(y, &mut cur);
                filter_row(
                    &cur,
                    (y > 0).then_some(prev.as_slice()),
                    bpp,
                    &mut raw,
                    &mut scratch,
                );
                std::mem::swap(&mut prev, &mut cur);
            }
            let adler = adler2::adler32_slice(&raw);
            let mut c = CompressorOxide::new(flags);
            let mut packed = Vec::with_capacity(raw.len() / 2);
            let flush = if y1 == h {
                TDEFLFlush::Finish
            } else {
                TDEFLFlush::Sync
            };
            let (status, _) = compress_to_output(&mut c, &raw, flush, |o| {
                packed.extend_from_slice(o);
                true
            });
            debug_assert!(matches!(status, TDEFLStatus::Okay | TDEFLStatus::Done));
            (packed, adler, raw.len())
        })
        .collect();

    let mut png = Vec::with_capacity(bands.iter().map(|b| b.0.len() + 12).sum::<usize>() + 64);
    png.extend_from_slice(b"\x89PNG\r\n\x1a\n");
    let color = match channels {
        1 => 0u8,
        2 => 4,
        3 => 2,
        _ => 6,
    };
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.extend_from_slice(&[depth, color, 0, 0, 0]);
    chunk(&mut png, b"IHDR", &ihdr);
    let flevel = match level {
        0 | 1 => 0x01u8,
        2..=5 => 0x5e,
        6 => 0x9c,
        _ => 0xda,
    };
    chunk(&mut png, b"IDAT", &[0x78, flevel]);
    let mut adler = 1u32;
    for (packed, a, len) in &bands {
        chunk(&mut png, b"IDAT", packed);
        adler = adler32_combine(adler, *a, *len);
    }
    chunk(&mut png, b"IDAT", &adler.to_be_bytes());
    chunk(&mut png, b"IEND", &[]);
    png
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adler_combine_matches_one_pass() {
        let data: Vec<u8> = (0..100_000u32).map(|i| (i * 7 % 251) as u8).collect();
        let (a, b) = data.split_at(31_337);
        let whole = adler2::adler32_slice(&data);
        let combined = adler32_combine(adler2::adler32_slice(a), adler2::adler32_slice(b), b.len());
        assert_eq!(whole, combined);
    }

    #[test]
    fn paeth_picks_the_nearest() {
        assert_eq!(paeth(10, 20, 10), 20);
        assert_eq!(paeth(20, 10, 10), 20);
        assert_eq!(paeth(5, 5, 5), 5);
    }
}
