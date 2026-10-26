//! `apply_exif_orientation`: an EXIF orientation (2-8) baked into the pixels, on any pixel size.

use rayon::prelude::*;

/// Output rows per parallel task, and the square tile a 90-degree turn copies at a time so
/// that both the read and the write stay in cache.
const ROWS: usize = 64;
const TILE: usize = 64;

/// Output dimensions of an (h, w) image under `orientation`.
pub fn oriented_dims(h: usize, w: usize, orientation: u8) -> (usize, usize) {
    if orientation >= 5 { (w, h) } else { (h, w) }
}

/// The source (row, column) of output pixel (y, x), for an (h, w) source.
#[inline(always)]
fn source(orientation: u8, h: usize, w: usize, y: usize, x: usize) -> (usize, usize) {
    match orientation {
        2 => (y, w - 1 - x),
        3 => (h - 1 - y, w - 1 - x),
        4 => (h - 1 - y, x),
        5 => (x, y),
        6 => (h - 1 - x, y),
        7 => (h - 1 - x, w - 1 - y),
        8 => (x, w - 1 - y),
        _ => (y, x),
    }
}

fn orient_px<const N: usize>(src: &[u8], h: usize, w: usize, orientation: u8, dst: &mut [u8]) {
    let src = src.as_chunks::<N>().0;
    let (oh, ow) = oriented_dims(h, w, orientation);
    dst.par_chunks_mut(ROWS * ow * N)
        .enumerate()
        .for_each(|(band, d)| {
            let d = d.as_chunks_mut::<N>().0;
            let y0 = band * ROWS;
            let rows = d.len() / ow;
            for x0 in (0..ow).step_by(TILE) {
                for dy in 0..rows {
                    let y = y0 + dy;
                    for x in x0..(x0 + TILE).min(ow) {
                        let (sy, sx) = source(orientation, h, w, y, x);
                        d[dy * ow + x] = src[sy * w + sx];
                    }
                }
            }
            debug_assert!(y0 + rows <= oh);
        });
}

/// `src`, an (h, w) image of `px`-byte pixels, turned by `orientation` into `dst`.
pub fn orient(src: &[u8], h: usize, w: usize, px: usize, orientation: u8, dst: &mut [u8]) {
    match px {
        1 => orient_px::<1>(src, h, w, orientation, dst),
        2 => orient_px::<2>(src, h, w, orientation, dst),
        3 => orient_px::<3>(src, h, w, orientation, dst),
        4 => orient_px::<4>(src, h, w, orientation, dst),
        6 => orient_px::<6>(src, h, w, orientation, dst),
        8 => orient_px::<8>(src, h, w, orientation, dst),
        12 => orient_px::<12>(src, h, w, orientation, dst),
        16 => orient_px::<16>(src, h, w, orientation, dst),
        _ => {
            let (_, ow) = oriented_dims(h, w, orientation);
            dst.par_chunks_mut(ow * px).enumerate().for_each(|(y, d)| {
                for (x, p) in d.chunks_exact_mut(px).enumerate() {
                    let (sy, sx) = source(orientation, h, w, y, x);
                    let o = (sy * w + sx) * px;
                    p.copy_from_slice(&src[o..o + px]);
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turns_a_2x3_image() {
        // 0 1 2
        // 3 4 5
        let src: Vec<u8> = (0..6).collect();
        let run = |o, px: usize| {
            let mut dst = [0u8; 6];
            orient(&src, 2, 3, px, o, &mut dst);
            dst
        };
        assert_eq!(run(2, 1), [2, 1, 0, 5, 4, 3]);
        assert_eq!(run(3, 1), [5, 4, 3, 2, 1, 0]);
        assert_eq!(run(4, 1), [3, 4, 5, 0, 1, 2]);
        assert_eq!(run(5, 1), [0, 3, 1, 4, 2, 5]);
        assert_eq!(run(6, 1), [3, 0, 4, 1, 5, 2]);
        assert_eq!(run(7, 1), [5, 2, 4, 1, 3, 0]);
        assert_eq!(run(8, 1), [2, 5, 1, 4, 0, 3]);
    }

    #[test]
    fn odd_pixel_sizes_take_the_generic_path() {
        let src: Vec<u8> = (0..30).collect();
        let mut fast = [0u8; 18];
        let mut slow = [0u8; 30];
        orient(&src, 2, 3, 5, 6, &mut slow);
        let src3: Vec<u8> = (0..18).collect();
        orient(&src3, 2, 3, 3, 6, &mut fast);
        assert_eq!(&slow[..5], &src[15..20]);
        assert_eq!(&fast[..3], &src3[9..12]);
    }
}
