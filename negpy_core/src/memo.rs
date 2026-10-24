//! Last results of detection work that several callers repeat on one frame: the base-log
//! meter, auto crop and the skew fit each find the film box on the same buffer.

use rayon::prelude::*;
use std::sync::Mutex;

const SEEDS: [u64; 2] = [0x9e37_79b9_7f4a_7c15, 0xd6e8_feb8_6659_fd93];

fn fold(mut h: [u64; 2], v: u64) -> [u64; 2] {
    const P: u64 = 0xa076_1d64_78bd_642f;
    for (s, seed) in h.iter_mut().zip(SEEDS) {
        let r = ((*s ^ v) as u128) * ((P ^ seed) as u128);
        *s = (r as u64) ^ ((r >> 64) as u64);
    }
    h
}

fn join(h: [u64; 2]) -> u128 {
    (h[0] as u128) << 64 | h[1] as u128
}

/// A 128-bit key of a float32 buffer's bits and shape: two seeded multiply-mix chains per block,
/// blocks in parallel, folded in order. Not for adversarial input.
pub fn key(data: &[f32], shape: &[usize]) -> u128 {
    let blocks: Vec<[u64; 2]> = data
        .par_chunks(1 << 16)
        .map(|block| {
            let (pairs, rest) = block.as_chunks::<2>();
            let h = pairs.iter().fold(SEEDS, |h, p| {
                fold(h, (p[0].to_bits() as u64) << 32 | p[1].to_bits() as u64)
            });
            rest.iter().fold(h, |h, v| fold(h, v.to_bits() as u64))
        })
        .collect();
    let h = shape
        .iter()
        .map(|&d| d as u64)
        .chain([data.len() as u64])
        .fold(SEEDS, fold);
    join(blocks.into_iter().flatten().fold(h, fold))
}

/// `key` extended by more values, such as the box a call refines.
pub fn extend(key: u128, extra: &[i64]) -> u128 {
    let h = [(key >> 64) as u64, key as u64];
    join(extra.iter().fold(h, |h, &v| fold(h, v as u64)))
}

/// The `N` most recently used values by key.
pub struct Memo<V, const N: usize> {
    slots: Mutex<Vec<(u128, V)>>,
}

impl<V: Clone, const N: usize> Default for Memo<V, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Clone, const N: usize> Memo<V, N> {
    pub const fn new() -> Self {
        Memo {
            slots: Mutex::new(Vec::new()),
        }
    }

    /// The value under `key`, or `f()` stored under it. `f` runs unlocked, so two callers may
    /// both compute a missing value; both get the same result.
    pub fn get_or(&self, key: u128, f: impl FnOnce() -> V) -> V {
        let lock = || self.slots.lock().unwrap_or_else(|e| e.into_inner());
        {
            let mut slots = lock();
            if let Some(i) = slots.iter().position(|(k, _)| *k == key) {
                let hit = slots.remove(i);
                slots.insert(0, hit);
                return slots[0].1.clone();
            }
        }
        let v = f();
        let mut slots = lock();
        slots.retain(|(k, _)| *k != key);
        slots.insert(0, (key, v.clone()));
        slots.truncate(N);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_sees_every_bit_and_the_shape() {
        let a: Vec<f32> = (0..200_000).map(|i| i as f32 * 0.25).collect();
        let k = key(&a, &[400, 500]);
        assert_eq!(k, key(&a.clone(), &[400, 500]));
        assert_ne!(k, key(&a, &[500, 400]));
        let mut b = a.clone();
        b[123_457] = f32::from_bits(b[123_457].to_bits() ^ 1);
        assert_ne!(k, key(&b, &[400, 500]));
        let mut c = a.clone();
        c[0] = -0.0;
        assert_ne!(key(&c, &[400, 500]), {
            c[0] = 0.0;
            key(&c, &[400, 500])
        });
    }

    #[test]
    fn memo_keeps_the_last_n() {
        let m: Memo<u32, 2> = Memo::new();
        assert_eq!(m.get_or(1, || 10), 10);
        assert_eq!(m.get_or(2, || 20), 20);
        assert_eq!(m.get_or(1, || 99), 10);
        assert_eq!(m.get_or(3, || 30), 30);
        assert_eq!(m.get_or(2, || 21), 21);
    }
}
