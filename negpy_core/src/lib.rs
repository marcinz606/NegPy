pub mod altprocess;
pub mod analysis;
pub mod autocrop;
pub mod bounds;
pub mod capture;
pub mod color;
pub mod components;
pub mod contours;
pub mod convert;
pub mod detect;
pub mod display;
pub mod distance;
pub mod filter;
pub mod finish;
pub mod flatfield;
pub mod gray;
pub mod hdr;
pub mod histogram;
pub mod inpaint;
pub mod jpeg;
pub mod lab;
pub mod lens;
pub mod loaders;
pub mod lsd;
pub mod mathf;
pub mod median;
pub mod memo;
pub mod meter;
pub mod morph;
pub mod phase;
pub mod print_curve;
pub mod process;
pub mod register;
pub mod resize;
pub mod retouch;
pub mod rgbscan;
pub mod seam;
pub mod sensor;
pub mod shapes;
pub mod sift;
pub mod skew;
pub mod stats;
pub mod stitch;
pub mod toning;
pub mod warp;
pub mod zones;

/// `f(dst[i], src[i])` for every i, in parallel blocks: a rayon task per element costs more than
/// the work. Pass a `move` closure: a capture read through a reference may alias `dst`, so the
/// loop reloads it per element and does not vectorize.
pub(crate) fn par_zip<T: Send, U: Sync, F: Fn(&mut T, &U) + Sync>(dst: &mut [T], src: &[U], f: F) {
    use rayon::prelude::*;
    const BLOCK: usize = 1 << 14;
    dst.par_chunks_mut(BLOCK)
        .zip(src.par_chunks(BLOCK))
        .for_each(|(d, s)| d.iter_mut().zip(s).for_each(|(d, s)| f(d, s)));
}

#[cfg(feature = "python")]
mod py;

#[cfg(feature = "python")]
#[pyo3::pymodule]
fn negpy_core(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    py::stats::register(m)?;
    py::resize::register(m)?;
    py::filter::register(m)?;
    py::morph::register(m)?;
    py::loaders::register(m)?;
    py::process::register(m)?;
    py::phase::register(m)?;
    py::rgbscan::register(m)?;
    py::capture::register(m)?;
    py::print_curve::register(m)?;
    py::gray::register(m)?;
    py::contours::register(m)?;
    py::components::register(m)?;
    py::convert::register(m)?;
    py::histogram::register(m)?;
    py::meter::register(m)?;
    py::retouch::register(m)?;
    py::bounds::register(m)?;
    py::color::register(m)?;
    py::toning::register(m)?;
    py::sensor::register(m)?;
    py::hdr::register(m)?;
    py::display::register(m)?;
    py::analysis::register(m)?;
    py::warp::register(m)?;
    py::lab::register(m)?;
    py::lens::register(m)?;
    py::altprocess::register(m)?;
    py::finish::register(m)?;
    py::flatfield::register(m)?;
    py::stitch::register(m)?;
    py::skew::register(m)?;
    py::register::register(m)?;
    py::autocrop::register(m)?;
    py::zones::register(m)?;
    py::pool::register(m)?;
    py::jpeg::register(m)
}
