use pyo3::prelude::*;

/// Sends the calling thread's large work to the background pool, or back to every core.
#[pyfunction]
fn set_background_thread(on: bool) {
    super::BACKGROUND.set(on);
}

/// The thread count of the pool that large work from the calling thread enters.
#[pyfunction]
fn pool_threads() -> usize {
    super::large_pool().current_num_threads()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_background_thread, m)?)?;
    m.add_function(wrap_pyfunction!(pool_threads, m)?)
}
