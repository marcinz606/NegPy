"""A background thread's large work enters the small pool, with the same bits."""

import threading

import numpy as np

import negpy_core


def _on_thread(fn):
    out = []
    t = threading.Thread(target=lambda: out.append(fn()))
    t.start()
    t.join()
    return out[0]


def _background(fn):
    def run():
        negpy_core.set_background_thread(True)
        return fn()

    return _on_thread(run)


def test_a_background_thread_uses_a_quarter_of_the_cores_and_at_least_two():
    cores = negpy_core.pool_threads()
    assert _background(negpy_core.pool_threads) == min(cores, max(2, cores // 4))
    assert _on_thread(negpy_core.pool_threads) == cores


def test_the_flag_stays_on_its_thread():
    def mark_then_read():
        negpy_core.set_background_thread(True)
        return _on_thread(negpy_core.pool_threads)

    assert _on_thread(mark_then_read) == negpy_core.pool_threads()


def test_the_background_pool_gives_the_same_bits():
    values = np.random.default_rng(0).random(1 << 20, dtype=np.float32)
    qs = [0.5, 1.0, 50.0, 99.0, 99.5]
    assert _background(lambda: negpy_core.percentiles(values, qs)) == negpy_core.percentiles(values, qs)
