from negpy.kernel.system.memory import available_system_memory_bytes, total_system_memory_bytes


def test_total_memory_probe_reports_a_size_or_unknown():
    """CI hosts differ; the probe only has to answer with bytes or None."""
    total = total_system_memory_bytes()
    assert total is None or (isinstance(total, int) and total > 0)


def test_available_memory_probe_reports_a_size_or_unknown():
    available = available_system_memory_bytes()
    assert available is None or (isinstance(available, int) and available > 0)
