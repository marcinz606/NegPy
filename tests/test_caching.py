from negpy.kernel.caching.logic import calculate_config_hash, CacheEntry
from negpy.kernel.caching.manager import PipelineCache
from negpy.features.exposure.models import ExposureConfig
import numpy as np


def test_calculate_config_hash_stability() -> None:
    config1 = ExposureConfig(density=1.0, grade=2.5)
    config2 = ExposureConfig(density=1.0, grade=2.5)

    h1 = calculate_config_hash(config1)
    h2 = calculate_config_hash(config2)

    assert h1 == h2
    assert isinstance(h1, str)
    assert len(h1) > 0


def test_calculate_config_hash_difference() -> None:
    config1 = ExposureConfig(density=1.0)
    config2 = ExposureConfig(density=1.1)

    assert calculate_config_hash(config1) != calculate_config_hash(config2)


def test_pipeline_cache_clear() -> None:
    cache = PipelineCache()
    dummy_data = np.zeros((10, 10), dtype=np.float32)
    entry = CacheEntry(config_hash="abc", data=dummy_data, metrics={})

    cache.source_hash = "source1"
    cache.base = entry

    cache.clear()

    assert cache.source_hash == ""
    assert cache.base is None
    assert cache.exposure is None


def test_config_hash_tells_minus_one_from_minus_two():
    """hash(-1.0) == hash(-2.0): a slider moving between them must not reuse the stage."""
    from negpy.features.geometry.models import GeometryConfig
    from negpy.kernel.caching.logic import calculate_config_hash

    assert calculate_config_hash((-1.0,)) != calculate_config_hash((-2.0,))
    assert calculate_config_hash(GeometryConfig(fine_rotation=-1.0)) != calculate_config_hash(GeometryConfig(fine_rotation=-2.0))
