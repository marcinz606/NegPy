from dataclasses import replace

from negpy.desktop.settings_catalog import meter_inputs_key
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG


def test_the_key_follows_what_metering_reads():
    cfg = DEFAULT_WORKSPACE_CONFIG
    key = meter_inputs_key(cfg)

    assert meter_inputs_key(replace(cfg, geometry=replace(cfg.geometry, crop_rect=(0.1, 0.1, 0.9, 0.9)))) != key
    assert meter_inputs_key(replace(cfg, process=replace(cfg.process, analysis_buffer=0.2))) != key
    assert meter_inputs_key(replace(cfg, process=replace(cfg.process, analysis_rect=(0.2, 0.2, 0.8, 0.8)))) != key


def test_the_average_flags_and_the_look_leave_it_alone():
    cfg = DEFAULT_WORKSPACE_CONFIG
    key = meter_inputs_key(cfg)

    assert meter_inputs_key(replace(cfg, process=replace(cfg.process, use_luma_average=True, use_color_average=True))) == key
    assert meter_inputs_key(replace(cfg, exposure=replace(cfg.exposure, density=cfg.exposure.density + 0.3))) == key
