"""Alternative Processes config: serialization and the lith_enabled migration."""

import logging
from dataclasses import replace

from negpy.domain.models import WorkspaceConfig
from negpy.features.altprocess.models import AltProcess, Sensitizer


class TestDefaults:
    def test_off_by_default(self):
        assert WorkspaceConfig().altproc.alt_process == AltProcess.NONE

    def test_roundtrip(self):
        base = WorkspaceConfig()
        cfg = replace(
            base,
            altproc=replace(
                base.altproc,
                alt_process=AltProcess.CYANOTYPE,
                cyano_sensitizer=Sensitizer.NEW,
                cyano_scale=2.4,
            ),
        )
        restored = WorkspaceConfig.from_flat_dict(cfg.to_dict()).altproc
        assert restored.alt_process == AltProcess.CYANOTYPE
        assert restored.cyano_sensitizer == Sensitizer.NEW
        assert restored.cyano_scale == 2.4


class TestLithEnabledMigration:
    def test_true_becomes_lith(self):
        assert WorkspaceConfig.from_flat_dict({"lith_enabled": True}).altproc.alt_process == AltProcess.LITH

    def test_false_becomes_none(self):
        assert WorkspaceConfig.from_flat_dict({"lith_enabled": False}).altproc.alt_process == AltProcess.NONE

    def test_the_lith_sliders_survive(self):
        alt = WorkspaceConfig.from_flat_dict({"lith_enabled": True, "lith_snatch": 0.8}).altproc
        assert alt.lith_snatch == 0.8

    def test_no_unknown_key_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="negpy.domain.models"):
            WorkspaceConfig.from_flat_dict({"lith_enabled": True})
        assert "lith_enabled" not in caplog.text
