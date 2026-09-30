import os
import tempfile
from dataclasses import replace

from PIL import Image

from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.storage.local_asset_store import LocalAssetStore
from negpy.kernel.system.config import APP_CONFIG
from negpy.services.assets import thumbnail_fingerprint as tf


def _fp(config: WorkspaceConfig, **overrides) -> str:
    kwargs = dict(
        workspace_color_space="Adobe RGB",
        input_icc_path=None,
    )
    kwargs.update(overrides)
    return tf.thumbnail_fingerprint(config, **kwargs)


class TestThumbnailFingerprint:
    def test_equal_configs_give_equal_fingerprints(self) -> None:
        assert _fp(WorkspaceConfig()) == _fp(WorkspaceConfig())

    def test_survives_a_settings_round_trip(self) -> None:
        """Background checks read the config back from the DB, so a stored-and-reloaded
        config must fingerprint the same as the one that rendered."""
        config = WorkspaceConfig()
        config = replace(config, exposure=replace(config.exposure, density=config.exposure.density + 0.1))
        reloaded = WorkspaceConfig.from_flat_dict(config.to_dict())
        assert _fp(reloaded) == _fp(config)

    def test_a_pixel_setting_changes_it(self) -> None:
        config = WorkspaceConfig()
        edited = replace(config, exposure=replace(config.exposure, density=config.exposure.density + 0.1))
        assert _fp(edited) != _fp(config)

    def test_metadata_and_export_do_not_change_it(self) -> None:
        config = WorkspaceConfig()
        meta_field = next(iter(config.metadata.__dataclass_fields__))
        export_field = next(iter(config.export.__dataclass_fields__))
        edited = replace(
            config,
            metadata=replace(config.metadata, **{meta_field: _other_value(getattr(config.metadata, meta_field))}),
            export=replace(config.export, **{export_field: _other_value(getattr(config.export, export_field))}),
        )
        assert _fp(edited) == _fp(config)

    def test_workspace_color_space_changes_it(self) -> None:
        assert _fp(WorkspaceConfig(), workspace_color_space="ProPhoto RGB") != _fp(WorkspaceConfig())

    def test_replacing_an_input_profile_file_changes_it(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "in.icc")
            with open(path, "wb") as fh:
                fh.write(b"a")
            before = _fp(WorkspaceConfig(), input_icc_path=path)
            with open(path, "wb") as fh:
                fh.write(b"ab")
            assert _fp(WorkspaceConfig(), input_icc_path=path) != before

    def test_render_version_is_part_of_it(self, monkeypatch) -> None:
        before = _fp(WorkspaceConfig())
        monkeypatch.setattr(tf, "THUMBNAIL_RENDER_VERSION", tf.THUMBNAIL_RENDER_VERSION + 1)
        assert _fp(WorkspaceConfig()) != before

    def test_comment_round_trip_and_foreign_comments(self) -> None:
        assert tf.decode_comment(tf.encode_comment("abc")) == "abc"
        assert tf.encode_comment(None) is None
        assert tf.decode_comment(None) is None
        assert tf.decode_comment(b"made by some other tool") is None
        assert tf.decode_comment(b"\xff\xfe") is None

    def test_only_a_real_match_is_current(self) -> None:
        assert tf.is_current("abc", "abc")
        assert not tf.is_current("abc", "abd")
        assert not tf.is_current(None, "abc")
        assert not tf.is_current(tf.QUICK, tf.QUICK)


class TestAssetStoreFingerprint:
    def setup_method(self) -> None:
        self.dir = tempfile.mkdtemp()
        self.store = LocalAssetStore(self.dir, self.dir)
        self.store.initialize()
        ts = APP_CONFIG.thumbnail_size
        self.img = Image.new("RGB", (ts, ts), (10, 20, 30))

    def test_saved_fingerprint_reads_back(self) -> None:
        self.store.save_thumbnail("h1", self.img, fingerprint="deadbeef")
        assert self.store.get_thumbnail_fingerprint("h1") == "deadbeef"
        # The image itself is still served as before.
        assert self.store.get_thumbnail("h1") is not None

    def test_legacy_thumbnail_without_a_comment_is_unknown(self) -> None:
        self.store.save_thumbnail("h2", self.img)
        assert self.store.get_thumbnail_fingerprint("h2") is None

    def test_a_derived_image_does_not_carry_its_sources_fingerprint(self) -> None:
        self.store.save_thumbnail("h4", self.img, fingerprint="deadbeef")
        turned = self.store.get_thumbnail("h4").transpose(Image.Transpose.ROTATE_90)
        self.store.save_thumbnail("h4", turned)
        assert self.store.get_thumbnail_fingerprint("h4") is None

    def test_missing_thumbnail_is_unknown(self) -> None:
        assert self.store.get_thumbnail_fingerprint("absent") is None

    def test_clearing_thumbnails_clears_fingerprints(self) -> None:
        self.store.save_thumbnail("h3", self.img, fingerprint="deadbeef")
        self.store.clear_thumbnails()
        assert self.store.get_thumbnail_fingerprint("h3") is None


def _other_value(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, str):
        return value + "x"
    raise AssertionError(f"pick another field; cannot vary {type(value).__name__}")
