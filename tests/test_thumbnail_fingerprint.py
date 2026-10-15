import json
import os
import tempfile
from dataclasses import dataclass, fields, replace

from PIL import Image

from negpy.domain.models import WorkspaceConfig
from negpy.features.hdr.models import HdrConfig
from negpy.features.lab.models import SharpenMethod
from negpy.features.process.models import DemosaicMode
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
        config = WorkspaceConfig()
        config = replace(config, exposure=replace(config.exposure, density=config.exposure.density + 0.1))
        reloaded = WorkspaceConfig.from_flat_dict(config.to_dict())
        assert _fp(reloaded) == _fp(config)

    def test_the_print_size_counts_only_while_a_carrier_is_drawn(self) -> None:
        config = WorkspaceConfig()
        larger = replace(config, export=replace(config.export, export_print_size=40.0))
        assert _fp(larger) == _fp(config)
        carrier = replace(config, finish=replace(config.finish, carrier_width=1.0))
        carrier_larger = replace(carrier, export=replace(carrier.export, export_print_size=40.0))
        assert _fp(carrier_larger) != _fp(carrier)

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

    def test_unhashed_fields_exist(self) -> None:
        config = WorkspaceConfig()
        for section, names in tf._UNHASHED_FIELDS.items():
            assert names <= set(getattr(config, section).__dataclass_fields__), section
        for section in tf.NON_PIXEL_SECTIONS | tf._BELOW_THUMBNAIL_SECTIONS:
            assert hasattr(config, section), section

    def test_export_only_and_below_thumbnail_settings_do_not_change_it(self) -> None:
        config = WorkspaceConfig()
        edited = replace(
            config,
            process=replace(
                config.process,
                demosaic_export=DemosaicMode.VNG,
                demosaic_preview=DemosaicMode.VNG,
                roll_name="Roll 7",
                baseline_source="roll:7",
            ),
            lab=replace(
                config.lab,
                sharpen=config.lab.sharpen + 0.5,
                sharpen_method=SharpenMethod.RL,
                sharpen_radius=config.lab.sharpen_radius + 1,
                sharpen_masking=config.lab.sharpen_masking + 0.1,
                chroma_denoise=config.lab.chroma_denoise + 1,
            ),
            retouch=replace(config.retouch, dust_remove=not config.retouch.dust_remove, manual_heal_strokes=[((0.5, 0.5),)]),
        )
        assert _fp(edited) == _fp(config)

    def test_a_visible_setting_in_a_partly_hashed_section_changes_it(self) -> None:
        config = WorkspaceConfig()
        assert _fp(replace(config, lab=replace(config.lab, saturation=config.lab.saturation + 0.2))) != _fp(config)
        assert _fp(replace(config, process=replace(config.process, highlight_reconstruction=1))) != _fp(config)

    def test_a_new_defaulted_field_leaves_it_unchanged(self) -> None:
        @dataclass(frozen=True)
        class GrownHdr(HdrConfig):
            hdr_new_field: bool = False

        config = WorkspaceConfig()
        assert _fp(replace(config, hdr=GrownHdr())) == _fp(config)
        assert _fp(replace(config, hdr=GrownHdr(hdr_new_field=True))) != _fp(config)

    def test_a_value_set_back_to_its_default_matches_the_default(self) -> None:
        config = WorkspaceConfig()
        density = config.exposure.density
        edited = replace(config, exposure=replace(config.exposure, density=density + 0.1))
        assert _fp(replace(edited, exposure=replace(edited.exposure, density=density))) == _fp(config)

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

    def test_rewriting_a_companion_file_changes_it(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            base = WorkspaceConfig()
            for name in ("hdr", "green", "blue", "part", "part_green", "part_blue"):
                with open(os.path.join(d, name), "wb") as fh:
                    fh.write(b"a")
            path = lambda name: os.path.join(d, name)  # noqa: E731
            configs = [
                replace(base, hdr=replace(base.hdr, hdr_paths=(path("hdr"),))),
                replace(base, rgbscan=replace(base.rgbscan, green_path=path("green"), blue_path=path("blue"))),
                replace(
                    base,
                    stitch=replace(base.stitch, stitch_paths=(path("part"),), stitch_triplets=((path("part_green"), path("part_blue")),)),
                ),
            ]
            before = [_fp(c) for c in configs]
            for name in ("hdr", "blue", "part_green"):
                with open(path(name), "wb") as fh:
                    fh.write(b"ab")
            assert all(a != b for a, b in zip(before, [_fp(c) for c in configs]))

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


_DEFAULTS_GOLDEN = os.path.join(os.path.dirname(__file__), "thumbnail_fingerprint_defaults.json")


def _current_defaults() -> dict[str, dict[str, str]]:
    config = WorkspaceConfig()
    return {
        f.name: dict(tf._section_defaults(type(getattr(config, f.name))))
        for f in fields(config)
        if f.name not in tf.NON_PIXEL_SECTIONS and f.name not in tf._BELOW_THUMBNAIL_SECTIONS
    }


def test_a_changed_default_bumps_the_render_version() -> None:
    # Fields missing from the golden are not checked; NEGPY_RECORD_THUMBNAIL_DEFAULTS=1 records them.
    with open(_DEFAULTS_GOLDEN) as fh:
        golden = json.load(fh)
    current = _current_defaults()
    changed = [
        f"{section}.{name}"
        for section, recorded in golden["defaults"].items()
        for name, value in recorded.items()
        if name in current.get(section, {}) and current[section][name] != value
    ]
    if golden["render_version"] == tf.THUMBNAIL_RENDER_VERSION:
        assert not changed, f"defaults changed without a THUMBNAIL_RENDER_VERSION bump: {changed}"
    else:
        assert os.environ.get("NEGPY_RECORD_THUMBNAIL_DEFAULTS"), "render version moved: rerun with NEGPY_RECORD_THUMBNAIL_DEFAULTS=1"
    if os.environ.get("NEGPY_RECORD_THUMBNAIL_DEFAULTS"):
        with open(_DEFAULTS_GOLDEN, "w") as fh:
            json.dump({"render_version": tf.THUMBNAIL_RENDER_VERSION, "defaults": current}, fh, indent=1, sort_keys=True)
            fh.write("\n")


def _other_value(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, str):
        return value + "x"
    raise AssertionError(f"pick another field; cannot vary {type(value).__name__}")
