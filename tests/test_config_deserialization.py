import json
import logging
import unittest
from dataclasses import replace
from negpy.domain.models import AspectRatio, ExportConfig, ExportFormat, ExportPreset, ExportResolutionMode, WorkspaceConfig
from negpy.features.process.models import ProcessMode
from negpy.kernel.caching.logic import calculate_config_hash


class TestConfigDeserialization(unittest.TestCase):
    def test_basic_deserialization(self):
        data = {
            "process_mode": ProcessMode.BW,
            "density": 1.2,
            "grade": 3.0,
            "export_fmt": "TIFF",
        }
        config = WorkspaceConfig.from_flat_dict(data)

        self.assertEqual(config.process.process_mode, ProcessMode.BW)
        self.assertEqual(config.exposure.density, 1.2)
        # Legacy 0-5 paper grade migrates to ISO R (150 - 20*G).
        self.assertEqual(config.exposure.grade, 90.0)
        self.assertEqual(config.export.export_fmt, "TIFF")

    def test_narrowband_scan_round_trip(self):
        self.assertFalse(WorkspaceConfig().to_dict()["narrowband_scan"])
        config = WorkspaceConfig.from_flat_dict({"narrowband_scan": True})
        self.assertTrue(config.process.narrowband_scan)

    def test_unknown_keys_warning(self):
        data = {
            "process_mode": ProcessMode.BW,
            "density": 0.5,
            "this_is_unknown": 42,
            "also_unknown": "hello",
        }
        with self.assertLogs("negpy.domain.models", level=logging.WARNING) as cm:
            config = WorkspaceConfig.from_flat_dict(data)

        self.assertEqual(config.process.process_mode, ProcessMode.BW)
        self.assertEqual(config.exposure.density, 0.5)
        self.assertTrue(any("Dropping unknown config keys" in msg for msg in cm.output))
        self.assertIn("also_unknown", cm.output[0])
        self.assertIn("this_is_unknown", cm.output[0])

    def test_no_warning_when_all_keys_valid(self):
        data = {"process_mode": ProcessMode.C41, "density": 0.0}
        with self.assertNoLogs("negpy.domain.models", level=logging.WARNING):
            WorkspaceConfig.from_flat_dict(data)

    def test_crossover_paper_black_roundtrip(self):
        config = WorkspaceConfig()
        config = replace(
            config,
            process=replace(
                config.process,
                white_point_trim_red=0.05,
                white_point_trim_green=-0.1,
                white_point_trim_blue=0.02,
                black_point_trim_red=-0.03,
                black_point_trim_green=0.07,
                black_point_trim_blue=0.11,
            ),
            exposure=replace(
                config.exposure,
                grade_trim_red=12.0,
                grade_trim_green=-8.0,
                grade_trim_blue=30.0,
                toe_trim_red=0.3,
                toe_trim_green=-0.1,
                toe_trim_blue=0.7,
                shoulder_trim_red=-0.5,
                shoulder_trim_green=0.2,
                shoulder_trim_blue=0.05,
                paper_black=True,
                midtone_gamma=-0.2,
                midtone_gamma_trim_red=0.15,
                midtone_gamma_trim_green=-0.25,
                midtone_gamma_trim_blue=0.4,
                toe_width_trim_red=1.2,
                toe_width_trim_green=-0.8,
                toe_width_trim_blue=0.4,
                shoulder_width_trim_red=-1.5,
                shoulder_width_trim_green=0.6,
                shoulder_width_trim_blue=2.0,
                shadow_density=-0.45,
                highlight_density=0.25,
                shadow_grade=-18.0,
                highlight_grade=22.0,
                shadow_grade_trim_red=6.0,
                shadow_grade_trim_green=-4.0,
                shadow_grade_trim_blue=9.0,
                highlight_grade_trim_red=-7.0,
                highlight_grade_trim_green=3.0,
                highlight_grade_trim_blue=-2.0,
            ),
        )
        reloaded = WorkspaceConfig.from_flat_dict(json.loads(json.dumps(config.to_dict(), default=str)))
        self.assertEqual(reloaded.exposure.grade_trim_red, 12.0)
        self.assertEqual(reloaded.exposure.grade_trim_green, -8.0)
        self.assertEqual(reloaded.exposure.grade_trim_blue, 30.0)
        self.assertEqual(reloaded.exposure.toe_trim_red, 0.3)
        self.assertEqual(reloaded.exposure.toe_trim_green, -0.1)
        self.assertEqual(reloaded.exposure.toe_trim_blue, 0.7)
        self.assertEqual(reloaded.exposure.shoulder_trim_red, -0.5)
        self.assertEqual(reloaded.exposure.shoulder_trim_green, 0.2)
        self.assertEqual(reloaded.exposure.shoulder_trim_blue, 0.05)
        self.assertTrue(reloaded.exposure.paper_black)
        self.assertEqual(reloaded.exposure.midtone_gamma, -0.2)
        self.assertEqual(reloaded.exposure.midtone_gamma_trim_red, 0.15)
        self.assertEqual(reloaded.exposure.midtone_gamma_trim_green, -0.25)
        self.assertEqual(reloaded.exposure.midtone_gamma_trim_blue, 0.4)
        self.assertEqual(reloaded.exposure.toe_width_trim_red, 1.2)
        self.assertEqual(reloaded.exposure.toe_width_trim_green, -0.8)
        self.assertEqual(reloaded.exposure.toe_width_trim_blue, 0.4)
        self.assertEqual(reloaded.exposure.shoulder_width_trim_red, -1.5)
        self.assertEqual(reloaded.exposure.shoulder_width_trim_green, 0.6)
        self.assertEqual(reloaded.exposure.shoulder_width_trim_blue, 2.0)
        self.assertEqual(reloaded.exposure.shadow_density, -0.45)
        self.assertEqual(reloaded.exposure.highlight_density, 0.25)
        self.assertEqual(reloaded.exposure.shadow_grade, -18.0)
        self.assertEqual(reloaded.exposure.highlight_grade, 22.0)
        self.assertEqual(reloaded.exposure.shadow_grade_trim_red, 6.0)
        self.assertEqual(reloaded.exposure.shadow_grade_trim_green, -4.0)
        self.assertEqual(reloaded.exposure.shadow_grade_trim_blue, 9.0)
        self.assertEqual(reloaded.exposure.highlight_grade_trim_red, -7.0)
        self.assertEqual(reloaded.exposure.highlight_grade_trim_green, 3.0)
        self.assertEqual(reloaded.exposure.highlight_grade_trim_blue, -2.0)
        self.assertEqual(reloaded.process.white_point_trim_red, 0.05)
        self.assertEqual(reloaded.process.white_point_trim_green, -0.1)
        self.assertEqual(reloaded.process.white_point_trim_blue, 0.02)
        self.assertEqual(reloaded.process.black_point_trim_red, -0.03)
        self.assertEqual(reloaded.process.black_point_trim_green, 0.07)
        self.assertEqual(reloaded.process.black_point_trim_blue, 0.11)

    def test_legacy_true_black_migrates_inverted(self):
        # True Black renamed to Paper Black with inverted polarity: a saved edit's
        # rendered look must survive the rename.
        off = WorkspaceConfig.from_flat_dict({"true_black": False})
        self.assertTrue(off.exposure.paper_black)
        on = WorkspaceConfig.from_flat_dict({"true_black": True})
        self.assertFalse(on.exposure.paper_black)

    def test_use_original_res_true_migrates_to_original_mode(self):
        data = {"use_original_res": True, "export_print_size": 30.0}
        config = WorkspaceConfig.from_flat_dict(data)
        self.assertEqual(config.export.export_resolution_mode, ExportResolutionMode.ORIGINAL.value)

    def test_use_original_res_false_migrates_to_print_mode(self):
        data = {"use_original_res": False, "export_print_size": 30.0}
        config = WorkspaceConfig.from_flat_dict(data)
        self.assertEqual(config.export.export_resolution_mode, ExportResolutionMode.PRINT.value)

    def test_explicit_mode_wins_over_legacy_use_original_res(self):
        data = {
            "use_original_res": True,
            "export_resolution_mode": ExportResolutionMode.TARGET_PX.value,
        }
        config = WorkspaceConfig.from_flat_dict(data)
        self.assertEqual(config.export.export_resolution_mode, ExportResolutionMode.TARGET_PX.value)

    def test_flatfield_apply_does_not_collide_with_rgbscan_enabled(self):
        """flatfield.apply and rgbscan.enabled must round-trip independently (#356)."""
        cfg = WorkspaceConfig(
            flatfield=replace(WorkspaceConfig().flatfield, apply=True, profile_id="abc123"),
            rgbscan=replace(WorkspaceConfig().rgbscan, enabled=False),
        )
        back = WorkspaceConfig.from_flat_dict(cfg.to_dict())
        self.assertTrue(back.flatfield.apply)
        self.assertFalse(back.rgbscan.enabled)

        cfg2 = WorkspaceConfig(
            flatfield=replace(WorkspaceConfig().flatfield, apply=False),
            rgbscan=replace(WorkspaceConfig().rgbscan, enabled=True),
        )
        back2 = WorkspaceConfig.from_flat_dict(cfg2.to_dict())
        self.assertFalse(back2.flatfield.apply)
        self.assertTrue(back2.rgbscan.enabled)

    def test_autocrop_rebate_trim_round_trips_without_clobbering_a_neighbour(self):
        cfg = WorkspaceConfig(geometry=replace(WorkspaceConfig().geometry, autocrop_rebate_trim=1.35, autocrop_offset=4))
        flat = cfg.to_dict()
        self.assertEqual(flat["autocrop_rebate_trim"], 1.35)
        back = WorkspaceConfig.from_flat_dict(flat)
        self.assertEqual(back.geometry.autocrop_rebate_trim, 1.35)
        self.assertEqual(back.geometry.autocrop_offset, 4)

    def test_config_without_rebate_trim_defaults_to_a_full_cut(self):
        back = WorkspaceConfig.from_flat_dict({"autocrop_offset": 2})
        self.assertEqual(back.geometry.autocrop_rebate_trim, 1.0)

    def test_legacy_use_original_res_does_not_warn(self):
        data = {"use_original_res": False}
        with self.assertNoLogs("negpy.domain.models", level=logging.WARNING):
            WorkspaceConfig.from_flat_dict(data)

    def test_legacy_use_roll_average_true_splits_to_both_axes(self):
        config = WorkspaceConfig.from_flat_dict({"use_roll_average": True})
        self.assertTrue(config.process.use_luma_average)
        self.assertTrue(config.process.use_color_average)

    def test_legacy_use_roll_average_false_splits_to_both_axes(self):
        config = WorkspaceConfig.from_flat_dict({"use_roll_average": False})
        self.assertFalse(config.process.use_luma_average)
        self.assertFalse(config.process.use_color_average)

    def test_legacy_use_roll_average_does_not_warn(self):
        with self.assertNoLogs("negpy.domain.models", level=logging.WARNING):
            WorkspaceConfig.from_flat_dict({"use_roll_average": True})

    def test_retired_dng_export_migrates_to_tiff(self):
        # DNG export was removed; a saved edit must land on the other 16-bit
        # format, not fall through the encoder to 8-bit JPEG.
        config = WorkspaceConfig.from_flat_dict({"export_fmt": "DNG"})
        self.assertEqual(config.export.export_fmt, ExportFormat.TIFF)

    def test_retired_dng_export_migrates_outside_flat_dict(self):
        # Covers the sticky last_export_config path (ExportConfig(**filtered))
        # and saved export presets, neither of which goes through from_flat_dict.
        self.assertEqual(ExportConfig(export_fmt="DNG").export_fmt, ExportFormat.TIFF)
        self.assertEqual(ExportPreset.from_dict({"export_fmt": "DNG"}).export_fmt, ExportFormat.TIFF)

    def test_crop_rect_survives_db_roundtrip_as_tuple(self):
        """Manual crop saved to JSON reloads as a list, making the frozen
        GeometryConfig unhashable and crashing the pipeline hash. The reloaded
        rect must be a tuple and geometry must stay hashable."""
        config = WorkspaceConfig()
        config = replace(config, geometry=replace(config.geometry, crop_rect=(0.1, 0.2, 0.8, 0.9)))

        # Exactly what repository.save_file_settings / load_file_settings do.
        reloaded = WorkspaceConfig.from_flat_dict(json.loads(json.dumps(config.to_dict(), default=str)))

        self.assertIsInstance(reloaded.geometry.crop_rect, tuple)
        self.assertEqual(reloaded.geometry.crop_rect, (0.1, 0.2, 0.8, 0.9))
        hash(reloaded.geometry)  # must not raise

    def test_crop_rect_hashable_in_engine_base_key(self):
        """DarkroomEngine wraps geometry in a plain tuple (base_key) before
        hashing; an unhashable geometry made calculate_config_hash fall through
        to asdict(tuple) -> 'asdict() should be called on dataclass instances'."""
        config = WorkspaceConfig()
        config = replace(config, geometry=replace(config.geometry, crop_rect=(0.1, 0.2, 0.8, 0.9)))
        reloaded = WorkspaceConfig.from_flat_dict(json.loads(json.dumps(config.to_dict(), default=str)))

        base_key = (
            reloaded.process.process_mode,
            reloaded.process.e6_normalize,
            reloaded.geometry,
            reloaded.process.analysis_buffer,
            reloaded.process.luma_range_clip,
        )
        self.assertIsInstance(calculate_config_hash(base_key), str)

    def test_analysis_rect_survives_db_roundtrip_as_tuple(self):
        """The freehand analysis region must reload as a tuple so the frozen
        ProcessConfig stays hashable for the pipeline cache key."""
        config = WorkspaceConfig()
        config = replace(config, process=replace(config.process, analysis_rect=(0.1, 0.2, 0.8, 0.9)))

        reloaded = WorkspaceConfig.from_flat_dict(json.loads(json.dumps(config.to_dict(), default=str)))

        self.assertIsInstance(reloaded.process.analysis_rect, tuple)
        self.assertEqual(reloaded.process.analysis_rect, (0.1, 0.2, 0.8, 0.9))
        hash(reloaded.process)  # must not raise

    def test_legacy_vignette_strength_migrates_to_stops(self):
        config = WorkspaceConfig.from_flat_dict({"vignette_strength": -0.5})
        self.assertAlmostEqual(config.finish.vignette_stops, 1.0)

    def test_legacy_vignette_strength_dropped_when_stops_present(self):
        config = WorkspaceConfig.from_flat_dict({"vignette_strength": -0.5, "vignette_stops": 0.3})
        self.assertAlmostEqual(config.finish.vignette_stops, 0.3)

    def test_legacy_vignette_strength_does_not_warn(self):
        with self.assertNoLogs("negpy.domain.models", level=logging.WARNING):
            WorkspaceConfig.from_flat_dict({"vignette_strength": 0.2})

    def test_legacy_carrier_enabled_false_zeros_width(self):
        # Pre-#542 saves always serialize both keys together; the old default
        # width (2.0) must not read as "on" under the new width>0 gating just
        # because the separate enabled toggle is gone.
        config = WorkspaceConfig.from_flat_dict({"carrier_enabled": False, "carrier_width": 2.0})
        self.assertEqual(config.finish.carrier_width, 0.0)

    def test_legacy_carrier_enabled_true_keeps_width(self):
        config = WorkspaceConfig.from_flat_dict({"carrier_enabled": True, "carrier_width": 3.0})
        self.assertEqual(config.finish.carrier_width, 3.0)

    def test_legacy_carrier_enabled_does_not_warn(self):
        with self.assertNoLogs("negpy.domain.models", level=logging.WARNING):
            WorkspaceConfig.from_flat_dict({"carrier_enabled": False, "carrier_width": 2.0})

    def test_autocrop_mode_defaults_to_image_for_legacy_dicts(self):
        config = WorkspaceConfig.from_flat_dict({"process_mode": ProcessMode.C41})
        self.assertEqual(config.geometry.autocrop_mode, "image")

    def test_autocrop_mode_survives_roundtrip(self):
        config = WorkspaceConfig()
        config = replace(config, geometry=replace(config.geometry, autocrop_mode="film"))

        reloaded = WorkspaceConfig.from_flat_dict(json.loads(json.dumps(config.to_dict(), default=str)))

        self.assertEqual(reloaded.geometry.autocrop_mode, "film")
        hash(reloaded.geometry)  # must not raise

    def test_autocrop_mode_invalid_value_coerces_to_image(self):
        config = WorkspaceConfig.from_flat_dict({"autocrop_mode": "banana"})
        self.assertEqual(config.geometry.autocrop_mode, "image")

    def test_stitch_config_round_trips(self):
        """Stitch fields must survive to_dict/from_flat_dict (JSON turns tuples into
        lists), keeping the frozen StitchConfig hashable for the pipeline cache key."""
        config = replace(
            WorkspaceConfig(),
            stitch=replace(
                WorkspaceConfig().stitch,
                stitch_enabled=True,
                stitch_paths=("/scans/b.raw", "/scans/c.raw"),
                stitch_transforms=((1.0, 0.0, 10.0, 0.0, 1.0, 20.0),),
                stitch_canvas=(4000, 3000),
                stitch_sizes=((2000, 3000), (2100, 3000)),
            ),
        )
        reloaded = WorkspaceConfig.from_flat_dict(json.loads(json.dumps(config.to_dict(), default=str)))

        self.assertTrue(reloaded.stitch.stitch_enabled)
        self.assertEqual(reloaded.stitch.stitch_paths, ("/scans/b.raw", "/scans/c.raw"))
        self.assertEqual(reloaded.stitch.stitch_canvas, (4000, 3000))
        self.assertEqual(reloaded.stitch.stitch_sizes, ((2000, 3000), (2100, 3000)))
        hash(reloaded.stitch)  # must not raise

    def test_legacy_process_mode_names_still_load(self):
        """The modes were renamed (C41 -> Color Negative, B&W -> B&W Negative, E-6 ->
        Transparency). Edits saved under the old names must open in the same mode."""
        for legacy, expected in (("C41", ProcessMode.C41), ("B&W", ProcessMode.BW), ("E-6", ProcessMode.E6)):
            config = WorkspaceConfig.from_flat_dict({"process_mode": legacy})
            self.assertEqual(config.process.process_mode, expected)
            self.assertIsInstance(config.process.process_mode, ProcessMode)

    def test_unknown_process_mode_falls_back_to_color_negative(self):
        """A corrupt or hand-edited value renders as it always did, rather than failing the load."""
        self.assertEqual(WorkspaceConfig.from_flat_dict({"process_mode": "Kodachrome"}).process.process_mode, ProcessMode.C41)

    def test_retired_enum_values_fall_back_to_their_default(self):
        config = WorkspaceConfig.from_flat_dict({"export_fmt": "PSD", "export_resolution_mode": "contact_sheet", "autocrop_ratio": "13:17"})
        self.assertEqual(config.export.export_fmt, ExportFormat.JPEG)
        self.assertEqual(config.export.export_resolution_mode, ExportResolutionMode.ORIGINAL)
        self.assertEqual(config.geometry.autocrop_ratio, AspectRatio.R_3_2)

    def test_no_sub_config_is_missing_from_the_known_keys_set(self):
        """`from_flat_dict` validates incoming keys against a hand-maintained
        `config_classes` list. A sub-config added to WorkspaceConfig but not to that list
        has every one of its keys dropped on load — a warning, then silent data loss.

        Round-tripping the default config is enough to catch it, because `to_dict` emits
        every sub-config's keys whatever their values. StitchConfig was missing once, and
        HdrConfig was missing when it was added; both showed up here."""
        with self.assertNoLogs("negpy.domain.models", level=logging.WARNING):
            WorkspaceConfig.from_flat_dict(WorkspaceConfig().to_dict())


if __name__ == "__main__":
    unittest.main()
