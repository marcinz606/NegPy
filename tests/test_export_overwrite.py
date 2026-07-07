import os
import tempfile
import unittest

from negpy.desktop.workers.export import (
    ExportTask,
    find_export_conflicts,
    resolve_export_dir,
    resolve_export_naming,
    resolve_export_target_path,
)
from negpy.domain.models import ExportFormat, ExportPreset, ExportPresetOutputMode, WorkspaceConfig


def _task(src_path, *, out_mode, out_path="", subfolder="", fmt=ExportFormat.JPEG, overwrite=True):
    preset = ExportPreset(
        name="t",
        export_fmt=fmt,
        output_mode=out_mode,
        output_path=out_path,
        output_subfolder=subfolder,
        overwrite=overwrite,
    )
    return ExportTask(
        file_info={"name": os.path.basename(src_path), "path": src_path, "hash": "h"},
        params=WorkspaceConfig(),
        export_settings=preset,
    )


class TestExportPathResolution(unittest.TestCase):
    def test_dir_same_as_source(self):
        t = _task(os.path.join("photos", "roll", "IMG_1.RAF"), out_mode=ExportPresetOutputMode.SAME_AS_SOURCE)
        self.assertEqual(resolve_export_dir(t), os.path.join("photos", "roll"))

    def test_dir_subfolder_of_source(self):
        t = _task(os.path.join("photos", "roll", "IMG_1.RAF"), out_mode=ExportPresetOutputMode.SUBFOLDER_OF_SOURCE, subfolder="JPEG")
        self.assertEqual(resolve_export_dir(t), os.path.join("photos", "roll", "JPEG"))

    def test_dir_absolute(self):
        t = _task(os.path.join("photos", "roll", "IMG_1.RAF"), out_mode=ExportPresetOutputMode.ABSOLUTE, out_path="out")
        self.assertEqual(resolve_export_dir(t), "out")

    def test_naming_uses_format_ext_and_original_name(self):
        t = _task(os.path.join("d", "IMG_1.RAF"), out_mode=ExportPresetOutputMode.ABSOLUTE, out_path="out", fmt=ExportFormat.TIFF)
        out_dir, filename, ext = resolve_export_naming(t)
        self.assertEqual((out_dir, filename, ext), ("out", "IMG_1", "tiff"))
        self.assertEqual(resolve_export_target_path(t), os.path.join("out", "IMG_1.tiff"))


class TestFindExportConflicts(unittest.TestCase):
    def test_detects_only_existing_targets(self):
        with tempfile.TemporaryDirectory() as d:
            t1 = _task(os.path.join(d, "A.RAF"), out_mode=ExportPresetOutputMode.ABSOLUTE, out_path=d)
            t2 = _task(os.path.join(d, "B.RAF"), out_mode=ExportPresetOutputMode.ABSOLUTE, out_path=d)
            # Pre-create only t1's target on disk.
            with open(resolve_export_target_path(t1), "wb"):
                pass
            self.assertEqual(find_export_conflicts([t1, t2]), [resolve_export_target_path(t1)])

    def test_no_conflicts_when_none_exist(self):
        with tempfile.TemporaryDirectory() as d:
            t = _task(os.path.join(d, "A.RAF"), out_mode=ExportPresetOutputMode.ABSOLUTE, out_path=d)
            self.assertEqual(find_export_conflicts([t]), [])


if __name__ == "__main__":
    unittest.main()
