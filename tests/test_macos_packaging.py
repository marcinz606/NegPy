"""Regression tests for metadata written into the macOS application bundle."""

import plistlib

from build import stamp_macos_bundle_version


def test_stamp_macos_bundle_version_sets_both_version_keys(tmp_path):
    app_path = tmp_path / "NegPy.app"
    contents = app_path / "Contents"
    contents.mkdir(parents=True)
    info_path = contents / "Info.plist"
    info_path.write_bytes(
        plistlib.dumps(
            {
                "CFBundleIdentifier": "NegPy",
                "CFBundleShortVersionString": "0.0.0",
            }
        )
    )

    stamp_macos_bundle_version(app_path, "0.37.2", resign=False)

    stamped = plistlib.loads(info_path.read_bytes())
    assert stamped["CFBundleShortVersionString"] == "0.37.2"
    assert stamped["CFBundleVersion"] == "0.37.2"
