from dataclasses import dataclass


DEFAULT_PREVIEW_METER_INSET_PERCENT = 10
MAX_PREVIEW_METER_INSET_PERCENT = 30


@dataclass(frozen=True)
class ScannerSettings:
    """Persisted scanner preferences, stored as JSON blob."""

    last_device_id: str = ""
    dpi: int = 3600
    depth: int = 16
    capture_ir: bool = False
    autofocus: bool = True
    samples_per_scan: int = 1
    # Hardware auto-exposure (SANE `ae`). A device-level style preference like
    # autofocus/samples_per_scan above, so it persists the same way.
    auto_exposure: bool = False
    # Validated archival recipe: capture_ir + samples_per_scan=4 together.
    # Persisted like the other device-profile toggles above. Frame number and
    # registered geometry (subframe_mm/br_y_device_px) are deliberately NOT
    # persisted here — they are specific to one physical frame position and
    # silently replaying them onto a different frame/session would be wrong.
    archival_split_capture: bool = False
    output_folder: str = ""
    output_format: str = "TIFF"
    filename_pattern: str = '{{ date }}_{{ "%03d" % seq }}'
    # Explicit hardware profile for an SA-30 or an SA-21 converted to expose
    # the same 40-slot transport. Default-off is safety-critical: a parked
    # feeder reports no capacity, so NegPy must not infer this profile merely
    # from the LS-5000 model. A live positive non-40 capacity remains
    # authoritative over this preference. Appended to preserve the positional
    # order of existing persisted preferences.
    sa30_compatible_roll_feeder: bool = False
    # Display-only brightness metering for roll thumbnails. The full negative
    # remains visible; this percentage is ignored at each edge only while
    # choosing display levels. Appended to preserve positional compatibility.
    preview_meter_inset_percent: int = DEFAULT_PREVIEW_METER_INSET_PERCENT
    # Display-only polarity for the whole-roll contact sheet. The default is a
    # readable positive; enabling this shows the leveled scanner negative.
    # Capture data and final output are unchanged in either mode.
    preview_show_non_inverted_negative: bool = False

    def __post_init__(self) -> None:
        """Normalize persisted settings that must never rely on truthiness.

        JSON numbers and strings are truthy in Python, but neither is an
        explicit confirmation of an SA-30 profile or archival split recipe.
        A valid saved archival split recipe always restores its required
        16-bit depth.

        Preview metering crops accept only JSON integers and clamp them to the
        safe UI range; all other values fall back to the documented default.
        """

        if type(self.sa30_compatible_roll_feeder) is not bool:
            object.__setattr__(self, "sa30_compatible_roll_feeder", False)
        if type(self.archival_split_capture) is not bool:
            object.__setattr__(self, "archival_split_capture", False)
        if self.archival_split_capture is True and self.depth != 16:
            object.__setattr__(self, "depth", 16)
        inset = self.preview_meter_inset_percent
        if type(inset) is not int:
            inset = DEFAULT_PREVIEW_METER_INSET_PERCENT
        else:
            inset = max(0, min(MAX_PREVIEW_METER_INSET_PERCENT, inset))
        object.__setattr__(self, "preview_meter_inset_percent", inset)
        if type(self.preview_show_non_inverted_negative) is not bool:
            object.__setattr__(self, "preview_show_non_inverted_negative", False)

    @classmethod
    def defaults(cls) -> "ScannerSettings":
        return cls()
