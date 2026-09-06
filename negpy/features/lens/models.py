from dataclasses import dataclass


IDENTITY = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class RectilinearWarp:
    coefficients: tuple[tuple[float, ...], ...]
    center: tuple[float, float] = (0.5, 0.5)


@dataclass(frozen=True)
class SonyWarp:
    distortion: tuple[float, ...] = ()
    ca_red: tuple[float, ...] = ()
    ca_blue: tuple[float, ...] = ()


@dataclass(frozen=True)
class LensMetadata:
    source: str = ""
    warps: tuple[RectilinearWarp | SonyWarp, ...] = ()
    reason: str = "No embedded lens correction data."
    # DNG opcodes use the active image, before DefaultCrop and EXIF orientation.
    active_area: tuple[int, int, int, int] | None = None
    buffer_area: tuple[int, int, int, int] | None = None

    @property
    def distortion(self) -> bool:
        return any(
            any(w.distortion) if isinstance(w, SonyWarp) else w.coefficients[0 if len(w.coefficients) == 1 else 1] != IDENTITY
            for w in self.warps
        )

    @property
    def ca(self) -> bool:
        return any(
            any(w.ca_red) or any(w.ca_blue)
            if isinstance(w, SonyWarp)
            else len(w.coefficients) == 3 and (w.coefficients[0] != w.coefficients[1] or w.coefficients[2] != w.coefficients[1])
            for w in self.warps
        )

    @property
    def available(self) -> bool:
        return self.distortion or self.ca

    @property
    def description(self) -> str:
        if not self.available:
            return self.reason
        corrections = " + ".join(label for enabled, label in ((self.distortion, "distortion"), (self.ca, "lateral CA")) if enabled)
        return f"{self.source}: {corrections}"
