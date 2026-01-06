from dataclasses import dataclass


@dataclass(frozen=True)
class ToningConfig:
    paper_profile: str = "None"
    selenium_strength: float = 0.0
    sepia_strength: float = 0.0
    process_mode: str = "C41"
