"""Custom TorchTitan configuration fields used by the SFU converter."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SFUConfig:
    """Select one exact paper intervention and its matched control."""

    case: Literal["b1", "b2", "b3", "b4", "b5"] = "b1"
    variant: Literal[
        "native",
        "polynomial",
        "pwl2_safe_f16",
        "d2_safe",
    ] = "native"
    strict: bool = True
    """Require the expected model family and number of patched modules."""


@dataclass
class JobConfig:
    """Fields merged into TorchTitan's ``JobConfig`` at parse time."""

    sfu: SFUConfig = field(default_factory=SFUConfig)


__all__ = ["JobConfig", "SFUConfig"]
