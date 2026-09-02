"""TorchTitan adapters for the paper's B1--B5 interventions.

Import :mod:`sfu_repro.torchtitan.plugin` through TorchTitan's
``experimental.custom_import`` setting.  Keeping registration in a separate
module lets this package remain importable without TorchTitan installed.
"""

from .config import JobConfig, SFUConfig

__all__ = ["JobConfig", "SFUConfig"]
