"""Portable experiment plumbing for the SFU polynomial paper.

The package initializer is deliberately inert. Optional CUDA, FlashAttention,
Transformers, and evaluation dependencies are imported only by the workflow
that needs them.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sfu-repro")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["__version__"]
