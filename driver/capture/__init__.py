"""Capture Subsystem."""
from __future__ import annotations

from driver.capture.scrcpy_client import ScrcpyStreamClient
from driver.capture.h264_decoder import PyAVH264Decoder
from driver.capture.synthetic_stream import SyntheticStreamGenerator

__all__ = [
    "ScrcpyStreamClient",
    "PyAVH264Decoder",
    "SyntheticStreamGenerator",
]
