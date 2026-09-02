"""Vision Pipeline Subsystem."""
from __future__ import annotations

from vision.pipeline.roi_slicer import ROISlicer, CardROI
from vision.pipeline.state_aggregator import StateAggregator, PlatoGameState, CardTrack
from vision.pipeline.scoreboard_ocr import ScoreboardOCR
from vision.pipeline.telemetry_parser import TelemetryParser, OpponentTelemetry
from vision.pipeline.melds import MeldSolver
from vision.pipeline.fallback_gate import DynamicFallbackGate
from vision.pipeline.cache import PerceptualCache

__all__ = [
    "ROISlicer",
    "CardROI",
    "StateAggregator",
    "PlatoGameState",
    "CardTrack",
    "ScoreboardOCR",
    "TelemetryParser",
    "OpponentTelemetry",
    "MeldSolver",
    "DynamicFallbackGate",
    "PerceptualCache",
]
