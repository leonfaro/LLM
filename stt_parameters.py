from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

__all__ = ["SegmentConfig", "HTTPSTTConfig", "load_http_stt_config"]


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in {"0", "false", "no", "off"}


DEFAULT_MODEL = "gpt-4o-mini-transcribe"
DEFAULT_ENDPOINT = "https://api.openai.com/v1/audio/transcriptions"
DEFAULT_LANGUAGE = "en"


@dataclass
class SegmentConfig:
    """Lightweight energy VAD used before HTTP transcription."""

    # Calibrate on startup
    energy_calibration_ms: int = 3000
    energy_floor_dbfs: float = -50.0

    # Tighter gate → fewer false positives
    energy_offset_db: float = 12.0

    # Segment shaping
    min_speech_ms: int = 300
    max_silence_ms: int = 200
    max_segment_seconds: float = 5.0
    pre_roll_ms: int = 120

    @classmethod
    def from_env(cls) -> "SegmentConfig":
        d = cls()
        return cls(
            energy_calibration_ms=_env_int("STT_VAD_ENERGY_CAL_MS", d.energy_calibration_ms),
            energy_floor_dbfs=_env_float("STT_VAD_ENERGY_FLOOR_DBFS", d.energy_floor_dbfs),
            energy_offset_db=_env_float("STT_VAD_ENERGY_OFFSET_DB", d.energy_offset_db),
            min_speech_ms=_env_int("STT_VAD_MIN_SPEECH_MS", d.min_speech_ms),
            max_silence_ms=_env_int("STT_VAD_MAX_SILENCE_MS", d.max_silence_ms),
            max_segment_seconds=_env_float("STT_VAD_MAX_SEGMENT_SECONDS", d.max_segment_seconds),
            pre_roll_ms=_env_int("STT_VAD_PRE_ROLL_MS", d.pre_roll_ms),
        )


@dataclass
class HTTPSTTConfig:
    """HTTP transcription parameters."""

    model: str = DEFAULT_MODEL
    language: str = DEFAULT_LANGUAGE
    endpoint: str = DEFAULT_ENDPOINT
    prompt: Optional[str] = field(default_factory=lambda: os.environ.get("STT_PROMPT"))

    # Keep SSE delta streaming ON by default
    enable_delta_streaming: bool = True

    segment: SegmentConfig = field(default_factory=SegmentConfig)

    @classmethod
    def from_env(cls, segment: Optional[SegmentConfig] = None) -> "HTTPSTTConfig":
        seg = segment or SegmentConfig.from_env()
        d = cls(segment=seg)
        return cls(
            model=os.environ.get("STT_MODEL", d.model),
            language=os.environ.get("STT_LANGUAGE", d.language),
            endpoint=os.environ.get("STT_ENDPOINT", d.endpoint),
            prompt=os.environ.get("STT_PROMPT", d.prompt),
            enable_delta_streaming=_env_bool("STT_ENABLE_DELTA_STREAMING", d.enable_delta_streaming),
            segment=seg,
        )


def load_http_stt_config() -> HTTPSTTConfig:
    return HTTPSTTConfig.from_env()
