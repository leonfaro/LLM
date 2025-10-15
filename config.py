from __future__ import annotations
import os

# === Simple knobs (edit these numbers if you dislike envs) ===
STT_MODEL = os.environ.get("STT_MODEL", "gpt-4o-mini-transcribe")
STT_LANGUAGE = os.environ.get("STT_LANGUAGE", "en")
STT_ENABLE_DELTA_STREAMING = os.environ.get("STT_ENABLE_DELTA_STREAMING", "1")  # keep SSE on

# VAD + segment shaping (milliseconds / decibels / seconds)
VAD = {
    "ENERGY_CAL_MS": int(os.environ.get("STT_VAD_ENERGY_CAL_MS", "1200")),
    "ENERGY_FLOOR_DBFS": float(os.environ.get("STT_VAD_ENERGY_FLOOR_DBFS", "-60.0")),
    "ENERGY_OFFSET_DB": float(os.environ.get("STT_VAD_ENERGY_OFFSET_DB", "12.0")),
    "MIN_SPEECH_MS": int(os.environ.get("STT_VAD_MIN_SPEECH_MS", "400")),
    "MAX_SILENCE_MS": int(os.environ.get("STT_VAD_MAX_SILENCE_MS", "300")),
    "MAX_SEGMENT_SECONDS": float(os.environ.get("STT_VAD_MAX_SEGMENT_SECONDS", "5.0")),
    "PRE_ROLL_MS": int(os.environ.get("STT_VAD_PRE_ROLL_MS", "220")),
}

# Write env once so downstream .from_env() picks them up predictably.
os.environ.setdefault("STT_MODEL", STT_MODEL)
os.environ.setdefault("STT_LANGUAGE", STT_LANGUAGE)
os.environ.setdefault("STT_ENABLE_DELTA_STREAMING", STT_ENABLE_DELTA_STREAMING)

os.environ.setdefault("STT_VAD_ENERGY_CAL_MS", str(VAD["ENERGY_CAL_MS"]))
os.environ.setdefault("STT_VAD_ENERGY_FLOOR_DBFS", str(VAD["ENERGY_FLOOR_DBFS"]))
os.environ.setdefault("STT_VAD_ENERGY_OFFSET_DB", str(VAD["ENERGY_OFFSET_DB"]))
os.environ.setdefault("STT_VAD_MIN_SPEECH_MS", str(VAD["MIN_SPEECH_MS"]))
os.environ.setdefault("STT_VAD_MAX_SILENCE_MS", str(VAD["MAX_SILENCE_MS"]))
os.environ.setdefault("STT_VAD_MAX_SEGMENT_SECONDS", str(VAD["MAX_SEGMENT_SECONDS"]))
os.environ.setdefault("STT_VAD_PRE_ROLL_MS", str(VAD["PRE_ROLL_MS"]))

# Re-export existing dataclasses and helper so rest of code imports from `config`.
from stt_parameters import SegmentConfig, HTTPSTTConfig, load_http_stt_config  # noqa: E402
