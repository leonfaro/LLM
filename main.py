#!/usr/bin/env python3
"""Unified LocalLLM toolkit with GUI and CLI entrypoints."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import queue
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable, Deque, Dict, List, Optional, Sequence, Set, Tuple
import re

from agents import OpenAIConversationsSession
from agent_workflow import WorkflowInput, run_workflow
from config import HTTPSTTConfig, SegmentConfig
from stt_core import (
    SAMPLE_RATE,
    FRAME_MS,
    FRAME_SAMPLES,
    list_input_devices,
    list_output_devices,
    find_loopback_candidate,
    find_monitor_device,
    ensure_blackhole_output,
    ensure_blackhole_input,
    build_captures,
    CaptureConfig,
    AudioMonitor,
    AudioCapture,
    TimedSegment,
    StreamState,
    EnergySegmenter,
    HTTPTranscriber,
    load_openai_api_key,
)


# ---------------------------------------------------------------------------
# Context prompt resource

CONTEXT_PROMPT_FILE = Path(__file__).resolve().parent / "context" / "context_agent_prompt.txt"



# ---------------------------------------------------------------------------
# LLM router (from llm_router.py)

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]


@dataclass
class LLMConfig:
    openai_model: str = os.environ.get("LLM_OPENAI_MODEL", os.environ.get("OPENAI_MODEL", "gpt-5"))
    openai_api_key: Optional[str] = load_openai_api_key()
    openai_base_url: Optional[str] = os.environ.get("OPENAI_BASE_URL")
    openai_timeout_s: float = float(os.environ.get("OPENAI_TIMEOUT_S", "120"))
    reasoning_effort: Optional[str] = os.environ.get("LLM_REASONING_EFFORT", "low")


class LLMRouter:
    def __init__(self, cfg: Optional[LLMConfig] = None):
        self.cfg = cfg or LLMConfig()
        self._openai_client: Optional["OpenAI"] = None
        self._conversation_id: Optional[str] = None
        self._last_response_id: Optional[str] = None

    def query(
        self,
        prompt: str,
        model: Optional[str] = None,
        on_stream: Optional[Callable[[str], None]] = None,
    ) -> str:
        return self._query_openai(prompt, model or self.cfg.openai_model, on_stream=on_stream)

    @property
    def supports_streaming(self) -> bool:
        return True

    def _ensure_openai_client(self) -> "OpenAI":
        if self._openai_client is not None:
            return self._openai_client
        if OpenAI is None:
            raise RuntimeError("openai backend requested but the 'openai' package is not installed.")
        if not self.cfg.openai_api_key:
            raise RuntimeError("openai backend requested but OPENAI_API_KEY is not set (or openai_api_key.txt is missing).")
        client = OpenAI(api_key=self.cfg.openai_api_key, base_url=self.cfg.openai_base_url)
        try:
            conv = client.conversations.create()
            self._conversation_id = getattr(conv, "id", None)
        except Exception:
            self._conversation_id = None
        self._openai_client = client
        return client

    def _query_openai(
        self,
        prompt: str,
        model: str,
        *,
        on_stream: Optional[Callable[[str], None]] = None,
    ) -> str:
        client = self._ensure_openai_client()

        stream_output: List[str] = []
        timeout = self.cfg.openai_timeout_s
        reasoning_payload = (
            {"effort": self.cfg.reasoning_effort}
            if self.cfg.reasoning_effort
            else None
        )

        if on_stream:
            try:
                stream_kwargs: Dict[str, object] = {
                    "model": model,
                    "input": [{"role": "user", "content": prompt}],
                    "timeout": timeout,
                }
                if reasoning_payload:
                    stream_kwargs["reasoning"] = reasoning_payload
                if self._conversation_id:
                    stream_kwargs["conversation"] = self._conversation_id
                if self._last_response_id:
                    stream_kwargs["previous_response_id"] = self._last_response_id
                with client.responses.stream(**stream_kwargs) as stream:
                    for event in stream:
                        etype = getattr(event, "type", "")
                        if etype == "response.output_text.delta":
                            delta = getattr(event, "delta", None)
                            if delta:
                                text = str(delta)
                                stream_output.append(text)
                                on_stream(text)
                        elif etype == "response.output_text.done":
                            continue
                        elif etype == "response.completed":
                            break
                    final = stream.get_final_response()
                    if final is not None:
                        self._last_response_id = getattr(final, "id", None)
            except Exception as exc:  # pragma: no cover - network failure
                raise RuntimeError(f"OpenAI streaming failed: {exc}") from exc
            if stream_output:
                return "".join(stream_output).strip()
            return _extract_response_text(final)

        try:
            create_kwargs: Dict[str, object] = {
                "model": model,
                "input": [{"role": "user", "content": prompt}],
                "timeout": timeout,
            }
            if reasoning_payload:
                create_kwargs["reasoning"] = reasoning_payload
            if self._conversation_id:
                create_kwargs["conversation"] = self._conversation_id
            if self._last_response_id:
                create_kwargs["previous_response_id"] = self._last_response_id
            response = client.responses.create(**create_kwargs)
            if response is not None:
                self._last_response_id = getattr(response, "id", None)
        except Exception as exc:  # pragma: no cover - network failure
            raise RuntimeError(f"OpenAI completion failed: {exc}") from exc

        return _extract_response_text(response)


def _extract_response_text(response) -> str:
    """
    Best-effort extraction of text from the OpenAI Responses API result.
    Structure may vary; this keeps it defensive.
    """
    if response is None:
        return ""
    try:
        text = getattr(response, "output_text", None)
        if callable(text):
            return str(text()).strip()
        if isinstance(text, str):
            return text.strip()
    except Exception:
        pass

    try:
        content = getattr(response, "content", None)
        if content:
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(item.get("text") or "")
                else:
                    parts.append(str(getattr(item, "text", "")))
            text = " ".join(p.strip() for p in parts if p)
            if text:
                return text.strip()
    except Exception:
        pass

    return str(response).strip()


def build_prompt(context: Optional[str], transcript: str) -> str:
    context = (context or "").strip()
    parts = []
    if context:
        parts.append("Context:\n" + context)
    parts.append("Transcript:\n" + transcript)
    parts.append("\nTask: Provide a concise, helpful response.")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Tk application (from dual_stt_app.py)


def ensure_repo_venv() -> None:
    if os.environ.get("LOCAL_LLM_SKIP_VENV") == "1":
        return

    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    venv_python = repo_root / ".venv" / "bin" / "python"

    try:
        current = Path(sys.executable).resolve()
    except FileNotFoundError:
        current = Path(sys.executable)

    if venv_python.exists() and current != venv_python.resolve():
        env = os.environ.copy()
        env["LOCAL_LLM_SKIP_VENV"] = "1"
        os.execve(str(venv_python), [str(venv_python), str(script_path), *sys.argv[1:]], env)


class TkLogHandler(logging.Handler):
    def __init__(self, callback: Callable[[str], None]):
        super().__init__()
        self.callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.callback(msg)
        except Exception:
            self.handleError(record)


class App(tk.Tk):
    @staticmethod
    def _env_flag(name: str, default: bool = True) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _preferred_mic_device(devices):
        if not devices:
            return None
        preferred = None
        fallback = devices[0]
        for device in devices:
            name_low = device.get("name", "").lower()
            if any(bad in name_low for bad in ("blackhole", "loopback", "soundflower")):
                continue
            if "macbook" in name_low and "microphone" in name_low:
                return device
            if preferred is None and "microphone" in name_low:
                preferred = device
        return preferred or fallback

    def __init__(self):
        super().__init__()
        self.title("Dual STT Tuner (Mic & System)")
        self.geometry("1100x700")
        # --- Session context directory ---
        default_ctx = Path(__file__).resolve().parent / "context"
        env_ctx = os.environ.get("LOCAL_LLM_CONTEXT_DIR")
        self.context_dir = Path(env_ctx).expanduser() if env_ctx else default_ctx
        self.session_context: Dict[str, str] = {}
        self._context_max_chars = 8_000

        self.cfg_mic = CaptureConfig()
        self.cfg_sys = CaptureConfig()
        self._queue_maxsize = 16
        self.q: Optional[queue.Queue] = None
        self.tx_mic: Optional[HTTPTranscriber] = None
        self.tx_sys: Optional[HTTPTranscriber] = None
        self._frame_dispatch_thread: Optional[threading.Thread] = None
        self._frame_dispatch_stop: Optional[threading.Event] = None
        self.cap_mic: Optional[AudioCapture] = None
        self.cap_sys: Optional[AudioCapture] = None
        self.sys_monitor: Optional[AudioMonitor] = None
        self.log_queue: "queue.Queue[str]" = queue.Queue(maxsize=256)
        self._log_handler: Optional[TkLogHandler] = None
        self._loopback_device_idx: Optional[int] = None
        self._loopback_name: Optional[str] = None
        self._calibration_after_id = None

        self.disable_mic = tk.BooleanVar(value=False)
        self.disable_sys = tk.BooleanVar(value=False)
        self._speaker_alias: Dict[str, str] = {"MIC": "Leon", "SYS": "Interviewer"}
        self._speaker_segment_maxlen = 128
        self._merged_segments_maxlen = 512
        self._stream_state: Dict[str, StreamState] = {
            label: StreamState(speaker=speaker) for label, speaker in self._speaker_alias.items()
        }
        self._speaker_live_text: Dict[str, str] = {alias: "" for alias in self._speaker_alias.values()}
        self._speaker_history: Dict[str, List[str]] = {alias: [] for alias in self._speaker_alias.values()}
        self._speaker_partial: Dict[str, str] = {alias: "" for alias in self._speaker_alias.values()}
        self._speaker_segments: Dict[str, Deque[TimedSegment]] = {
            alias: deque(maxlen=self._speaker_segment_maxlen) for alias in self._speaker_alias.values()
        }
        self._merged_segments: Deque[TimedSegment] = deque(maxlen=self._merged_segments_maxlen)
        self._help_anchor: float = 0.0
        self._help_slice_lock = threading.Lock()
        self.conversation_log: List[str] = []
        self.conversation_text: Optional[ScrolledText] = None
        self.live_text_widgets: Dict[str, ScrolledText] = {}
        self._conversation_entries: List[Dict[str, object]] = []
        self.agent_entry: Optional[ScrolledText] = None
        self.agent_send_button: Optional[ttk.Button] = None
        self.agent_screenshot_button: Optional[ttk.Button] = None
        # Stage UI and state
        self.agent1_stage_box: Optional[ScrolledText] = None
        self.agent2_stage_box: Optional[ScrolledText] = None
        self._latest_stage_outputs: Dict[str, str] = {"Agent1": "", "Agent2": ""}
        self._agent_thread: Optional[threading.Thread] = None
        self._agent_running = False
        self.pending_manual_notes: List[str] = []
        self.pending_screenshots: List[str] = []
        self.session_log_path: Optional[Path] = None
        self._session_logged_keys: Set[Tuple[float, float, str, str]] = set()
        self._session_flush_after: Optional[str] = None
        self.agent_session = OpenAIConversationsSession()

        sys_model_default = "gpt-4o-mini-transcribe"
        mic_model_default = "gpt-4o-mini-transcribe"
        sys_model = os.environ.get("LOCAL_LLM_SYS_MODEL", sys_model_default)
        mic_model = os.environ.get("LOCAL_LLM_MIC_MODEL", mic_model_default)
        sys_segment_cfg = SegmentConfig.from_env()
        sys_base_cfg = HTTPSTTConfig.from_env(segment=sys_segment_cfg)
        self._http_cfg_sys = replace(sys_base_cfg, model=sys_model, language="en")
        mic_segment_cfg = SegmentConfig.from_env()
        mic_base_cfg = HTTPSTTConfig.from_env(segment=mic_segment_cfg)
        mic_streaming = mic_base_cfg.enable_delta_streaming and not mic_model.lower().startswith("whisper")
        self._http_cfg_mic = replace(
            mic_base_cfg,
            model=mic_model,
            language="en",
            enable_delta_streaming=mic_streaming,
        )
        # Default per-stream segment tuning
        mic_defaults = SegmentConfig(
            energy_calibration_ms=3000,
            energy_floor_dbfs=-50.0,
            energy_offset_db=12.0,
            min_speech_ms=300,
            max_silence_ms=200,
            max_segment_seconds=5.0,
            pre_roll_ms=120,
        )
        sys_defaults = SegmentConfig(
            energy_calibration_ms=1200,
            energy_floor_dbfs=-60.0,
            energy_offset_db=0.0,
            min_speech_ms=120,
            max_silence_ms=120,
            max_segment_seconds=1.0,
            pre_roll_ms=100,
        )
        self._http_cfg_mic = replace(self._http_cfg_mic, segment=mic_defaults)
        self._http_cfg_sys = replace(self._http_cfg_sys, segment=sys_defaults)
        # Editable segment parameters tracked per stream
        def _segment_vars_from_cfg(cfg: SegmentConfig) -> Dict[str, tk.StringVar]:
            return {
                "energy_calibration_ms": tk.StringVar(value=str(cfg.energy_calibration_ms)),
                "energy_floor_dbfs": tk.StringVar(value=str(cfg.energy_floor_dbfs)),
                "energy_offset_db": tk.StringVar(value=str(cfg.energy_offset_db)),
                "min_speech_ms": tk.StringVar(value=str(cfg.min_speech_ms)),
                "max_silence_ms": tk.StringVar(value=str(cfg.max_silence_ms)),
                "max_segment_seconds": tk.StringVar(value=str(cfg.max_segment_seconds)),
                "pre_roll_ms": tk.StringVar(value=str(cfg.pre_roll_ms)),
            }

        self._segment_vars: Dict[str, Dict[str, tk.StringVar]] = {
            "mic": _segment_vars_from_cfg(self._http_cfg_mic.segment),
            "sys": _segment_vars_from_cfg(self._http_cfg_sys.segment),
        }

        self.monitor_var = tk.StringVar()
        self.monitor_combo: Optional[ttk.Combobox] = None
        self._monitor_devices: Dict[str, Tuple[int, str, int]] = {}

        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True)

        self.controls_frame = ttk.Frame(body)
        self.controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=6)

        self.content_frame = ttk.Frame(body)
        self.content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=6)

        self._build_controls(self.controls_frame)
        self._build_textareas(self.content_frame)
        self._refresh_devices()
        self._setup_logging_bridge()
        self.after(200, self._drain_logs)

    def _build_controls(self, parent):
        devices_frame = ttk.LabelFrame(parent, text="Devices")
        devices_frame.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(devices_frame, text="Mic input").grid(row=0, column=0, sticky="w")
        self.mic_var = tk.StringVar()
        self.mic_combo = ttk.Combobox(devices_frame, width=38, textvariable=self.mic_var, state="readonly")
        self.mic_combo.grid(row=1, column=0, sticky="we", pady=(0, 6))

        ttk.Label(devices_frame, text="System loopback").grid(row=2, column=0, sticky="w")
        self.sys_var = tk.StringVar()
        self.sys_combo = ttk.Combobox(devices_frame, width=38, textvariable=self.sys_var, state="readonly")
        self.sys_combo.grid(row=3, column=0, sticky="we")
        self.sys_combo.bind("<<ComboboxSelected>>", self._on_sys_selected)

        ttk.Label(devices_frame, text="Monitor output").grid(row=4, column=0, sticky="w", pady=(6, 0))
        self.monitor_combo = ttk.Combobox(devices_frame, width=38, textvariable=self.monitor_var, state="readonly")
        self.monitor_combo.grid(row=5, column=0, sticky="we")
        self.monitor_combo.bind("<<ComboboxSelected>>", self._on_monitor_selected)

        devices_frame.grid_columnconfigure(0, weight=1)

        refresh_btn = ttk.Button(devices_frame, text="Refresh devices", command=self._refresh_devices)
        refresh_btn.grid(row=6, column=0, sticky="we", pady=(6, 0))
        options_frame = ttk.LabelFrame(parent, text="Options")
        options_frame.pack(fill=tk.X, pady=(0, 6))

        mic_off = ttk.Checkbutton(
            options_frame, text="Disable mic capture", variable=self.disable_mic, command=self._on_capture_toggle
        )
        mic_off.grid(row=0, column=0, sticky="w")

        sys_off = ttk.Checkbutton(
            options_frame,
            text="Disable system capture",
            variable=self.disable_sys,
            command=self._on_capture_toggle,
        )
        sys_off.grid(row=1, column=0, sticky="w", pady=(6, 0))

        gate_frame = ttk.LabelFrame(parent, text="Energy Gate")
        gate_frame.pack(fill=tk.X, pady=(0, 6))
        gate_frame.columnconfigure(1, weight=1)
        gate_frame.columnconfigure(2, weight=1)
        ttk.Label(gate_frame, text="Parameter").grid(row=0, column=0, sticky="w")
        ttk.Label(gate_frame, text="Input (Mic)").grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(gate_frame, text="Output (System)").grid(row=0, column=2, sticky="w", padx=(8, 0))
        energy_fields = [
            ("Calibration (ms)", "energy_calibration_ms"),
            ("Floor (dBFS)", "energy_floor_dbfs"),
            ("Offset (dB)", "energy_offset_db"),
            ("Min speech (ms)", "min_speech_ms"),
            ("Max silence (ms)", "max_silence_ms"),
            ("Max segment (s)", "max_segment_seconds"),
            ("Pre-roll (ms)", "pre_roll_ms"),
        ]
        for row, (label, key) in enumerate(energy_fields, start=1):
            ttk.Label(gate_frame, text=label).grid(row=row, column=0, sticky="w", pady=(0, 2))
            mic_entry = ttk.Entry(gate_frame, textvariable=self._segment_vars["mic"][key], width=10)
            mic_entry.grid(row=row, column=1, sticky="we", padx=(8, 4), pady=(0, 2))
            sys_entry = ttk.Entry(gate_frame, textvariable=self._segment_vars["sys"][key], width=10)
            sys_entry.grid(row=row, column=2, sticky="we", padx=(8, 0), pady=(0, 2))

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=(6, 0))

        new_session_btn = ttk.Button(btn_frame, text="New Session", command=self.start_new_session)
        new_session_btn.pack(fill=tk.X, pady=(0, 4))
        grab_ctx_btn = ttk.Button(btn_frame, text="Grab context", command=self.grab_context)
        grab_ctx_btn.pack(fill=tk.X, pady=(0, 4))

        listen_btn = ttk.Button(btn_frame, text="Listen", command=self.toggle_listen)
        listen_btn.pack(fill=tk.X, pady=(0, 4))

        clear_input_btn = ttk.Button(btn_frame, text="Clear Input", command=self.clear_input_transcript)
        clear_input_btn.pack(fill=tk.X, pady=(0, 4))
        clear_output_btn = ttk.Button(btn_frame, text="Clear Output", command=self.clear_output_transcript)
        clear_output_btn.pack(fill=tk.X, pady=(0, 4))
        clear_all_btn = ttk.Button(btn_frame, text="Clear All", command=self.clear_all)
        clear_all_btn.pack(fill=tk.X, pady=(0, 4))
        self.help_btn = ttk.Button(btn_frame, text="Help!", command=self.on_help)
        self.help_btn.pack(fill=tk.X, pady=(0, 4))

        self.status = tk.StringVar(value="Idle")
        status_lbl = ttk.Label(btn_frame, textvariable=self.status)
        status_lbl.pack(anchor="e", pady=(8, 0))

    def _build_textareas(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=3)
        parent.rowconfigure(1, weight=3)
        parent.rowconfigure(2, weight=1)

        live_container = ttk.Labelframe(parent, text="Live Transcripts")
        live_container.grid(row=0, column=0, sticky="nsew")
        self._build_live_boxes(live_container)

        convo_container = ttk.Labelframe(parent, text="Conversation")
        convo_container.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        self._build_conversation_panel(convo_container)

        log_frame = ttk.Frame(parent)
        log_frame.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        ttk.Label(log_frame, text="Logs").pack(anchor="w")
        self.log_text = ScrolledText(log_frame, wrap="word", height=6)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _build_live_boxes(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

        mappings = (("MIC", "Input — Leon"), ("SYS", "Output — Interviewer"))
        for col, (label, title) in enumerate(mappings):
            frame = ttk.Labelframe(parent, text=title)
            frame.grid(row=0, column=col, sticky="nsew", padx=4, pady=4)
            box = ScrolledText(frame, wrap="word", height=8)
            box.pack(fill=tk.BOTH, expand=True)
            speaker = self._speaker_alias.get(label, label)
            self.live_text_widgets[speaker] = box
            self._write_text(box, "", newline=False)

    def _build_conversation_panel(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)   # Agent responses
        parent.rowconfigure(1, weight=3)   # Conversation
        parent.rowconfigure(2, weight=1)   # Input

        agents_frame = ttk.Labelframe(parent, text="Agent Responses")
        agents_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        agents_frame.columnconfigure(0, weight=1)
        agents_frame.columnconfigure(1, weight=1)
        agents_frame.rowconfigure(1, weight=1)
        ttk.Label(agents_frame, text="Agent1 Response", anchor="w").grid(row=0, column=0, sticky="w")
        ttk.Label(agents_frame, text="Agent2 Response", anchor="w").grid(row=0, column=1, sticky="w")
        a1 = ScrolledText(agents_frame, wrap="word", height=4)
        a1.grid(row=1, column=0, sticky="nsew", padx=(0, 2))
        a2 = ScrolledText(agents_frame, wrap="word", height=4)
        a2.grid(row=1, column=1, sticky="nsew", padx=(2, 0))
        self.agent1_stage_box = a1
        self.agent2_stage_box = a2
        self._write_text(a1, "", newline=False)
        self._write_text(a2, "", newline=False)

        text = ScrolledText(parent, wrap="word", height=8)
        text.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self.conversation_text = text
        self._write_text(text, "", newline=False)

        input_frame = ttk.Frame(parent)
        input_frame.grid(row=2, column=0, sticky="nsew", padx=4, pady=(0, 4))
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(0, weight=1)

        entry = ScrolledText(input_frame, wrap="word", height=3)
        entry.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.agent_entry = entry

        screenshot_btn = ttk.Button(input_frame, text="Screenshot", command=self._capture_screenshot)
        screenshot_btn.grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.agent_screenshot_button = screenshot_btn

        send_btn = ttk.Button(input_frame, text="Send", command=self._send_agent_message)
        send_btn.grid(row=1, column=1, sticky="e", padx=(6, 0), pady=(6, 0))
        self.agent_send_button = send_btn

    def _apply_segment_settings(self) -> None:
        vars_map = getattr(self, "_segment_vars", None)
        if not vars_map:
            return

        field_specs = [
            ("energy_calibration_ms", int, 0),
            ("energy_floor_dbfs", float, None),
            ("energy_offset_db", float, None),
            ("min_speech_ms", int, 0),
            ("max_silence_ms", int, 0),
            ("max_segment_seconds", float, 0.1),
            ("pre_roll_ms", int, 0),
        ]

        def _build_segment(var_dict: Dict[str, tk.StringVar], current: SegmentConfig) -> SegmentConfig:
            cleaned: Dict[str, object] = {}
            for key, caster, min_value in field_specs:
                current_default = getattr(current, key)
                var = var_dict.get(key)
                if var is None:
                    cleaned[key] = current_default
                    continue
                raw = var.get().strip()
                if not raw:
                    value = current_default
                else:
                    try:
                        value = caster(raw)
                    except (ValueError, TypeError):
                        value = current_default
                if isinstance(value, (int, float)) and min_value is not None and value < min_value:
                    value = caster(min_value)
                if isinstance(value, int) and value < 0:
                    value = 0
                var.set(str(value))
                cleaned[key] = value

            return SegmentConfig(
                energy_calibration_ms=int(cleaned.get("energy_calibration_ms", current.energy_calibration_ms)),
                energy_floor_dbfs=float(cleaned.get("energy_floor_dbfs", current.energy_floor_dbfs)),
                energy_offset_db=float(cleaned.get("energy_offset_db", current.energy_offset_db)),
                min_speech_ms=int(cleaned.get("min_speech_ms", current.min_speech_ms)),
                max_silence_ms=int(cleaned.get("max_silence_ms", current.max_silence_ms)),
                max_segment_seconds=float(cleaned.get("max_segment_seconds", current.max_segment_seconds)),
                pre_roll_ms=int(cleaned.get("pre_roll_ms", current.pre_roll_ms)),
            )

        mic_vars = vars_map.get("mic", {})
        sys_vars = vars_map.get("sys", {})
        mic_segment = _build_segment(mic_vars, self._http_cfg_mic.segment)
        sys_segment = _build_segment(sys_vars, self._http_cfg_sys.segment)

        self._http_cfg_mic = replace(self._http_cfg_mic, segment=mic_segment)
        self._http_cfg_sys = replace(self._http_cfg_sys, segment=sys_segment)

    def _refresh_live_box(self, speaker: str):
        widget = self.live_text_widgets.get(speaker)
        if widget is None:
            return
        history = self._speaker_history.get(speaker, [])
        partial = self._speaker_partial.get(speaker, "")
        lines = list(history)
        if partial:
            lines.append(partial)
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        if lines:
            widget.insert(tk.END, "\n".join(lines))
        widget.see(tk.END)
        widget.configure(state=tk.DISABLED)

    def _refresh_all_live_boxes(self):
        for speaker in self.live_text_widgets:
            self._refresh_live_box(speaker)

    def _read_text_file(self, path: Path) -> Optional[str]:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            return text if text.strip() else None
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"Failed to read context file {path.name}: {exc}")
            return None

    def _load_context_prompt(self) -> Optional[str]:
        try:
            text = CONTEXT_PROMPT_FILE.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            self._append_log(f"Context prompt file not found: {CONTEXT_PROMPT_FILE}")
            return None
        except OSError as exc:
            self._append_log(f"Failed to read context prompt: {exc}")
            return None

        if not text:
            self._append_log("Context prompt file is empty.")
            return None
        return text

    def _load_context_buckets(self, directory: Path) -> Dict[str, str]:
        files = sorted({p.resolve() for p in directory.rglob("*") if p.is_file()})
        buckets = {"cv_text": [], "prior_interview_notes": [], "constraints_or_prefs": []}
        totals = {"cv_text": 0, "prior_interview_notes": 0, "constraints_or_prefs": 0}

        def _bucket_for(name: str) -> str:
            n = name.lower()
            if any(key in n for key in ("cv", "resume", "lebenslauf", "extended-cv", "extended_cv")):
                return "cv_text"
            if any(key in n for key in ("constraint", "preference")):
                return "constraints_or_prefs"
            if any(key in n for key in ("interview", "transcrib", "transcript", "gespraech", "gespräch")):
                return "prior_interview_notes"
            return "prior_interview_notes"

        for path in files:
            text = self._read_text_file(path)
            if not text:
                continue
            bucket = _bucket_for(path.name)
            remaining = self._context_max_chars - totals[bucket]
            if remaining <= 0:
                continue
            if len(text) > remaining:
                text = text[:remaining]
            buckets[bucket].append(text)
            totals[bucket] += len(text)

        return {k: "\n\n".join(v).strip() for k, v in buckets.items() if v}

    def grab_context(self) -> None:
        directory = self.context_dir
        if not directory.exists():
            self._append_log(f"Context dir not found: {directory}")
            return
        if self._agent_running:
            self._append_log("Agent busy; wait for the current reply before sending context.")
            return

        prompt_text = self._load_context_prompt()
        if not prompt_text:
            return

        self._append_log(f"Preparing context package from {directory}")
        try:
            ctx = self._load_context_buckets(directory)
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"Context load failed: {exc}")
            return
        self.session_context = dict(ctx)
        cv_len = len(self.session_context.get("cv_text", ""))
        notes_len = len(self.session_context.get("prior_interview_notes", ""))
        cons_len = len(self.session_context.get("constraints_or_prefs", ""))
        self._append_log(
            f"Context loaded (cv={cv_len} chars, notes={notes_len} chars, constraints={cons_len} chars)"
        )

        all_files = sorted(p for p in directory.rglob("*") if p.is_file())
        entries: List[str] = []
        for path in all_files:
            text = self._read_text_file(path)
            if not text:
                continue
            try:
                rel_path = path.relative_to(directory)
            except ValueError:
                rel_path = path
            cleaned = text.strip()
            if not cleaned:
                continue
            entries.append(f"### {rel_path}\n{cleaned}")

        if not entries:
            self._append_log("No readable context files found to send.")
            return

        payload_sections: List[str] = [prompt_text.strip(), *entries]
        payload_text = "\n\n".join(section for section in payload_sections if section.strip()).strip()
        if not payload_text:
            self._append_log("Context payload empty; nothing sent.")
            return

        self._append_log(f"Sending {len(entries)} context files to agent.")
        self._run_agent_async(payload_text, screenshots=None)
        self.status.set("Context sent to agent.")

    def _append_conversation_entry(self, segments: Sequence[TimedSegment], formatted_text: str):
        formatted_text = (formatted_text or "").strip()
        if not formatted_text:
            return

        entry = {
            "timestamp": time.time(),
            "text": formatted_text,
            "segments": [
                {
                    "speaker": seg.speaker,
                    "text": seg.text,
                    "started_at": seg.started_at,
                    "ended_at": seg.ended_at,
                    "reason": seg.reason,
                }
                for seg in segments
            ],
        }
        self._conversation_entries.append(entry)
        self.conversation_log.append(formatted_text)

        widget = self.conversation_text
        if widget is not None:
            widget.configure(state=tk.NORMAL)
            widget.insert(tk.END, formatted_text + "\n\n")
            widget.configure(state=tk.DISABLED)
            widget.see(tk.END)

    def _append_conversation_line(self, label: str, text: str) -> None:
        cleaned = str(text or "").strip()
        if not cleaned:
            return
        stamp = time.strftime("%H:%M:%S", time.localtime())
        entry_line = f"[{stamp}] {label}: {cleaned}"
        self.conversation_log.append(entry_line)
        if self.conversation_text is not None:
            widget = self.conversation_text
            widget.configure(state=tk.NORMAL)
            widget.insert(tk.END, entry_line + "\n\n")
            widget.configure(state=tk.DISABLED)
            widget.see(tk.END)
        self._conversation_entries.append({"timestamp": time.time(), "text": entry_line, "segments": []})

    def _clear_transcript(self, label: str, log: bool = True):
        speaker = self._speaker_alias.get(label, label)
        self._speaker_live_text[speaker] = ""
        if speaker in self._speaker_segments:
            self._speaker_segments[speaker].clear()
        if label in self._stream_state:
            state = self._stream_state[label]
            state.current_start = None
            state.last_update = None
        self._speaker_history[speaker] = []
        self._speaker_partial[speaker] = ""
        self._merged_segments = deque(
            (seg for seg in self._merged_segments if seg.label != label),
            maxlen=self._merged_segments_maxlen,
        )
        self._help_anchor = self._merged_segments[-1].ended_at if self._merged_segments else 0.0
        self._refresh_live_box(speaker)
        if log:
            label_name = "Mic" if label == "MIC" else "System"
            self._append_log(f"{label_name} transcript cleared")

    def clear_input_transcript(self):
        self._clear_transcript("MIC")

    def clear_output_transcript(self):
        self._clear_transcript("SYS")

    def clear_conversation(self, log: bool = True):
        self.conversation_log.clear()
        self._conversation_entries.clear()
        self._help_anchor = self._merged_segments[-1].ended_at if self._merged_segments else 0.0
        if self.conversation_text is not None:
            self._write_text(self.conversation_text, "", newline=False)
        if self.agent_entry is not None:
            self.agent_entry.delete("1.0", tk.END)
        if self.agent_send_button is not None:
            self.agent_send_button.configure(state=tk.NORMAL)
        self.pending_manual_notes.clear()
        self.pending_screenshots.clear()
        self._agent_running = False
        self._agent_thread = None
        self.agent_session = OpenAIConversationsSession()
        self.session_context.clear()
        if log:
            self._append_log("Conversation log cleared")

    def clear_all(self):
        self._clear_transcript("MIC", log=False)
        self._clear_transcript("SYS", log=False)
        self.clear_conversation(log=False)
        self._append_log("All transcripts and conversation cleared")
        self.agent_session = OpenAIConversationsSession()
        self.session_context.clear()
        self._append_log("Session context cleared")

    def clear_texts(self):
        self._clear_transcript("MIC", log=False)
        self._clear_transcript("SYS", log=False)
        self._append_log("Transcripts cleared")

    @staticmethod
    def _segment_key(seg: TimedSegment) -> Tuple[float, float, str, str]:
        return (seg.started_at, seg.ended_at, seg.label, seg.text)

    def start_new_session(self):
        self._cancel_session_logging()
        now = datetime.now()
        base_name = now.strftime("%Y%m%d_%H%M_transcription")
        root = Path(__file__).resolve().parent
        path = root / f"{base_name}.txt"
        counter = 1
        while path.exists():
            path = root / f"{base_name}_{counter}.txt"
            counter += 1
        try:
            with path.open("w", encoding="utf-8") as handle:
                handle.write(f"Session started at {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        except OSError as exc:
            self._append_log(f"Failed to create session log: {exc}")
            self.session_log_path = None
            self._session_logged_keys.clear()
            return

        self.session_log_path = path
        self._session_logged_keys = {self._segment_key(seg) for seg in self._merged_segments}
        self._append_log(f"Session log created: {path.name}")
        self._schedule_session_flush()
        self.agent_session = OpenAIConversationsSession()
        self.session_context.clear()
        self._append_log("Session context cleared")

    def _cancel_session_logging(self):
        if self._session_flush_after is not None:
            try:
                self.after_cancel(self._session_flush_after)
            except Exception:
                pass
            self._session_flush_after = None

    def _schedule_session_flush(self):
        if self.session_log_path is None:
            return
        self._session_flush_after = self.after(1000, self._flush_session_segments)

    def _flush_session_segments(self):
        self._session_flush_after = None
        if self.session_log_path is None:
            return

        segments = sorted(self._merged_segments, key=lambda seg: seg.started_at)
        new_lines: List[str] = []
        for seg in segments:
            key = self._segment_key(seg)
            if key in self._session_logged_keys:
                continue
            english_text = str(seg.text or "").strip()
            if not english_text:
                self._session_logged_keys.add(key)
                continue
            speaker = "Interviewee" if seg.label == "MIC" else "Interviewer"
            timestamp = time.strftime("%H:%M:%S", time.localtime(seg.started_at))
            new_lines.append(f"[{timestamp}] {speaker}: {english_text}")
            self._session_logged_keys.add(key)

        if new_lines:
            try:
                with self.session_log_path.open("a", encoding="utf-8") as handle:
                    handle.write("\n".join(new_lines) + "\n")
            except OSError as exc:
                self._append_log(f"Failed to append to session log: {exc}")

        self._schedule_session_flush()

    def _refresh_devices(self):
        devs = list_input_devices()
        display = [f'#{d["index"]}: {d["name"]}' for d in devs]
        self.mic_combo["values"] = display
        if display:
            best_device = self._preferred_mic_device(devs)
            if best_device:
                label = f'#{best_device["index"]}: {best_device["name"]}'
                try:
                    idx = display.index(label)
                except ValueError:
                    idx = 0
            else:
                idx = 0
            self.mic_combo.current(idx)
            self.mic_var.set(display[idx])

        self._loopback_device_idx = None
        self._loopback_name = None
        loopback_labels = []
        tokens = ("blackhole", "loopback", "soundflower")
        for dev in devs:
            name_low = dev["name"].lower()
            if any(token in name_low for token in tokens):
                loopback_labels.append(f'#{dev["index"]}: {dev["name"]}')

        if not loopback_labels:
            if ensure_blackhole_input():
                devs = list_input_devices()
                loopback_labels = [
                    f'#{d["index"]}: {d["name"]}'
                    for d in devs
                    if any(token in d["name"].lower() for token in tokens)
                ]

        if loopback_labels:
            current = self.sys_var.get()
            self.sys_combo["values"] = loopback_labels
            if current in loopback_labels:
                selection = current
            else:
                selection = loopback_labels[0]
            self.sys_combo.set(selection)
            self._loopback_device_idx = self._parse_idx(selection)
            self._loopback_name = selection.split(": ", 1)[1]
            self._append_log(f"Loopback device set to {self._loopback_name}")
        else:
            placeholder = "No BlackHole loopback detected"
            self.sys_combo["values"] = [placeholder]
            self.sys_combo.set(placeholder)
            self._loopback_name = None
            self._append_log("No system loopback device detected")

        outputs = list_output_devices()
        monitor_labels: List[str] = []
        self._monitor_devices.clear()
        env_monitor = os.environ.get("LOCAL_LLM_MONITOR_DEVICE", "").strip().lower()
        env_match: Optional[str] = None
        for dev in outputs:
            label = f'#{dev["index"]}: {dev["name"]}'
            monitor_labels.append(label)
            channels = max(1, min(2, int(dev.get("channels", 1))))
            self._monitor_devices[label] = (dev["index"], dev["name"], channels)
            if env_monitor and env_monitor in dev["name"].lower():
                env_match = label

        if monitor_labels:
            existing = self.monitor_var.get()
            if self.monitor_combo is not None:
                self.monitor_combo["values"] = monitor_labels
            if existing in monitor_labels:
                selection = existing
            elif env_match:
                selection = env_match
            else:
                selection = monitor_labels[0]
            self.monitor_var.set(selection)
            if self.monitor_combo is not None:
                self.monitor_combo.set(selection)
        else:
            placeholder = "No output device detected"
            if self.monitor_combo is not None:
                self.monitor_combo["values"] = [placeholder]
                self.monitor_combo.set(placeholder)
            self.monitor_var.set("")

        self._on_capture_toggle()

    def _setup_logging_bridge(self):
        logger = logging.getLogger("dual_stt")
        handler = TkLogHandler(self._queue_log)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger.addHandler(handler)
        self._log_handler = handler

    def _queue_log(self, message: str):
        try:
            self.log_queue.put_nowait(message)
        except queue.Full:
            pass

    def _drain_logs(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self._append_log(msg)
        except queue.Empty:
            pass
        finally:
            self.after(200, self._drain_logs)

    def _append_log(self, message: str):
        stamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{stamp}] {message}\n")
        self.log_text.see(tk.END)

    def _on_capture_toggle(self):
        if self.cap_mic and self.disable_mic.get():
            self._append_log("Stopping mic capture per toggle")
            self.cap_mic.stop()
            self.cap_mic.join(timeout=1.0)
            self.cap_mic = None
        if self.cap_sys and self.disable_sys.get():
            self._append_log("Stopping system capture per toggle")
            self.cap_sys.stop()
            self.cap_sys.join(timeout=1.0)
            self.cap_sys = None
        if self.sys_monitor and self.disable_sys.get():
            try:
                self.sys_monitor.stop()
            except Exception:
                pass
            self.sys_monitor = None

        if self._frame_dispatch_thread:
            if not self.disable_mic.get() and self.cap_mic is None:
                idx = self._parse_idx(self.mic_combo.get())
                if idx is not None:
                    self.cap_mic = AudioCapture("MIC", idx, self.cfg_mic, self.q)
                    self.cap_mic.start()
                    self._append_log("Mic capture re-armed")
            if not self.disable_sys.get() and self.cap_sys is None and self._loopback_device_idx is not None:
                sys_idx = self._loopback_device_idx
                if ensure_blackhole_output():
                    self._append_log("Verified system output via BlackHole 2ch")
                else:
                    self._append_log(
                        "Unable to force system output to BlackHole 2ch automatically; please confirm manually."
                    )
                self._restart_monitor()
                self.cap_sys = AudioCapture("SYS", sys_idx, self.cfg_sys, self.q, monitor=self.sys_monitor)
                self.cap_sys.start()
                self._append_log("System capture re-armed")

    def _write_text(self, widget, text: str, newline: bool = False):
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        if text:
            suffix = "\n" if newline and not text.endswith("\n") else ""
            widget.insert(tk.END, text + suffix)
        widget.see(tk.END)
        widget.configure(state=tk.DISABLED)

    def _monitor_selection(self) -> Tuple[Optional[int], Optional[str], Optional[int]]:
        selection = self.monitor_var.get()
        info = self._monitor_devices.get(selection)
        if info:
            return info
        return find_monitor_device()

    def _restart_monitor(self) -> None:
        if self.sys_monitor is not None:
            try:
                self.sys_monitor.stop()
            except Exception:
                pass
            self.sys_monitor = None

        mon_idx, mon_name, mon_channels = self._monitor_selection()
        if mon_idx is None:
            self._append_log("No speaker found for monitoring; system audio will stay on BlackHole")
            return

        try:
            channels = mon_channels if mon_channels is not None else 2
            self.sys_monitor = AudioMonitor(mon_idx, channels=channels)
            self.sys_monitor.start()
            label = mon_name or "monitor"
            self._append_log(f"Routing BlackHole audio to {label}")
            if self.cap_sys is not None:
                self.cap_sys.monitor = self.sys_monitor
        except Exception as exc:
            self.sys_monitor = None
            self._append_log(f"Failed to start monitor output {mon_name}: {exc}")

    def _on_monitor_selected(self, _event=None):
        selection = self.monitor_var.get()
        if selection:
            self._append_log(f"Monitor output set to {selection}")
        if self.cap_sys and not self.disable_sys.get():
            self._restart_monitor()

    def _run_agent_async(self, transcript: str, screenshots: Optional[List[str]] = None) -> None:
        transcript = (transcript or "").strip()
        if not transcript:
            return
        if self._agent_running:
            self._append_log("Agent busy; skipping overlapping request")
            return

        self._agent_running = True
        if getattr(self, "help_btn", None):
            self.help_btn.configure(state=tk.DISABLED)
        self._append_conversation_line("Agent status", "Agent is working…")
        if self.agent_send_button is not None:
            self.agent_send_button.configure(state=tk.DISABLED)
        if self.agent1_stage_box is not None:
            self._write_text(self.agent1_stage_box, "…waiting for Agent1…", newline=False)
        if self.agent2_stage_box is not None:
            self._write_text(self.agent2_stage_box, "…waiting for Agent2…", newline=False)

        def _stage_cb(stage: str, text: str) -> None:
            def _ui():
                cleaned = str(text or "").strip()
                if cleaned:
                    # cache latest stage output
                    self._latest_stage_outputs[stage] = cleaned
                    # show it immediately in the dedicated box
                    if stage == "Agent1":
                        widget = self.agent1_stage_box
                    elif stage == "Agent2":
                        widget = self.agent2_stage_box
                    else:
                        widget = None
                    if widget is not None:
                        widget.configure(state=tk.NORMAL)
                        widget.delete("1.0", tk.END)
                        widget.insert(tk.END, cleaned)
                        widget.see(tk.END)
                        widget.configure(state=tk.DISABLED)
                self._append_log(f"{stage} response ready")

            self.after(0, _ui)

        def _worker():
            try:
                ctx = getattr(self, "session_context", {}) or {}
                result = asyncio.run(
                    run_workflow(
                        WorkflowInput(
                            input_as_text=transcript,
                            screenshots=(screenshots or []),
                            cv_text=ctx.get("cv_text"),
                            prior_interview_notes=ctx.get("prior_interview_notes"),
                            constraints_or_prefs=ctx.get("constraints_or_prefs"),
                        ),
                        session=self.agent_session,
                        on_stage=_stage_cb,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self.after(0, self._handle_agent_failure, exc)
                return
            self.after(0, lambda: self._handle_agent_success(result))

        thread = threading.Thread(target=_worker, name="agent-runner", daemon=True)
        self._agent_thread = thread
        thread.start()

    def _handle_agent_success(self, result: Dict[str, object]) -> None:
        self._agent_running = False
        self._agent_thread = None
        if self.agent_send_button is not None:
            self.agent_send_button.configure(state=tk.NORMAL)
        if getattr(self, "help_btn", None):
            self.help_btn.configure(state=tk.NORMAL)
        # Pull staged outputs; fall back to result fields if necessary
        a1 = self._latest_stage_outputs.get("Agent1", "")
        a2 = self._latest_stage_outputs.get("Agent2", "")
        if isinstance(result, dict):
            if not a1:
                raw1 = result.get("agent1_output")
                a1 = str(raw1).strip() if isinstance(raw1, str) else ""
            if not a2:
                raw2 = result.get("agent2_output") or result.get("output_text")
                a2 = str(raw2).strip() if isinstance(raw2, str) else ""

        if a1:
            self._append_conversation_line("Agent1", a1)
        if a2:
            self._append_conversation_line("Agent2", a2)
        if not a1 and not a2:
            self._append_conversation_line("Agent", "Agent returned no content.")
            self._append_log("Agent response empty")
            return
        # Ensure the stage boxes show final outputs
        if a1 and self.agent1_stage_box is not None:
            self._write_text(self.agent1_stage_box, a1, newline=False)
        if a2 and self.agent2_stage_box is not None:
            self._write_text(self.agent2_stage_box, a2, newline=False)
        self._latest_stage_outputs["Agent2"] = a2
        self._append_log("Agent responses appended to Conversation")

    def _handle_agent_failure(self, exc: Exception) -> None:
        self._agent_running = False
        self._agent_thread = None
        if self.agent_send_button is not None:
            self.agent_send_button.configure(state=tk.NORMAL)
        if getattr(self, "help_btn", None):
            self.help_btn.configure(state=tk.NORMAL)
        message = f"Agent error: {exc}"
        self._append_conversation_line("Agent error", message)
        self._append_log(message)
        if self.agent_entry is not None:
            self.agent_entry.focus_set()

    def _collect_transcript_lines(
        self,
        include_partials: bool = True,
        window_seconds: int = 120,
        max_lines: int = 120,
    ) -> List[str]:
        window_seconds = max(5, int(window_seconds))
        cutoff = time.time() - window_seconds
        segments = sorted(
            (seg for seg in self._merged_segments if seg.started_at >= cutoff),
            key=lambda seg: seg.started_at,
        )
        lines: List[str] = []
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            timestamp = time.strftime("%H:%M:%S", time.localtime(seg.started_at))
            lines.append(f"[{timestamp}] {seg.speaker}: {text}")
        if include_partials:
            for speaker, raw in self._speaker_partial.items():
                raw = (raw or "").strip()
                if not raw:
                    continue
                cleaned = re.sub(r"^\[\d{2}:\d{2}:\d{2}\]\s+", "", raw).rstrip(" …")
                if cleaned:
                    timestamp = time.strftime("%H:%M:%S")
                    lines.append(f"[{timestamp}] {speaker}: {cleaned}")
        if len(lines) > max_lines:
            keep = max(1, int(max_lines))
            lines = lines[-keep:]
        return lines

    def _dispatch_agent_request(self) -> None:
        try:
            waited = False
            flush_raw = os.environ.get("LOCAL_LLM_STT_FLUSH_TIMEOUT_MS", "500")
            try:
                flush_ms = int(flush_raw)
            except ValueError:
                flush_ms = 500
            flush_sec = max(0, min(2000, flush_ms)) / 1000.0
            if self.tx_mic:
                if getattr(self.tx_mic, "flush_and_wait", None):
                    if self.tx_mic.flush_and_wait(timeout=flush_sec):
                        waited = True
                else:
                    self.tx_mic.flush_now()
            if self.tx_sys:
                if getattr(self.tx_sys, "flush_and_wait", None):
                    if self.tx_sys.flush_and_wait(timeout=flush_sec):
                        waited = True
                else:
                    self.tx_sys.flush_now()
        except Exception:
            pass
        if waited:
            try:
                self.update_idletasks()
                self.update()
            except Exception:
                pass
        wait_ms = int(os.environ.get("LOCAL_LLM_SEND_COALESCE_MS", "0"))
        if wait_ms > 0:
            time.sleep(min(500, max(0, wait_ms)) / 1000.0)

        win_s_raw = os.environ.get("LOCAL_LLM_TRANSCRIPT_WINDOW_S", "120")
        max_lines_raw = os.environ.get("LOCAL_LLM_TRANSCRIPT_MAX_LINES", "120")
        try:
            win_s = int(win_s_raw)
        except ValueError:
            win_s = 120
        try:
            max_lines = int(max_lines_raw)
        except ValueError:
            max_lines = 120

        transcript_lines = self._collect_transcript_lines(
            include_partials=True,
            window_seconds=win_s,
            max_lines=max_lines,
        )
        manual_notes = list(self.pending_manual_notes)
        screenshot_paths = list(self.pending_screenshots)

        sections: List[str] = []
        if transcript_lines:
            sections.append("Transcripts:\n" + "\n".join(transcript_lines))
        if manual_notes:
            sections.append("Notes:\n" + "\n\n".join(manual_notes))
        if screenshot_paths:
            formatted = "\n".join(f"{idx + 1}. {path}" for idx, path in enumerate(screenshot_paths, start=1))
            sections.append("Screenshots:\n" + formatted)

        payload = "\n\n".join(sections).strip()
        if not payload:
            self._append_log("Nothing available to send to agent.")
            return

        self.pending_manual_notes.clear()
        self.pending_screenshots.clear()
        self._append_log(
            f"Sending to agent (segments={len(transcript_lines)}, notes={len(manual_notes)}, screenshots={len(screenshot_paths)})"
        )
        self._run_agent_async(payload, screenshots=screenshot_paths)

    def _send_agent_message(self) -> None:
        if self.agent_entry is None:
            return
        text = self.agent_entry.get("1.0", tk.END)
        cleaned = (text or "").strip()
        if cleaned:
            self.agent_entry.delete("1.0", tk.END)
            self.agent_entry.focus_set()
            stamp = time.strftime("%H:%M:%S", time.localtime())
            entry_line = f"[{stamp}] You: {cleaned}"
            self.conversation_log.append(entry_line)
            if self.conversation_text is not None:
                widget = self.conversation_text
                widget.configure(state=tk.NORMAL)
                widget.insert(tk.END, entry_line + "\n\n")
                widget.configure(state=tk.DISABLED)
                widget.see(tk.END)
            self._conversation_entries.append({"timestamp": time.time(), "text": entry_line, "segments": []})
            self.pending_manual_notes.append(cleaned)
            self._append_log("Manual agent note queued")
        else:
            self._append_log("Send pressed; no manual note added, using stored transcripts and screenshots.")

        if self._agent_running:
            self._append_log("Agent busy; wait for the current reply before sending another message")
            return
        self._dispatch_agent_request()

    def _capture_screenshot(self) -> None:
        try:
            import mss
            from mss import tools
        except ImportError:
            self._append_log("Screenshot capture unavailable (mss not installed).")
            return

        screenshots_dir = Path(__file__).resolve().parent / "screenshots"
        try:
            screenshots_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._append_log(f"Failed to prepare screenshots directory: {exc}")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        target_path = screenshots_dir / filename

        try:
            self.update_idletasks()
            try:
                self.withdraw()
                self.update()
                time.sleep(0.15)

                with mss.mss() as sct:
                    monitor = sct.monitors[0]
                    shot = sct.grab(monitor)
                    tools.to_png(shot.rgb, shot.size, output=str(target_path))
            finally:
                self.deiconify()
                self.lift()
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"Screenshot capture failed: {exc}")
            return

        display_path: Path = target_path
        try:
            root_dir = Path(__file__).resolve().parent
            display_path = target_path.relative_to(root_dir)
        except ValueError:
            pass

        stamp = time.strftime("%H:%M:%S", time.localtime())
        note = f"[{stamp}] Screenshot captured: {display_path}"
        self.conversation_log.append(note)
        if self.conversation_text is not None:
            widget = self.conversation_text
            widget.configure(state=tk.NORMAL)
            widget.insert(tk.END, note + "\n\n")
            widget.configure(state=tk.DISABLED)
            widget.see(tk.END)
        self._conversation_entries.append({"timestamp": time.time(), "text": note, "segments": []})
        self.pending_screenshots.append(str(target_path))
        self._append_log(f"Screenshot stored for next send: {display_path}")

    def _on_sys_selected(self, _event):
        selection = self.sys_combo.get()
        idx = self._parse_idx(selection)
        self._loopback_device_idx = idx
        if selection and ": " in selection:
            self._loopback_name = selection.split(": ", 1)[1]
        else:
            self._loopback_name = selection
        self._on_capture_toggle()

    def _cancel_calibration_timer(self):
        if self._calibration_after_id is not None:
            try:
                self.after_cancel(self._calibration_after_id)
            except Exception:
                pass
            self._calibration_after_id = None

    def toggle_listen(self):
        self._apply_segment_settings()
        if any((self.cap_mic, self.cap_sys, self.tx_mic, self.tx_sys, self._frame_dispatch_thread)):
            self.stop_all()
            return

        self._cancel_calibration_timer()
        for label in ("MIC", "SYS"):
            self._clear_transcript(label, log=False)

        self.cfg_mic.dump_segments = False
        self.cfg_sys.dump_segments = False

        mic_idx = self._parse_idx(self.mic_combo.get())
        sys_idx = self._loopback_device_idx
        mic_enabled = not self.disable_mic.get()
        sys_enabled = not self.disable_sys.get()

        if not mic_enabled:
            mic_idx = None
            self._append_log("Mic capture disabled via toggle")

        if not sys_enabled:
            sys_idx = None
            self._append_log("System capture disabled via toggle (default)")

        self.sys_monitor = None

        if sys_idx is not None:
            if ensure_blackhole_output():
                self._append_log("Verified system output via BlackHole 2ch")
            else:
                self._append_log(
                    "Unable to force system output to BlackHole 2ch automatically; please confirm manually."
                )

            mon_idx, mon_name, mon_channels = find_monitor_device()
            if mon_idx is not None:
                try:
                    self.sys_monitor = AudioMonitor(mon_idx, channels=mon_channels)
                    self.sys_monitor.start()
                    monitor_label = mon_name or "monitor"
                    self._append_log(f"Routing BlackHole audio to {monitor_label}")
                except Exception as exc:
                    self.sys_monitor = None
                    self._append_log(f"Failed to start monitor output {mon_name}: {exc}")
            else:
                self._append_log("No speaker found for monitoring; system audio will stay on BlackHole")
        elif sys_enabled:
            self._append_log("System capture unavailable (no loopback device)")

        if mic_idx is None and mic_enabled:
            self._append_log("Mic capture unavailable (no device selected)")

        if mic_idx is None and sys_idx is None:
            self.status.set("Idle")
            self._append_log("No capture devices enabled; listen aborted")
            return

        self.q = queue.Queue(maxsize=self._queue_maxsize)
        self.tx_mic = None
        self.tx_sys = None
        if mic_idx is not None:
            self.tx_mic = HTTPTranscriber("MIC", self.on_text, config=self._http_cfg_mic)
            self.tx_mic.start()
        if sys_idx is not None:
            self.tx_sys = HTTPTranscriber("SYS", self.on_text, config=self._http_cfg_sys)
            self.tx_sys.start()

        self._frame_dispatch_stop = threading.Event()
        self._frame_dispatch_thread = threading.Thread(
            target=self._dispatch_frames,
            name="frame-dispatch",
            daemon=True,
        )
        self._frame_dispatch_thread.start()

        if mic_idx is not None:
            self.cap_mic = AudioCapture("MIC", mic_idx, self.cfg_mic, self.q)
            self.cap_mic.start()
        if sys_idx is not None:
            self.cap_sys = AudioCapture("SYS", sys_idx, self.cfg_sys, self.q, monitor=self.sys_monitor)
            self.cap_sys.start()

        if self.cap_mic or self.cap_sys:
            self.status.set("Listening…")
            self._append_log("Capture threads started (server VAD, English)")
        else:
            self.status.set("Idle")
            self._append_log("No capture devices started; check device selection")

    def stop_all(self):
        self._cancel_calibration_timer()
        capture_threads = [self.cap_mic, self.cap_sys]
        for thread in capture_threads:
            try:
                if thread:
                    thread.stop()
            except Exception:
                pass
        for thread in capture_threads:
            try:
                if thread:
                    thread.join(timeout=1)
            except Exception:
                pass

        if self._frame_dispatch_stop:
            self._frame_dispatch_stop.set()
        if self._frame_dispatch_thread:
            try:
                self._frame_dispatch_thread.join(timeout=1)
            except Exception:
                pass
        self._frame_dispatch_thread = None
        self._frame_dispatch_stop = None

        for tx in (self.tx_mic, self.tx_sys):
            if not tx:
                continue
            try:
                tx.stop()
            except Exception:
                pass
        for tx in (self.tx_mic, self.tx_sys):
            if not tx:
                continue
            try:
                tx.join(timeout=1)
            except Exception:
                pass
        self.tx_mic = self.tx_sys = None
        if self.sys_monitor:
            try:
                self.sys_monitor.stop()
            except Exception:
                pass
        self.sys_monitor = None

        if self.q is not None:
            try:
                while True:
                    self.q.get_nowait()
                    try:
                        self.q.task_done()
                    except ValueError:
                        pass
            except queue.Empty:
                pass
        self.q = None

        self.cap_mic = None
        self.cap_sys = None
        self.status.set("Stopped")
        self._append_log("Capture threads stopped")
        self._refresh_all_live_boxes()

    def _dispatch_frames(self):
        q = self.q
        stop_event = self._frame_dispatch_stop
        if q is None or stop_event is None:
            return

        while not stop_event.is_set():
            try:
                label, payload, meta = q.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                data = payload
                if isinstance(data, memoryview):
                    data = data.tobytes()
                elif hasattr(data, "tobytes"):
                    data = data.tobytes()
                elif not isinstance(data, (bytes, bytearray)):
                    data = bytes(data)
            except Exception:
                data = b""

            try:
                if meta.get("reason") == "frame" and data:
                    if label == "MIC" and self.tx_mic:
                        self.tx_mic.send_frame(data)
                    elif label == "SYS" and self.tx_sys:
                        self.tx_sys.send_frame(data)
            finally:
                try:
                    q.task_done()
                except ValueError:
                    pass

    def _parse_idx(self, value: str):
        if not value or value.startswith("No loopback"):
            return None
        try:
            return int(value.split(":", 1)[0].strip("#"))
        except Exception:
            return None

    def _ensure_stream_state(self, label: str) -> None:
        if label in self._stream_state:
            return
        speaker = self._speaker_alias.get(label)
        if speaker is None:
            speaker = label
            self._speaker_alias[label] = speaker
        if speaker not in self._speaker_live_text:
            self._speaker_live_text[speaker] = ""
        if speaker not in self._speaker_history:
            self._speaker_history[speaker] = []
        if speaker not in self._speaker_partial:
            self._speaker_partial[speaker] = ""
        if speaker not in self._speaker_segments:
            self._speaker_segments[speaker] = deque(maxlen=self._speaker_segment_maxlen)
        self._stream_state[label] = StreamState(speaker=speaker)

    def _track_stream_update(self, label: str, text: str, meta: dict, reason: str) -> Optional[TimedSegment]:
        self._ensure_stream_state(label)
        state = self._stream_state[label]
        speaker = state.speaker
        now = time.time()
        candidate = str(text or "")
        candidate_stripped = candidate.strip()

        if reason == "partial":
            if not candidate_stripped:
                self._speaker_partial[speaker] = ""
                return None
            if state.current_start is None:
                state.current_start = now
            state.last_update = now
            stamp = time.strftime("%H:%M:%S", time.localtime())
            self._speaker_partial[speaker] = f"[{stamp}] {candidate_stripped} …"
            return None

        if reason == "too_short_dropped":
            state.current_start = None
            state.last_update = now
            self._speaker_partial[speaker] = ""
            return None

        if not candidate_stripped:
            state.current_start = None
            state.last_update = now
            self._speaker_partial[speaker] = ""
            return None

        duration = meta.get("duration_s")
        started_at: float
        if duration is not None:
            try:
                duration_f = float(duration)
            except (TypeError, ValueError):
                duration_f = None
        else:
            duration_f = None
        if duration_f and duration_f > 0:
            started_at = max(0.0, now - duration_f)
        elif state.current_start is not None:
            started_at = state.current_start
        else:
            started_at = now

        segment = TimedSegment(
            label=label,
            speaker=speaker,
            text=candidate_stripped,
            started_at=started_at,
            ended_at=now,
            reason=reason,
        )

        state.current_start = None
        state.last_update = now
        stamp = time.strftime("%H:%M:%S", time.localtime())
        self._speaker_history.setdefault(speaker, []).append(f"[{stamp}] {candidate_stripped}")
        self._speaker_partial[speaker] = ""
        if speaker not in self._speaker_segments:
            self._speaker_segments[speaker] = deque(maxlen=self._speaker_segment_maxlen)
        self._speaker_segments[speaker].append(segment)
        self._merged_segments.append(segment)
        return segment

    def on_text(self, label, text, meta):
        def _update():
            if label == "MIC" and self.disable_mic.get():
                return
            if label == "SYS" and self.disable_sys.get():
                return
            if meta.get("error"):
                reason = meta.get("reason", "error")
                self._append_log(f"{label} error ({reason}): {meta['error']}")
                return

            reason = meta.get("reason", "segment")
            segment = self._track_stream_update(label, text, meta, reason)
            speaker = self._speaker_alias.get(label, label)
            self._refresh_live_box(speaker)

            dur = meta.get("duration_s")
            dbg = f"{label} {reason}"
            if dur is not None:
                dbg += f" {dur:.2f}s"

            if reason == "partial":
                return

            if reason == "too_short_dropped":
                self._append_log(dbg)
                return

            self._append_log(dbg)

        self.after(0, _update)

    def on_help(self):
        payload_text: Optional[str] = None
        with self._help_slice_lock:
            cut_segments = [seg for seg in self._merged_segments if seg.ended_at > self._help_anchor]
            if not cut_segments:
                self._append_log("Help! invoked but no new segments were available")
                return

            cut_segments.sort(key=lambda seg: seg.started_at)

            speaker_groups: Dict[str, List[TimedSegment]] = {}
            for seg in cut_segments:
                speaker_groups.setdefault(seg.speaker, []).append(seg)

            formatted_lines: List[str] = []
            for seg in cut_segments:
                ts = seg.started_at
                timestamp = time.strftime("%H:%M:%S", time.localtime(ts))
                label = seg.speaker
                text = seg.text.strip()
                if not text:
                    continue
                formatted_lines.append(f"[{timestamp}] {label}: {text}")

            payload = "\n".join(formatted_lines).strip()
            if not payload:
                self._append_log("Help! had segments but they were empty after cleaning")
                return

            self._append_conversation_entry(cut_segments, payload)

            for speaker in speaker_groups:
                self._speaker_live_text[speaker] = ""
                if speaker in self._speaker_segments:
                    self._speaker_segments[speaker].clear()
                self._refresh_live_box(speaker)

            self._help_anchor = cut_segments[-1].ended_at
            payload_text = payload

        if payload_text is None:
            return

        self._append_log("Help! dispatched transcript slice to agent")
        self._run_agent_async(payload_text, screenshots=None)

    def destroy(self):
        logger = logging.getLogger("dual_stt")
        if self._log_handler:
            logger.removeHandler(self._log_handler)
            self._log_handler = None
        self._cancel_session_logging()
        thread = self._agent_thread
        if thread and thread.is_alive():
            try:
                thread.join(timeout=0.5)
            except Exception:
                pass
        self._agent_thread = None
        self._agent_running = False
        super().destroy()


def run_dual_stt_app() -> None:
    ensure_repo_venv()
    App().mainloop()


# ---------------------------------------------------------------------------
# CLI pipeline (from stt_to_llm.py)


def _pick_default_mic() -> Optional[int]:
    devs = list_input_devices()
    if not devs:
        return None
    preferred = None
    fallback = devs[0]
    for device in devs:
        name_low = device.get("name", "").lower()
        if any(bad in name_low for bad in ("blackhole", "loopback", "soundflower")):
            continue
        if "macbook" in name_low and "microphone" in name_low:
            return device["index"]
        if preferred is None and "microphone" in name_low:
            preferred = device
    return (preferred or fallback)["index"]


CONTEXT_MAX_CHARS = 8_000
CONTEXT_MAX_FILE_BYTES = 8_000


def _load_context_file(path: Path) -> Optional[str]:
    if not path.exists():
        print(f"Context file not found: {path}", file=sys.stderr)
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Failed to read context file {path}: {exc}", file=sys.stderr)
        return None


def _load_context_dir(directory: Path) -> Optional[str]:
    if not directory.exists():
        print(f"Context directory not found: {directory}", file=sys.stderr)
        return None
    if not directory.is_dir():
        print(f"Context path is not a directory: {directory}", file=sys.stderr)
        return None

    patterns = ["*.txt", "*.md"]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(directory.rglob(pattern))
    files = sorted({f.resolve() for f in files})
    if not files:
        print(f"No text context files under {directory}", file=sys.stderr)
        return None

    buffer: List[str] = []
    total_chars = 0
    truncated = False

    for file_path in files:
        try:
            size = file_path.stat().st_size
        except OSError:
            continue
        if size > CONTEXT_MAX_FILE_BYTES:
            max_kb = max(1, CONTEXT_MAX_FILE_BYTES // 1024)
            print(f"Skipping large context file (>{max_kb} KB): {file_path}", file=sys.stderr)
            continue
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Failed to read context file {file_path}: {exc}", file=sys.stderr)
            continue
        if not content.strip():
            continue
        remaining = CONTEXT_MAX_CHARS - total_chars
        if remaining <= 0:
            truncated = True
            break
        if len(content) > remaining:
            buffer.append(content[:remaining])
            total_chars += remaining
            truncated = True
            break
        buffer.append(content)
        total_chars += len(content)

    if truncated:
        print(
            f"Context truncated to {CONTEXT_MAX_CHARS} characters; remaining files ignored.",
            file=sys.stderr,
        )

    if not buffer:
        return None
    return "\n\n".join(buffer)


def _merge_context(parts: Sequence[str]) -> Optional[str]:
    valid = [p.strip() for p in parts if p and p.strip()]
    if not valid:
        return None
    return "\n\n".join(valid)


def _join_for_llm(segments: Sequence[str]) -> str:
    return " ".join(s.strip() for s in segments if s.strip())


def stt_to_llm_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Headless mic STT → sink")
    parser.add_argument("--device-index", type=int, default=None, help="Mic device index (default: first input)")
    parser.add_argument(
        "--batch",
        type=int,
        default=int(os.environ.get("STT_BATCH", "1")),
        help="Segments to accumulate before sending (default: 1)",
    )
    parser.add_argument(
        "--context-file",
        default=os.environ.get("STT_CONTEXT_FILE", None),
        help="Optional path to extra context",
    )
    parser.add_argument(
        "--context-dir",
        default=os.environ.get("STT_CONTEXT_DIR", None),
        help="Directory of context snippets (.txt/.md)",
    )
    parser.add_argument("--send-partials", action="store_true", help="Also send partials (not just endpoints)")
    parser.add_argument(
        "--sink",
        default=os.environ.get("STT_SINK", "llm"),
        choices=["llm", "console"],
        help="Delivery sink: llm (OpenAI) or console logging",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce console logs")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = CaptureConfig()

    mic_idx = args.device_index if args.device_index is not None else _pick_default_mic()
    if mic_idx is None:
        print("No microphone input device found.", file=sys.stderr)
        return 2

    context_parts: List[str] = []
    if args.context_file:
        text = _load_context_file(Path(args.context_file))
        if text:
            context_parts.append(text)
    if args.context_dir:
        text = _load_context_dir(Path(args.context_dir))
        if text:
            context_parts.append(text)

    context_text = _merge_context(context_parts)

    frame_queue: queue.Queue = queue.Queue(maxsize=64)
    outputs_lock = threading.Lock()
    pending_segments: List[str] = []
    batch_size = max(1, int(args.batch))

    router: Optional[LLMRouter] = None
    send_partials = bool(args.send_partials)
    if args.sink == "llm":
        router = LLMRouter()

    if args.sink == "llm":
        assert router is not None

        def _send_to_llm(segments: Sequence[str], reason: str) -> None:
            text = _join_for_llm(segments)
            if not text:
                return
            prompt = build_prompt(context_text, text)
            streamed = False
            last_stream_char: Optional[str] = None
            header_printed = False

            def _ensure_header() -> None:
                nonlocal header_printed
                if header_printed or args.quiet:
                    return
                stamp = time.strftime("%H:%M:%S")
                print(f"[{stamp}] === MODEL REPLY ({reason}) ===")
                header_printed = True

            def _on_stream(chunk: str) -> None:
                nonlocal streamed, last_stream_char
                if not chunk:
                    return
                _ensure_header()
                streamed = True
                last_stream_char = chunk[-1]
                print(chunk, end="", flush=True)

            try:
                reply = router.query(prompt, on_stream=None if args.quiet else _on_stream)
            except Exception as exc:  # noqa: BLE001
                if streamed and last_stream_char != "\n":
                    print()
                if not header_printed:
                    _ensure_header()
                print(f"LLM error ({reason}): {exc}", file=sys.stderr)
                return
            if not header_printed:
                _ensure_header()
            reply_text = reply.strip()
            if streamed:
                if last_stream_char != "\n":
                    print()
                if not args.quiet:
                    print()
            else:
                print(reply_text)
                if not args.quiet:
                    print()

        sink_fn = _send_to_llm
    else:

        def _send_to_console(segments: Sequence[str], reason: str) -> None:
            text = "\n".join(s.strip() for s in segments if s.strip())
            if not text:
                return
            print(f"[{time.strftime('%H:%M:%S')}] ({reason}) {text}")

        sink_fn = _send_to_console

    def _emit_to_sink(segments: Sequence[str], reason: str) -> None:
        if not segments:
            return
        threading.Thread(target=sink_fn, args=(list(segments), reason), daemon=True).start()

    def on_text(label: str, text: str, meta: dict):
        if label != "MIC":
            return
        if not text:
            return

        reason = meta.get("reason", "segment")
        if reason == "partial":
            if send_partials:
                _emit_to_sink([text], reason)
            return

        to_send: Optional[List[str]] = None
        with outputs_lock:
            pending_segments.append(text)
            if len(pending_segments) >= batch_size:
                to_send = pending_segments[:]
                pending_segments.clear()

        if to_send:
            _emit_to_sink(to_send, reason)

    transcriber = HTTPTranscriber("MIC", on_text)
    transcriber.start()

    cap = AudioCapture("MIC", mic_idx, cfg, frame_queue)
    cap.start()

    stop_event = threading.Event()

    def forward_frames():
        while not stop_event.is_set():
            try:
                label, payload, meta = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if meta.get("reason") != "frame" or label != "MIC":
                    continue
                data = payload
                if isinstance(data, memoryview):
                    data = data.tobytes()
                elif hasattr(data, "tobytes"):
                    data = data.tobytes()
                elif not isinstance(data, (bytes, bytearray)):
                    data = bytes(data)
                if data:
                    transcriber.send_frame(bytes(data))
            finally:
                try:
                    frame_queue.task_done()
                except ValueError:
                    pass

    forward_thread = threading.Thread(target=forward_frames, name="frame-forward", daemon=True)
    forward_thread.start()

    if not args.quiet:
        print("Listening… (Ctrl+C to stop)")

    try:
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        try:
            cap.stop()
        except Exception:
            pass
        try:
            transcriber.stop()
        except Exception:
            pass
        try:
            cap.join(timeout=1)
        except Exception:
            pass
        try:
            transcriber.join(timeout=1)
        except Exception:
            pass
        try:
            forward_thread.join(timeout=1)
        except Exception:
            pass
        try:
            while True:
                frame_queue.get_nowait()
                try:
                    frame_queue.task_done()
                except ValueError:
                    pass
        except queue.Empty:
            pass

    with outputs_lock:
        leftover = pending_segments[:]
        pending_segments.clear()
    if leftover:
        sink_fn(leftover, "flush")

    return 0


# ---------------------------------------------------------------------------
# Entry point helpers

__all__ = [
    "App",
    "AudioCapture",
    "AudioMonitor",
    "CaptureConfig",
    "HTTPSTTConfig",
    "HTTPTranscriber",
    "LLMConfig",
    "LLMRouter",
    "SegmentConfig",
    "TimedSegment",
    "StreamState",
    "build_prompt",
    "build_captures",
    "ensure_blackhole_output",
    "ensure_blackhole_input",
    "find_monitor_device",
    "find_loopback_candidate",
    "list_input_devices",
    "list_output_devices",
    "load_openai_api_key",
    "FRAME_MS",
    "FRAME_SAMPLES",
    "SAMPLE_RATE",
    "run_dual_stt_app",
    "stt_to_llm_main",
]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(argv if argv is not None else sys.argv)
    if len(args) > 1 and args[1] in {"stt", "cli"}:
        return stt_to_llm_main(args[2:])
    run_dual_stt_app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
