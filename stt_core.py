from __future__ import annotations

import io
import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

import httpx
import numpy as np
import sounddevice as sd

from config import HTTPSTTConfig, SegmentConfig


# ---------------------------------------------------------------------------
# Configuration helpers


def _default_key_file() -> Path:
    # Prefer openai_api_key.txt alongside main.py, fall back to the parent dir.
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "openai_api_key.txt",
        script_dir.parent / "openai_api_key.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_openai_api_key() -> Optional[str]:
    """Load the OpenAI API key from env or a local text file."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        key = key.strip()
        if key:
            return key

    key_file = os.environ.get("OPENAI_API_KEY_FILE")
    path = Path(key_file).expanduser() if key_file else _default_key_file()
    if not path.exists():
        return None

    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


# ---------------------------------------------------------------------------
# Shared audio constants

DUAL_LOG = logging.getLogger("dual_stt")
if not DUAL_LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
DUAL_LOG.propagate = False

HTTP_LOG = logging.getLogger("http_stt")

SAMPLE_RATE = 16000
FRAME_MS = 10
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 160 samples @16 kHz, 10 ms frame

WAV_DUMP_ROOT = Path("/tmp")
WAV_DUMP_PREFIX = "dual_stt"


# ---------------------------------------------------------------------------
# Device helpers


def list_input_devices() -> List[Dict[str, object]]:
    devices: List[Dict[str, object]] = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            devices.append({"index": i, "name": dev["name"], "channels": dev["max_input_channels"]})
    return devices


def list_output_devices() -> List[Dict[str, object]]:
    devices: List[Dict[str, object]] = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] > 0:
            devices.append({"index": i, "name": dev["name"], "channels": dev["max_output_channels"]})
    return devices


def _match_device(want: str, name: str) -> bool:
    return want.lower() in name.lower()


def find_loopback_candidate() -> Tuple[Optional[int], Optional[str]]:
    priorities = ("blackhole 2ch", "blackhole", "loopback", "soundflower")
    devices = list_input_devices()
    for want in priorities:
        for dev in devices:
            if _match_device(want, dev["name"]):
                return dev["index"], dev["name"]
    return None, None


def find_monitor_device(preferred: Optional[List[str]] = None) -> Tuple[Optional[int], Optional[str], int]:
    preferences = preferred or []
    env_preferred = os.environ.get("LOCAL_LLM_MONITOR_DEVICE")
    if env_preferred:
        preferences = [env_preferred] + preferences

    fallback_excludes = ("blackhole", "loopback", "soundflower")
    devices = list_output_devices()

    def _pick(names: List[str]) -> Tuple[Optional[int], Optional[str], int]:
        for want in names:
            for dev in devices:
                if _match_device(want, dev["name"]):
                    channels = max(1, min(2, int(dev.get("channels", 1))))
                    return dev["index"], dev["name"], channels
        return None, None, 0

    if preferences:
        idx, name, ch = _pick(preferences)
        if idx is not None:
            return idx, name, ch

    common = ["macbook", "speakers", "display audio", "headphones", "built-in output"]
    idx, name, ch = _pick(common)
    if idx is not None:
        return idx, name, ch

    for dev in devices:
        lowered = dev["name"].lower()
        if not any(token in lowered for token in fallback_excludes):
            channels = max(1, min(2, int(dev.get("channels", 1))))
            return dev["index"], dev["name"], channels

    return None, None, 0


def _int16_to_float32(samples: np.ndarray) -> np.ndarray:
    return (samples.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


@dataclass
class CaptureConfig:
    dump_segments: bool = False


class AudioMonitor:
    def __init__(self, device_idx: int, channels: int = 2, sample_rate: int = SAMPLE_RATE, blocksize: int = FRAME_SAMPLES):
        self.device_idx = device_idx
        self.channels = max(1, channels)
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self._lock = threading.Lock()
        self._queue: Deque[np.ndarray] = deque()
        self._current: Optional[np.ndarray] = None
        self._current_pos = 0
        self._stream: Optional[sd.OutputStream] = None

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.blocksize,
            device=self.device_idx,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        with self._lock:
            self._queue.clear()
            self._current = None
            self._current_pos = 0

    def feed(self, pcm16: np.ndarray) -> None:
        if pcm16.size == 0:
            return
        mono = _int16_to_float32(pcm16).reshape(-1, 1)
        audio = np.repeat(mono, self.channels, axis=1) if self.channels >= 2 else mono
        with self._lock:
            self._queue.append(audio)

    def _callback(self, outdata, frames, _time_info, status) -> None:
        if status:
            DUAL_LOG.debug("monitor status: %s", status)
        outdata.fill(0.0)
        written = 0
        with self._lock:
            while written < frames:
                if self._current is None:
                    if not self._queue:
                        break
                    self._current = self._queue.popleft()
                    self._current_pos = 0

                remaining = self._current.shape[0] - self._current_pos
                if remaining <= 0:
                    self._current = None
                    continue

                take = min(remaining, frames - written)
                slice_ = self._current[self._current_pos : self._current_pos + take]
                outdata[written : written + take] = slice_
                written += take
                self._current_pos += take

                if self._current_pos >= self._current.shape[0]:
                    self._current = None

        if written < frames:
            outdata[written:] = 0.0


class AudioCapture(threading.Thread):
    def __init__(self, label: str, device_idx: int, cfg: CaptureConfig, out_q: queue.Queue, monitor: Optional[AudioMonitor] = None):
        super().__init__(daemon=True)
        self.label = label
        self.device_idx = device_idx
        self.cfg = cfg
        self.out_q = out_q
        self.monitor = monitor
        self._stop_event = threading.Event()
        self._recent_dumps: Deque[Path] = deque(maxlen=16)

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        blocksize = FRAME_SAMPLES

        def cb(indata, _frames, _time_info, status):
            if status:
                DUAL_LOG.debug("%s status: %s", self.label, status)
            if self._stop_event.is_set():
                raise sd.CallbackStop()

            frame = indata[:, 0].astype(np.float32)
            pcm16 = np.clip(frame * 32768.0, -32768, 32767).astype(np.int16)

            if self.monitor is not None:
                self.monitor.feed(pcm16.copy())

            self._emit_frame(pcm16)

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            latency="low",
            device=self.device_idx,
            callback=cb,
        ):
            while not self._stop_event.is_set():
                sd.sleep(100)

    def _emit_frame(self, pcm16: np.ndarray) -> None:
        payload = pcm16.tobytes()
        try:
            self.out_q.put_nowait((self.label, payload, {"reason": "frame"}))
        except queue.Full:
            DUAL_LOG.warning("%s dropped frame: queue full", self.label)
            return

        if getattr(self.cfg, "dump_segments", False):
            self._dump_wav(pcm16)

    def _dump_wav(self, pcm: np.ndarray) -> None:
        try:
            WAV_DUMP_ROOT.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            fname = f"{WAV_DUMP_PREFIX}_{self.label.lower()}_{stamp}_{int(time.time() * 1000) % 1000}.wav"
            path = WAV_DUMP_ROOT / fname
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(pcm.tobytes())
            self._recent_dumps.append(path)
            while len(self._recent_dumps) > self._recent_dumps.maxlen:
                old = self._recent_dumps.popleft()
                try:
                    old.unlink(missing_ok=True)
                except Exception:
                    pass
        except Exception as exc:
            DUAL_LOG.warning("%s wav dump failed: %s", self.label, exc)


def _switchaudio_cmd() -> Optional[str]:
    return shutil.which("SwitchAudioSource")


def _switchaudio_current_output() -> Optional[str]:
    cmd = _switchaudio_cmd()
    if not cmd:
        return None
    try:
        res = subprocess.run([cmd, "-c", "-t", "output"], capture_output=True, text=True, check=True)
        return res.stdout.strip() or None
    except subprocess.CalledProcessError as exc:
        DUAL_LOG.warning("SwitchAudioSource current output query failed: %s", exc)
    except FileNotFoundError:
        pass
    return None


def _switchaudio_set_output(target_name: str) -> bool:
    cmd = _switchaudio_cmd()
    if not cmd:
        return False
    try:
        subprocess.run([cmd, "-s", target_name, "-t", "output"], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as exc:
        DUAL_LOG.warning("SwitchAudioSource failed to set output: %s", exc)
    except FileNotFoundError:
        pass
    return False


def _switchaudio_current_input() -> Optional[str]:
    cmd = _switchaudio_cmd()
    if not cmd:
        return None
    try:
        res = subprocess.run([cmd, "-c", "-t", "input"], capture_output=True, text=True, check=True)
        return res.stdout.strip() or None
    except subprocess.CalledProcessError as exc:
        DUAL_LOG.warning("SwitchAudioSource current input query failed: %s", exc)
    except FileNotFoundError:
        pass
    return None


def _switchaudio_set_input(target_name: str) -> bool:
    cmd = _switchaudio_cmd()
    if not cmd:
        return False
    try:
        subprocess.run([cmd, "-s", target_name, "-t", "input"], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as exc:
        DUAL_LOG.warning("SwitchAudioSource failed to set input: %s", exc)
    except FileNotFoundError:
        pass
    return False


def ensure_blackhole_output(target_name: str = "BlackHole 2ch") -> bool:
    preferred = os.environ.get("LOCAL_LLM_SYSTEM_OUTPUT", target_name)
    current = _switchaudio_current_output()
    if current and _match_device(preferred, current):
        return True

    changed = _switchaudio_set_output(preferred)
    if not changed:
        DUAL_LOG.warning(
            "Unable to automatically switch system output to %s. Install switchaudio-osx or set it manually.",
            preferred,
        )
        return False

    current = _switchaudio_current_output()
    return bool(current and _match_device(preferred, current))


def ensure_blackhole_input(target_name: str = "BlackHole 2ch") -> bool:
    current = _switchaudio_current_input()
    if current and _match_device(target_name, current):
        return True

    changed = _switchaudio_set_input(target_name)
    if not changed:
        DUAL_LOG.warning(
            "Unable to automatically switch system input to %s. Install switchaudio-osx or set it manually.",
            target_name,
        )
        return False

    current = _switchaudio_current_input()
    return bool(current and _match_device(target_name, current))


def build_captures(cfg_mic: CaptureConfig, cfg_sys: CaptureConfig) -> Tuple[Optional[int], Optional[int]]:
    devs = list_input_devices()
    default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else None
    mic_idx = default_in if default_in is not None else (devs[0]["index"] if devs else None)
    sys_idx, _ = find_loopback_candidate()
    return mic_idx, sys_idx


# ---------------------------------------------------------------------------
# HTTP transcriber (from http_transcriber.py)


@dataclass
class TimedSegment:
    """Timestamped transcript segment tagged with its source/speaker."""

    label: str
    speaker: str
    text: str
    started_at: float
    ended_at: float
    reason: str

    @property
    def duration(self) -> float:
        return max(0.0, self.ended_at - self.started_at)


@dataclass
class StreamState:
    """Tracks rolling metadata for a capture stream (mic/system)."""

    speaker: str
    current_start: Optional[float] = None
    last_update: Optional[float] = None


class EnergySegmenter:
    """Very lightweight energy gate to chunk audio prior to HTTP transcription."""

    def __init__(self, cfg: SegmentConfig):
        self.cfg = cfg
        self._calibration_frames = max(1, cfg.energy_calibration_ms // FRAME_MS)
        self._calibration_samples: List[float] = []
        self._noise_floor_dbfs = cfg.energy_floor_dbfs
        self._threshold_dbfs = self._noise_floor_dbfs + cfg.energy_offset_db
        self._speaking = False
        self._silence_frames = 0
        self._segment_frames: List[bytes] = []
        self._pre_roll_frames = max(1, cfg.pre_roll_ms // FRAME_MS)
        self._prebuffer: List[bytes] = []
        self._started_frames = 0
        self._min_speech_frames = max(1, cfg.min_speech_ms // FRAME_MS)
        self._max_silence_frames = max(1, cfg.max_silence_ms // FRAME_MS)
        self._max_segment_frames = max(
            self._min_speech_frames + 1,
            int(cfg.max_segment_seconds * 1000 / FRAME_MS),
        )

    @staticmethod
    def _frame_dbfs(pcm16: bytes) -> float:
        samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return -120.0
        rms = np.sqrt(np.mean(np.square(samples))) + 1e-12
        db = 20.0 * np.log10(rms / 32768.0)
        if not np.isfinite(db):
            return -120.0
        return float(db)

    def _update_threshold(self) -> None:
        self._threshold_dbfs = max(self.cfg.energy_floor_dbfs, self._noise_floor_dbfs + self.cfg.energy_offset_db)

    def add_frame(self, pcm16: bytes) -> List[bytes]:
        """Return a list of finalized segments (as PCM bytes) created by this frame."""
        segments: List[bytes] = []
        dbfs = self._frame_dbfs(pcm16)

        if self._calibration_frames > 0:
            self._calibration_samples.append(dbfs)
            self._calibration_frames -= 1
            if self._calibration_frames == 0 and self._calibration_samples:
                baseline = float(np.mean(self._calibration_samples))
                if not np.isfinite(baseline):
                    baseline = self.cfg.energy_floor_dbfs
                self._noise_floor_dbfs = max(self.cfg.energy_floor_dbfs, baseline)
                self._update_threshold()
            self._prebuffer.append(pcm16)
            if len(self._prebuffer) > self._pre_roll_frames:
                self._prebuffer.pop(0)
            return segments

        is_speech = dbfs >= self._threshold_dbfs

        if not self._speaking:
            if is_speech:
                self._speaking = True
                self._silence_frames = 0
                self._started_frames = 0
                if self._prebuffer:
                    self._segment_frames.extend(self._prebuffer)
                self._segment_frames.append(pcm16)
                self._started_frames += 1
                self._prebuffer.clear()
            else:
                self._prebuffer.append(pcm16)
                if len(self._prebuffer) > self._pre_roll_frames:
                    self._prebuffer.pop(0)
                self._noise_floor_dbfs = (0.95 * self._noise_floor_dbfs) + (0.05 * dbfs)
                self._update_threshold()
        else:
            self._segment_frames.append(pcm16)
            self._started_frames += 1
            if is_speech:
                self._silence_frames = 0
                if self._started_frames >= self._max_segment_frames:
                    if self._started_frames >= self._min_speech_frames:
                        segments.append(b"".join(self._segment_frames))
                    self._segment_frames.clear()
                    self._speaking = False
                    self._silence_frames = 0
                    self._started_frames = 0
            else:
                self._silence_frames += 1
                if self._silence_frames >= self._max_silence_frames or self._started_frames >= self._max_segment_frames:
                    if self._started_frames >= self._min_speech_frames:
                        segments.append(b"".join(self._segment_frames))
                    self._segment_frames.clear()
                    self._speaking = False
                    self._silence_frames = 0
                    self._started_frames = 0

        return segments

    def flush(self) -> Optional[bytes]:
        """Flush any pending speech segment."""
        if self._segment_frames and self._started_frames >= self._min_speech_frames:
            data = b"".join(self._segment_frames)
            self._segment_frames.clear()
            self._speaking = False
            self._silence_frames = 0
            self._started_frames = 0
            return data
        self._segment_frames.clear()
        self._speaking = False
        self._silence_frames = 0
        self._started_frames = 0
        return None


class HTTPTranscriber(threading.Thread):
    """Queues PCM frames, chunks via light VAD, and posts to gpt-4o-transcribe."""

    def __init__(
        self,
        label: str,
        on_text: Callable[[str, str, dict], None],
        *,
        config: Optional[HTTPSTTConfig] = None,
        max_queue: int = 2048,
    ):
        super().__init__(daemon=True)
        self.label = label
        self.on_text = on_text
        self.cfg = config or HTTPSTTConfig.from_env()
        self._frames: "queue.Queue[bytes]" = queue.Queue(maxsize=max_queue)
        self._stop_event = threading.Event()
        self._flush_flag = threading.Event()
        self._flush_event = threading.Event()
        self._flush_event.set()
        self._seg = EnergySegmenter(self.cfg.segment)
        self._api_key = load_openai_api_key()
        self._seg_accum: str = ""
        headers: Dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        self._client = httpx.Client(timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None), headers=headers)

    def stop(self) -> None:
        self._stop_event.set()

    def flush_now(self) -> None:
        self._flush_flag.set()

    def flush_and_wait(self, timeout: float = 1.5) -> bool:
        self._flush_event.clear()
        self._flush_flag.set()
        return self._flush_event.wait(timeout)

    def send_frame(self, pcm16: bytes) -> None:
        if not pcm16 or self._stop_event.is_set():
            return
        try:
            self._frames.put_nowait(pcm16)
        except queue.Full:
            try:
                _ = self._frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frames.put_nowait(pcm16)
            except queue.Full:
                HTTP_LOG.warning("%s frame dropped (queue saturated)", self.label)

    def run(self) -> None:
        if not self._api_key:
            HTTP_LOG.error("OPENAI_API_KEY missing; STT disabled for %s", self.label)
            return

        while not self._stop_event.is_set():
            if self._flush_flag.is_set():
                self._flush_flag.clear()
                trailing = self._seg.flush()
                if trailing:
                    self._transcribe_segment(trailing)
                self._flush_event.set()
                continue
            try:
                frame = self._frames.get(timeout=0.1)
            except queue.Empty:
                continue

            segments = self._seg.add_frame(frame)
            for segment in segments:
                self._transcribe_segment(segment)

        trailing = self._seg.flush()
        if trailing:
            self._transcribe_segment(trailing)
        self._flush_event.set()

    # ---- helpers -----------------------------------------------------

    def _transcribe_segment(self, pcm16: bytes) -> None:
        if not pcm16:
            return

        wav_buffer = self._pcm_to_wav(pcm16)
        use_sse = bool(self.cfg.enable_delta_streaming and self._model_uses_sse(self.cfg.model))

        data = {
            "model": self.cfg.model,
            "language": self.cfg.language,
            "stream": use_sse,
            "response_format": "text",
            "temperature": 0,
        }
        if self.cfg.prompt:
            data["prompt"] = self.cfg.prompt

        files = {"file": ("segment.wav", wav_buffer.getvalue(), "audio/wav")}
        self._seg_accum = ""

        try:
            if use_sse:
                HTTP_LOG.info("%s POST %s stream=True model=%s", self.label, self.cfg.endpoint, self.cfg.model)
                stream_args = {"data": data, "files": files, "headers": {"Accept": "text/event-stream"}}
                with self._client.stream("POST", self.cfg.endpoint, **stream_args) as response:
                    if response.status_code != 200:
                        raw = response.read()
                        try:
                            payload = response.json()
                        except Exception:
                            try:
                                payload = json.loads(raw.decode("utf-8", errors="ignore"))
                            except Exception:
                                payload = raw.decode("utf-8", errors="ignore")
                        HTTP_LOG.error("%s transcription error %s: %r", self.label, response.status_code, payload)

                        data["stream"] = False
                        HTTP_LOG.info("%s retry non-streaming", self.label)
                        resp2 = self._client.post(self.cfg.endpoint, data=data, files=files)
                        if resp2.status_code != 200:
                            try:
                                err = resp2.json()
                            except Exception:
                                err = resp2.text
                            HTTP_LOG.error(
                                "%s non-streaming transcription error %s: %r", self.label, resp2.status_code, err
                            )
                            return
                        self._handle_json_response(resp2)
                        return

                    try:
                        for line in response.iter_lines():
                            if not line:
                                continue
                            if isinstance(line, bytes):
                                line = line.decode("utf-8", errors="ignore")
                            line = line.strip()
                            if not line or line.startswith(":") or not line.startswith("data:"):
                                continue
                            payload = line[5:].strip()
                            if payload == "[DONE]":
                                break
                            try:
                                message = json.loads(payload)
                            except json.JSONDecodeError:
                                HTTP_LOG.debug("%s SSE non-JSON payload: %s", self.label, payload)
                                continue
                            self._handle_sse(message)
                    except Exception as exc:
                        HTTP_LOG.error("%s SSE parsing failed: %s", self.label, exc)
            else:
                HTTP_LOG.info("%s POST %s stream=False model=%s", self.label, self.cfg.endpoint, self.cfg.model)
                response = self._client.post(self.cfg.endpoint, data=data, files=files)
                if response.status_code != 200:
                    try:
                        payload = response.json()
                    except Exception:
                        payload = response.text
                    HTTP_LOG.error("%s transcription error %s: %r", self.label, response.status_code, payload)
                    return
                self._handle_json_response(response)
        except Exception as exc:
            HTTP_LOG.error("%s upload failed: %s", self.label, exc)

    def _handle_sse(self, message: dict) -> None:
        msg_type = message.get("type") or message.get("event") or ""
        text = (
            message.get("text")
            or message.get("delta")
            or (message.get("transcript") or {}).get("text")
            or (message.get("transcript") or {}).get("delta")
            or (message.get("data") or {}).get("text")
            or (message.get("data") or {}).get("delta")
        ) or ""
        text = str(text)

        lowered = msg_type.lower()

        if lowered.endswith(".delta") or lowered.endswith("_delta"):
            if not self.cfg.enable_delta_streaming:
                return
            if text:
                if self._seg_accum:
                    self._seg_accum = f"{self._seg_accum} {text}".strip()
                else:
                    self._seg_accum = text
                self.on_text(
                    self.label,
                    self._seg_accum,
                    {"reason": "partial", "language": self.cfg.language},
                )
        elif lowered.endswith(".completed") or lowered.endswith(".done") or lowered.endswith("completed"):
            final_text = (text or message.get("transcript", {}).get("text") or "").strip()
            if not final_text:
                final_text = self._seg_accum.strip()
            if final_text:
                self.on_text(self.label, final_text, {"reason": "endpoint", "language": self.cfg.language})
            self._seg_accum = ""
        elif msg_type == "error" or message.get("error"):
            HTTP_LOG.error("%s SSE error: %s", self.label, message)

    @staticmethod
    def _pcm_to_wav(pcm16: bytes) -> io.BytesIO:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm16)
        buffer.seek(0)
        return buffer

    @staticmethod
    def _model_uses_sse(model: str) -> bool:
        prefix = (model or "").lower()
        return not prefix.startswith("whisper")

    def _handle_json_response(self, response: httpx.Response) -> None:
        language = self.cfg.language
        payload: Optional[object]
        try:
            payload = response.json()
        except Exception:
            payload = response.text

        if isinstance(payload, dict):
            detected = payload.get("language") or payload.get("detected_language")
            if isinstance(detected, str) and detected.strip():
                language = detected.strip()
            final_text = self._extract_json_text(payload)
        else:
            final_text = str(payload or "").strip()

        if final_text:
            self.on_text(self.label, final_text, {"reason": "endpoint", "language": language})
            self._seg_accum = ""

    @staticmethod
    def _extract_json_text(payload: Dict[str, object]) -> str:
        candidates: List[str] = []

        direct = payload.get("text")
        if isinstance(direct, str) and direct.strip():
            candidates.append(direct)

        transcript = payload.get("transcript")
        if isinstance(transcript, dict):
            nested = transcript.get("text") or transcript.get("delta")
            if isinstance(nested, str) and nested.strip():
                candidates.append(nested)

        data_field = payload.get("data")
        if isinstance(data_field, dict):
            for key in ("text", "transcript"):
                nested_val = data_field.get(key)
                if isinstance(nested_val, str) and nested_val.strip():
                    candidates.append(nested_val)
                elif isinstance(nested_val, dict):
                    nested_text = nested_val.get("text") or nested_val.get("delta")
                    if isinstance(nested_text, str) and nested_text.strip():
                        candidates.append(nested_text)

        segments = payload.get("segments")
        if isinstance(segments, list):
            parts: List[str] = []
            for segment in segments:
                if isinstance(segment, dict):
                    text = segment.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
            if parts:
                candidates.append(" ".join(parts))

        for candidate in candidates:
            candidate_text = str(candidate).strip()
            if candidate_text:
                return candidate_text
        return ""


__all__ = [
    "SAMPLE_RATE",
    "FRAME_MS",
    "FRAME_SAMPLES",
    "list_input_devices",
    "list_output_devices",
    "find_loopback_candidate",
    "find_monitor_device",
    "ensure_blackhole_output",
    "ensure_blackhole_input",
    "build_captures",
    "CaptureConfig",
    "AudioMonitor",
    "AudioCapture",
    "TimedSegment",
    "StreamState",
    "EnergySegmenter",
    "HTTPTranscriber",
]
