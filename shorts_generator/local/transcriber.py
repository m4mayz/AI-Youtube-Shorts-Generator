"""Local transcription via faster-whisper.

Reads a local media file and returns the same shape the highlight generator
expects: {duration, segments[start, end, text]}.
"""
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

from ..config import (
    GROQ_BASE_URL,
    GROQ_TRANSCRIBE_MODEL,
    LOCAL_FFMPEG_PATH,
    LOCAL_OUTPUT_DIR,
    LOCAL_WHISPER_DEVICE,
    LOCAL_WHISPER_MODEL,
    TRANSCRIBER_PROVIDER,
    require_groq_keys,
)


_GROQ_CHUNK_SECONDS = 1800


def _transcript_cache_path(media_path: str, provider: str = "faster-whisper") -> Path:
    """Return the .srt cache path for a media file."""
    cache_dir = Path(LOCAL_OUTPUT_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".groq.srt" if provider == "groq" else ".srt"
    return cache_dir / (Path(media_path).stem + suffix)


def _format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _parse_srt_timestamp(value: str) -> float:
    match = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", value.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value!r}")
    hours, minutes, seconds, millis = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds + (millis / 1000.0)


def _write_srt_cache(media_path: str, transcript: Dict, provider: str) -> Path:
    cache_path = _transcript_cache_path(media_path, provider)
    lines = []
    for idx, segment in enumerate(transcript.get("segments", []), start=1):
        start = _format_srt_timestamp(float(segment["start"]))
        end = _format_srt_timestamp(float(segment["end"]))
        text = str(segment.get("text", "")).strip().replace("\r", "").replace("\n", " ")
        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    cache_path.write_text("\n".join(lines), encoding="utf-8")
    return cache_path


def _load_srt_cache(cache_path: Path) -> Dict:
    content = cache_path.read_text(encoding="utf-8-sig").strip()
    if not content:
        return {"duration": 0.0, "segments": []}

    segments = []
    for block in re.split(r"\n\s*\n", content):
        lines = [line.strip("\ufeff") for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if "-->" not in lines[0] and len(lines) > 1 and "-->" in lines[1]:
            lines = lines[1:]
        if not lines or "-->" not in lines[0]:
            continue
        start_raw, end_raw = [part.strip() for part in lines[0].split("-->", 1)]
        text = "\n".join(lines[1:]).strip()
        segments.append(
            {
                "start": _parse_srt_timestamp(start_raw),
                "end": _parse_srt_timestamp(end_raw),
                "text": text,
            }
        )

    duration = segments[-1]["end"] if segments else 0.0
    return {"duration": duration, "segments": segments}


def _resolve_device() -> str:
    if LOCAL_WHISPER_DEVICE != "auto":
        return LOCAL_WHISPER_DEVICE
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            # Test that CUDA actually works (catches missing cuBLAS/cuDNN libs)
            torch.zeros(1, device="cuda")
            return "cuda"
    except (ImportError, OSError, RuntimeError):
        pass
    return "cpu"


def _groq_segments(response, offset: float) -> list:
    """Normalize Groq SDK segments and restore source-video timestamps."""
    raw_segments = getattr(response, "segments", None)
    if raw_segments is None and isinstance(response, dict):
        raw_segments = response.get("segments")

    segments = []
    for segment in raw_segments or []:
        if isinstance(segment, dict):
            start, end, text = segment.get("start"), segment.get("end"), segment.get("text")
        else:
            start = getattr(segment, "start", None)
            end = getattr(segment, "end", None)
            text = getattr(segment, "text", "")
        if start is None or end is None:
            continue
        segments.append({
            "start": float(start) + offset,
            "end": float(end) + offset,
            "text": str(text or "").strip(),
        })
    return segments


def _is_retryable_groq_error(error: Exception) -> bool:
    """Only rotate credentials for failures another key can plausibly fix."""
    status_code = getattr(error, "status_code", None)
    if status_code in {401, 403, 408, 409, 429, 500, 502, 503, 504}:
        return True
    return error.__class__.__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }


def _transcribe_groq(media_path: str, language: Optional[str]) -> Dict:
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("Install local dependencies with: pip install -r requirements-local.txt") from error

    keys = require_groq_keys()
    clients = [OpenAI(api_key=key, base_url=GROQ_BASE_URL) for key in keys]
    active_key = 0
    print(
        f"[transcribe/groq] model={GROQ_TRANSCRIBE_MODEL} keys={len(keys)}",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix="groq_transcribe_") as temp_dir:
        chunk_pattern = str(Path(temp_dir) / "chunk_%03d.mp3")
        subprocess.run([
            LOCAL_FFMPEG_PATH, "-y", "-loglevel", "error",
            "-i", media_path,
            "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "libmp3lame", "-b:a", "64k",
            "-f", "segment", "-segment_time", str(_GROQ_CHUNK_SECONDS),
            "-reset_timestamps", "1", chunk_pattern,
        ], check=True)

        chunks = sorted(Path(temp_dir).glob("chunk_*.mp3"))
        if not chunks:
            raise RuntimeError("FFmpeg extracted no audio for Groq transcription.")

        segments = []
        for index, chunk_path in enumerate(chunks):
            print(f"[transcribe/groq] chunk {index + 1}/{len(chunks)}", flush=True)
            while True:
                try:
                    with chunk_path.open("rb") as audio_file:
                        request = {
                            "file": audio_file,
                            "model": GROQ_TRANSCRIBE_MODEL,
                            "response_format": "verbose_json",
                            "timestamp_granularities": ["segment"],
                            "temperature": 0,
                        }
                        if language:
                            request["language"] = language
                        response = clients[active_key].audio.transcriptions.create(**request)
                    break
                except Exception as error:
                    if not _is_retryable_groq_error(error) or active_key + 1 >= len(clients):
                        raise
                    active_key += 1
                    print(
                        f"[transcribe/groq] key failed; switching to "
                        f"{active_key + 1}/{len(clients)}",
                        flush=True,
                    )
            # ponytail: fixed chunks can split one word; add overlap/dedup only
            # if real transcripts show errors at the 30-minute boundaries.
            segments.extend(_groq_segments(response, index * _GROQ_CHUNK_SECONDS))

    duration = segments[-1]["end"] if segments else 0.0
    print(f"[transcribe/groq] {len(segments)} segments, {duration:.0f}s of audio", flush=True)
    return {"duration": duration, "segments": segments}


def transcribe_local(media_path: str, language: Optional[str] = None) -> Dict:
    """Transcribe a local file with the configured provider and cache as SRT."""
    provider = TRANSCRIBER_PROVIDER
    if provider not in {"faster-whisper", "groq"}:
        raise ValueError("TRANSCRIBER_PROVIDER must be 'faster-whisper' or 'groq'.")

    cache_path = _transcript_cache_path(media_path, provider)
    if cache_path.exists():
        source_mtime = os.path.getmtime(media_path)
        cache_mtime = cache_path.stat().st_mtime
        if cache_mtime >= source_mtime:
            print(f"[transcribe/local] reusing cached transcript: {cache_path}", flush=True)
            cached = _load_srt_cache(cache_path)
            # Treat empty cache as invalid (likely from a failed/partial run) — delete and re-transcribe
            if not cached["segments"] or cached["duration"] <= 0.0:
                print(f"[transcribe/local] cache is empty/invalid, deleting: {cache_path}", flush=True)
                cache_path.unlink(missing_ok=True)
            else:
                print(
                    f"[transcribe/local] {len(cached['segments'])} cached segments, "
                    f"{cached['duration']:.0f}s of audio",
                    flush=True,
                )
                return cached

    if provider == "groq":
        transcript = _transcribe_groq(media_path, language)
        cache_path = _write_srt_cache(media_path, transcript, provider)
        print(f"[transcribe/groq] wrote cache: {cache_path}", flush=True)
        return transcript

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "faster-whisper is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    device = _resolve_device()
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"[transcribe/local] faster-whisper model={LOCAL_WHISPER_MODEL} device={device}", flush=True)

    from ..config import LOCAL_WHISPER_VAD_FILTER, LOCAL_WHISPER_VAD_PARAMETERS

    model = WhisperModel(LOCAL_WHISPER_MODEL, device=device, compute_type=compute_type)

    transcribe_kwargs = {
        "audio": media_path,
        "language": language,
        "beam_size": 5,
        "condition_on_previous_text": False,
    }
    if LOCAL_WHISPER_VAD_FILTER:
        transcribe_kwargs["vad_filter"] = True
        transcribe_kwargs["vad_parameters"] = LOCAL_WHISPER_VAD_PARAMETERS
    else:
        transcribe_kwargs["vad_filter"] = False

    segments_iter, info = model.transcribe(**transcribe_kwargs)

    segments = []
    for s in segments_iter:
        segments.append({
            "start": float(s.start),
            "end": float(s.end),
            "text": (s.text or "").strip(),
        })

    duration = float(getattr(info, "duration", 0.0)) or (segments[-1]["end"] if segments else 0.0)
    print(f"[transcribe/local] {len(segments)} segments, {duration:.0f}s of audio", flush=True)
    transcript = {"duration": duration, "segments": segments}
    cache_path = _write_srt_cache(media_path, transcript, provider)
    print(f"[transcribe/local] wrote cache: {cache_path}", flush=True)
    return transcript
