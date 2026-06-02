#!/usr/bin/env python3
"""Transcribe + summarise an audio file locally using Transformers + Voxtral.

Usage (with uv):
  uv run main.py ./assets/test_1.wav
  uv run main.py ./assets/test_1.wav --diarize   # with speaker diarization

What this script does:
  1) Loads Voxtral in-process (no separate server).
  2) Optionally runs pyannote speaker diarization to identify who is speaking.
  3) Runs transcription mode for best ASR accuracy (per-speaker if diarized).
  4) Frees MPS memory aggressively (gc + torch.mps.empty_cache).
  5) Summarises the transcript in chunks to keep context/memory bounded.

Optimisations:
  - Apple Silicon MPS has a hard allocator watermark; long generations and huge
    contexts can OOM even for small extra allocations.
  - Chunking + lower max_new_tokens reduces KV-cache pressure.
Notes:
  - No separate model server is required; Transformers runs in-process.
  - Chunking uses soundfile when possible; for unsupported formats (e.g. some mp3/m4a builds),
    it will fall back to non-chunked transcription unless you have ffmpeg installed.
  - Diarization uses pyannote-audio. You must accept terms at
    https://huggingface.co/pyannote/speaker-diarization-3.1 and set HF_TOKEN.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, cast

import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration

logger = logging.getLogger("voxtral")


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def pick_device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    # fp16 on MPS reduces memory; CPU uses fp32 for compatibility.
    return torch.float16 if device.type == "mps" else torch.float32


def mps_cleanup(tag: str = "") -> None:
    """Best-effort memory cleanup between steps on MPS."""
    if tag:
        logger.debug("Cleanup requested (%s)", tag)
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ── Typed wrappers for third-party libs with incomplete/missing stubs ──


def _sf_read(path: str) -> tuple[Any, int]:
    """Read audio file via soundfile, returning (samples, sample_rate)."""
    import soundfile as _sf  # type: ignore[import-untyped]

    sf = cast(Any, _sf)
    data, sr = sf.read(path, always_2d=False)
    return data, int(sr)


def _sf_write(path: str, data: Any, samplerate: int) -> None:
    """Write audio data to file via soundfile."""
    import soundfile as _sf  # type: ignore[import-untyped]

    sf = cast(Any, _sf)
    sf.write(path, data, samplerate)


def _sf_importable() -> bool:
    """Check whether the soundfile package can be imported."""
    try:
        import soundfile  # type: ignore[import-untyped]  # noqa: F401

        return True
    except ImportError:
        return False


def _resample(audio: Any, from_sr: int, to_sr: int) -> Any:
    """Resample an audio array via soxr."""
    import soxr as _soxr  # type: ignore[import-untyped]

    mod = cast(Any, _soxr)
    return mod.resample(audio, from_sr, to_sr)


@dataclass
class SpeakerSegment:
    """A segment of audio attributed to a single speaker."""
    speaker: str
    start: float  # seconds
    end: float    # seconds
    text: str = ""


def run_diarization(
    audio_path: Path,
    device: torch.device,
    hf_token: Optional[str] = None,
    num_speakers: Optional[int] = None,
) -> List[SpeakerSegment]:
    """Run pyannote speaker diarization and return speaker segments."""
    from pyannote.audio import Pipeline

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "Diarization requires a HuggingFace token. "
            "Set HF_TOKEN env var or pass --hf-token. "
            "You must also accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1"
        )

    logger.info("STAGE: speaker diarization")
    pipeline = cast(Any, Pipeline).from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=token,
    )
    if pipeline is None:
        raise SystemExit("Failed to load pyannote diarization pipeline")

    # pyannote supports mps/cuda/cpu
    if device.type in ("mps", "cuda"):
        pipeline.to(device)

    kwargs: dict[str, Any] = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers

    diarization = pipeline(str(audio_path), **kwargs)

    # Newer pyannote returns DiarizeOutput; extract the Annotation object.
    annotation = getattr(diarization, "speaker_diarization", diarization)

    segments: List[SpeakerSegment] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(SpeakerSegment(
            speaker=speaker,
            start=turn.start,
            end=turn.end,
        ))

    logger.info("Diarization found %d segments with %d speakers",
                len(segments), len({s.speaker for s in segments}))

    # Free diarization pipeline memory
    del pipeline
    del diarization
    mps_cleanup("after diarization")

    return segments


def extract_segment_audio(
    audio_path: Path,
    segment: SpeakerSegment,
    out_path: Path,
) -> Path:
    """Extract a time segment from audio to a WAV file."""
    try:
        import numpy as np

        audio, sr = _sf_read(str(audio_path))
        if hasattr(audio, "ndim") and audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        start_sample = int(segment.start * sr)
        end_sample = int(segment.end * sr)
        chunk: Any = audio[start_sample:end_sample]
        _sf_write(str(out_path), chunk, sr)
        return out_path
    except Exception:
        pass

    # Fallback to ffmpeg
    if _have_ffmpeg():
        duration = segment.end - segment.start
        cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-ss", str(segment.start),
            "-t", str(duration),
            "-ac", "1", "-ar", "16000",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_path

    raise RuntimeError("Cannot extract audio segment: soundfile and ffmpeg both failed")


def merge_adjacent_segments(
    segments: List[SpeakerSegment],
    max_gap: float = 0.5,
) -> List[SpeakerSegment]:
    """Merge consecutive segments from the same speaker if gap is small."""
    if not segments:
        return []
    merged = [SpeakerSegment(
        speaker=segments[0].speaker,
        start=segments[0].start,
        end=segments[0].end,
    )]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg.speaker == prev.speaker and (seg.start - prev.end) <= max_gap:
            prev.end = seg.end
        else:
            merged.append(SpeakerSegment(
                speaker=seg.speaker,
                start=seg.start,
                end=seg.end,
            ))
    return merged


def load_summary_prompt(prompt_path: Optional[Path]) -> str:
    """Load a summary prompt from a file. Falls back to a built-in default."""
    if prompt_path is None:
        # Default: look next to the script in assets/
        prompt_path = Path(__file__).resolve().parent / "assets" / "call_summarisation_prompt.md"
    prompt_path = prompt_path.expanduser().resolve()
    if prompt_path.exists():
        logger.info("Loaded summary prompt from %s", prompt_path)
        return prompt_path.read_text(encoding="utf-8").strip()
    logger.warning("Summary prompt file not found: %s — using built-in default", prompt_path)
    return (
        "Summarise the following transcript.\n\n"
        "Return:\n"
        "1) A 1-paragraph executive summary\n"
        "2) 5-10 bullet points of key details"
    )


def chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if chunk_chars <= 0:
        return [text]
    overlap = max(0, min(overlap, chunk_chars // 2))

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _ffmpeg_convert_to_wav(src: Path, dst: Path) -> None:
    """Convert audio to 16kHz mono WAV using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(dst),
    ]
    logger.info("Converting with ffmpeg: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load_audio_mono_16k(path: Path) -> Tuple[Any, Optional[int]]:
    """Load audio using soundfile if available.

    Returns:
      (audio_array, sample_rate) or (None, None) if unsupported.

    We import soundfile lazily so users who only do non-chunked transcription aren't blocked.
    """
    try:
        audio, sr = _sf_read(str(path))
    except ImportError as e:
        logger.warning("soundfile not available (%s). Chunked transcription may be limited.", e)
        return None, None
    except Exception as e:
        logger.warning("soundfile could not read %s (%s)", path, e)
        return None, None

    # Convert to mono if needed
    try:
        import numpy as np
        if hasattr(audio, "ndim") and audio.ndim == 2:
            audio = np.mean(audio, axis=1)
    except Exception:
        # If numpy isn't there (unlikely), just keep as-is.
        pass

    # Resample to 16k if needed (optional). Many pipelines accept other sample rates,
    # but for stable chunking we target 16k.
    if sr != 16000:
        try:
            audio = _resample(audio, sr, 16000)
            sr = 16000
        except Exception as e:
            logger.warning(
                "Could not resample from %s Hz to 16000 Hz (missing soxr?). Proceeding with %s Hz. (%s)",
                sr,
                sr,
                e,
            )

    return audio, sr


def split_audio_to_wav_chunks(
    audio_path: Path,
    chunk_seconds: int,
    overlap_seconds: int,
) -> Tuple[List[Path], Optional[tempfile.TemporaryDirectory[str]]]:
    """Split audio into N-second WAV chunks and return chunk file paths.

    - Prefers soundfile for reading.
    - If soundfile cannot read the format, tries ffmpeg to convert to WAV then chunk.

    Returns:
      (chunk_paths, tempdir_handle)

    tempdir_handle must be kept alive until you're done using chunk_paths.
    """
    if chunk_seconds <= 0:
        return [audio_path], None

    tmpdir = tempfile.TemporaryDirectory(prefix="voxtral_chunks_")
    tmp_path = Path(tmpdir.name)

    # First try reading directly.
    audio, sr = load_audio_mono_16k(audio_path)

    # If direct read fails, try ffmpeg convert and read again.
    converted = None
    if audio is None:
        if _have_ffmpeg():
            converted = tmp_path / "converted.wav"
            try:
                _ffmpeg_convert_to_wav(audio_path, converted)
                audio, sr = load_audio_mono_16k(converted)
            except Exception as e:
                logger.error("ffmpeg conversion failed: %s", e)
                audio = None
        else:
            logger.warning(
                "Cannot chunk %s (unsupported by soundfile) and ffmpeg not found. Falling back to single-file transcription.",
                audio_path,
            )
            tmpdir.cleanup()
            return [audio_path], None

    if audio is None or sr is None:
        logger.warning("Audio could not be loaded for chunking. Falling back to single-file transcription.")
        tmpdir.cleanup()
        return [audio_path], None

    # Write chunks as wav files
    if not _sf_importable():
        logger.warning("soundfile is required to write chunks. Falling back to single-file transcription.")
        tmpdir.cleanup()
        return [audio_path], None

    import math

    total_samples = len(audio)
    chunk_samples = int(chunk_seconds * sr)
    overlap_samples = int(max(0, overlap_seconds) * sr)
    step = max(1, chunk_samples - overlap_samples)

    num_chunks = math.ceil(max(1, total_samples - overlap_samples) / step)
    logger.info(
        "Chunking audio: sr=%sHz, total=%.1fs, chunk=%ss, overlap=%ss -> ~%s chunks",
        sr,
        total_samples / sr,
        chunk_seconds,
        overlap_seconds,
        num_chunks,
    )

    chunk_paths: List[Path] = []
    start = 0
    idx = 0
    while start < total_samples:
        end = min(total_samples, start + chunk_samples)
        chunk: Any = audio[start:end]
        out = tmp_path / f"chunk_{idx:04d}.wav"
        _sf_write(str(out), chunk, sr)
        chunk_paths.append(out)
        if end >= total_samples:
            break
        start += step
        idx += 1

    # keep tmpdir alive
    return chunk_paths, tmpdir


@torch.inference_mode()
def transcribe_one(
    model: VoxtralForConditionalGeneration,
    processor: Any,
    device: torch.device,
    repo_id: str,
    audio_path: Path,
    language: Optional[str],
    max_new_tokens: int,
) -> str:
    """Transcribe a single audio file path using Voxtral transcription mode."""
    if language:
        inputs = processor.apply_transcription_request(
            language=language,
            audio=str(audio_path),
            model_id=repo_id,
        )
    else:
        inputs = processor.apply_transcription_request(
            audio=str(audio_path),
            model_id=repo_id,
        )

    inputs = inputs.to(device)

    # NOTE: Do not pass temperature here; transcription is greedy (do_sample=False)
    # and Transformers may warn that temperature is ignored.
    outputs = cast(Any, model).generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    text = processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1] :],
        skip_special_tokens=True,
    )[0].strip()

    del outputs
    del inputs
    mps_cleanup("after transcribe_one")

    return text


@torch.inference_mode()
def transcribe_audio(
    model: VoxtralForConditionalGeneration,
    processor: Any,
    device: torch.device,
    repo_id: str,
    audio_path: Path,
    language: Optional[str],
    max_new_tokens: int,
    chunk_seconds: int,
    chunk_overlap_seconds: int,
) -> str:
    """Chunked transcription wrapper."""
    logger.info("STAGE: transcription")

    if chunk_seconds > 0:
        chunk_paths, tmpdir = split_audio_to_wav_chunks(
            audio_path, chunk_seconds=chunk_seconds, overlap_seconds=chunk_overlap_seconds
        )
    else:
        chunk_paths, tmpdir = [audio_path], None

    try:
        parts: List[str] = []
        for i, p in enumerate(chunk_paths, start=1):
            logger.info("Transcribing chunk %s/%s: %s", i, len(chunk_paths), p.name)
            part = transcribe_one(
                model=model,
                processor=processor,
                device=device,
                repo_id=repo_id,
                audio_path=p,
                language=language,
                max_new_tokens=max_new_tokens,
            )
            parts.append(part)

        transcript = "\n".join([t for t in parts if t])
        logger.info("Transcription complete (chars=%s)", len(transcript))
        return transcript
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()


@torch.inference_mode()
def generate_text(
    model: VoxtralForConditionalGeneration,
    processor: Any,
    device: torch.device,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    conversation: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    inputs = processor.apply_chat_template(conversation)
    inputs = inputs.to(device)

    outputs = cast(Any, model).generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=(temperature > 0),
    )

    text = processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1] :],
        skip_special_tokens=True,
    )[0].strip()

    del outputs
    del inputs
    mps_cleanup("after generate_text")

    return text


def write_output_file(
    out_path: Path,
    audio_path: Path,
    model_id: str,
    transcript: str,
    summary: str,
    diarized: bool,
) -> None:
    """Write a structured Markdown report to *out_path*."""
    sep = "─" * 60
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections = [
        f"# Transcription Report",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| **Audio file** | `{audio_path.name}` |",
        f"| **Model** | `{model_id}` |",
        f"| **Date** | {now} |",
        f"| **Diarization** | {'Yes' if diarized else 'No'} |",
        "",
        sep,
        "",
        "## Transcript",
        "",
        transcript,
        "",
        sep,
        "",
        "## Summary",
        "",
        summary,
        "",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sections), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_path", type=Path, help="Path to audio file (wav/mp3/m4a/...) ")
    ap.add_argument("--model", default="mistralai/Voxtral-Mini-3B-2507", help="HF model id")
    ap.add_argument(
        "--language",
        default=None,
        help="Optional language hint (e.g. en, fr). Omit for auto-detect.",
    )

    # Lower defaults to reduce MPS pressure.
    ap.add_argument("--max-new-tokens-transcript", type=int, default=512)
    ap.add_argument("--max-new-tokens-summary", type=int, default=256)

    # Chunking for transcription
    ap.add_argument(
        "--transcribe-chunk-seconds",
        type=int,
        default=30,
        help="Split audio into N-second chunks for transcription. Use 0 to disable.",
    )
    ap.add_argument(
        "--transcribe-chunk-overlap-seconds",
        type=int,
        default=0,
        help="Overlap between transcription chunks (seconds).",
    )

    # Chunking for summarisation
    ap.add_argument("--chunk-chars", type=int, default=4000, help="Chars per summary chunk")
    ap.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between summary chunks")

    # CPU fallbacks
    ap.add_argument(
        "--transcribe-on-cpu",
        action="store_true",
        help="Run the transcription step on CPU (slower, avoids MPS OOM).",
    )
    ap.add_argument(
        "--summary-on-cpu",
        action="store_true",
        help="Run the summarisation step on CPU (slower, avoids MPS OOM).",
    )

    # Diarization
    ap.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization (requires pyannote-audio + HF_TOKEN).",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Hint for number of speakers (optional, pyannote auto-detects if omitted).",
    )
    ap.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token for pyannote models. Falls back to HF_TOKEN env var.",
    )

    # Output
    ap.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file path. Defaults to <audio_stem>_voxtral_<timestamp>.md",
    )

    # Summary prompt
    ap.add_argument(
        "--summary-prompt",
        type=Path,
        default=None,
        help="Path to summary prompt file. Defaults to assets/call_summarisation_prompt.md",
    )

    # Logging
    ap.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level: DEBUG, INFO, WARNING, ERROR",
    )

    args = ap.parse_args()
    setup_logging(args.log_level)

    audio_path: Path = args.audio_path.expanduser().resolve()
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    logger.info("STAGE: load processor")
    repo_id = args.model
    processor: Any = cast(Any, AutoProcessor).from_pretrained(repo_id)

    base_device = pick_device()
    base_dtype = pick_dtype(base_device)

    logger.info("STAGE: load model")
    logger.info("Base device=%s dtype=%s", base_device, base_dtype)

    model: Any = cast(Any, VoxtralForConditionalGeneration).from_pretrained(
        repo_id, torch_dtype=base_dtype
    )
    model.to(base_device)
    model.eval()

    # ---- DIARIZE (optional) ----
    diarized_segments: Optional[List[SpeakerSegment]] = None
    if args.diarize:
        diarized_segments = run_diarization(
            audio_path=audio_path,
            device=base_device,
            hf_token=args.hf_token,
            num_speakers=args.num_speakers,
        )
        diarized_segments = merge_adjacent_segments(diarized_segments)

    # ---- TRANSCRIBE ----
    transcribe_device = torch.device("cpu") if args.transcribe_on_cpu else base_device
    if transcribe_device.type != base_device.type:
        logger.info("Moving model to %s for transcription", transcribe_device)
        model.to(transcribe_device)
        mps_cleanup("after move to transcribe_device")

    if diarized_segments:
        # Transcribe per speaker segment
        logger.info("STAGE: diarized transcription (%d segments)", len(diarized_segments))
        tmpdir = tempfile.TemporaryDirectory(prefix="voxtral_diarize_")
        try:
            for i, seg in enumerate(diarized_segments, start=1):
                logger.info(
                    "Transcribing segment %d/%d [%s] %.1fs-%.1fs",
                    i, len(diarized_segments), seg.speaker, seg.start, seg.end,
                )
                seg_path = Path(tmpdir.name) / f"seg_{i:04d}.wav"
                extract_segment_audio(audio_path, seg, seg_path)
                seg.text = transcribe_one(
                    model=model,
                    processor=processor,
                    device=transcribe_device,
                    repo_id=repo_id,
                    audio_path=seg_path,
                    language=args.language,
                    max_new_tokens=args.max_new_tokens_transcript,
                )
        finally:
            tmpdir.cleanup()

        transcript = "\n".join(
            f"[{seg.speaker}]: {seg.text}" for seg in diarized_segments if seg.text
        )
        logger.info("Diarized transcription complete (chars=%s)", len(transcript))
    else:
        transcript = transcribe_audio(
            model=model,
            processor=processor,
            device=transcribe_device,
            repo_id=repo_id,
            audio_path=audio_path,
            language=args.language,
            max_new_tokens=args.max_new_tokens_transcript,
            chunk_seconds=args.transcribe_chunk_seconds,
            chunk_overlap_seconds=args.transcribe_chunk_overlap_seconds,
        )

    # ---- SUMMARISE ----
    logger.info("STAGE: summarisation")

    summary_device = torch.device("cpu") if args.summary_on_cpu else base_device

    if summary_device.type != transcribe_device.type:
        logger.info("Moving model to %s for summarisation", summary_device)
        model.to(summary_device)
        mps_cleanup("after move to summary_device")

    summary_prompt = load_summary_prompt(args.summary_prompt)
    chunks = chunk_text(transcript, args.chunk_chars, args.chunk_overlap)

    if not chunks:
        summary = "(No transcript text to summarise.)"
    elif len(chunks) == 1:
        prompt = (
            f"{summary_prompt}\n\n"
            f"TRANSCRIPT:\n{chunks[0]}"
        )
        summary = generate_text(
            model=model,
            processor=processor,
            device=summary_device,
            prompt_text=prompt,
            max_new_tokens=args.max_new_tokens_summary,
            temperature=0.2,
        )
    else:
        partials: List[str] = []
        for i, chunk in enumerate(chunks, start=1):
            logger.info("Summarising transcript chunk %s/%s", i, len(chunks))
            prompt = (
                f"{summary_prompt}\n\n"
                f"Summarise part {i}/{len(chunks)} of this transcript "
                "based on the instructions above. "
                "Return concise bullet points focusing on facts, names, dates, and key events.\n\n"
                f"TRANSCRIPT PART:\n{chunk}"
            )
            part = generate_text(
                model=model,
                processor=processor,
                device=summary_device,
                prompt_text=prompt,
                max_new_tokens=args.max_new_tokens_summary,
                temperature=0.2,
            )
            partials.append(part)

        combined = "\n\n".join(
            f"PART {i+1} SUMMARY:\n{p}" for i, p in enumerate(partials)
        )

        final_prompt = (
            f"{summary_prompt}\n\n"
            "You are given summaries of transcript chunks. "
            "Produce a final consolidated summary based on the instructions above.\n\n"
            f"CHUNK SUMMARIES:\n{combined}"
        )

        summary = generate_text(
            model=model,
            processor=processor,
            device=summary_device,
            prompt_text=final_prompt,
            max_new_tokens=args.max_new_tokens_summary,
            temperature=0.2,
        )

    # Determine output path
    if args.output:
        out_path = args.output.expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{audio_path.stem}_voxtral_{ts}.md"

    write_output_file(
        out_path=out_path,
        audio_path=audio_path,
        model_id=repo_id,
        transcript=transcript,
        summary=summary,
        diarized=args.diarize,
    )
    logger.info("Output written to %s", out_path)


if __name__ == "__main__":
    main()
