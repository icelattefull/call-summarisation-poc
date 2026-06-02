#!/usr/bin/env python3
"""Transcribe + diarize + summarise an audio file using VibeVoice-ASR.

Usage (with uv):
  uv run vibevoice.py ./assets/test_1.wav
  uv run vibevoice.py ./assets/test_1.wav --prompt "Meeting about Project Alpha"

What this script does:
  1) Loads VibeVoice-ASR-HF in-process via Transformers (no separate server).
  2) Transcribes audio with built-in speaker diarization and timestamps.
  3) For audio >60 minutes, splits into chunks and transcribes each.
  4) Optionally summarises the transcript using a separate LLM call.

Notes:
  - VibeVoice-ASR natively produces speaker IDs and timestamps—no pyannote needed.
  - The model is ~8B params (based on Qwen2.5-7B). MPS uses float32.
  - Supports 50+ languages with auto-detection and code-switching.
  - tokenizer_chunk_size can be reduced if you hit OOM on MPS.
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
from transformers import BatchFeature, VibeVoiceAsrForConditionalGeneration, VibeVoiceAsrProcessor

logger = logging.getLogger("vibevoice")


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    # MPS / CPU need float32
    return torch.float32


def get_system_memory_gb() -> float:
    """Get total physical memory in GB (macOS / Linux)."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
    except (ValueError, OSError, AttributeError):
        return 0.0


def cleanup(tag: str = "") -> None:
    if tag:
        logger.debug("Cleanup requested (%s)", tag)
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Audio chunking (for files >60 min)
# ---------------------------------------------------------------------------

def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def get_audio_duration(path: Path) -> Optional[float]:
    """Get duration in seconds using soundfile or ffmpeg."""
    try:
        import soundfile as sf  # type: ignore[reportMissingTypeStubs]
        return float(sf.info(str(path)).duration)  # type: ignore[reportUnknownMemberType]
    except Exception:
        pass

    if _have_ffmpeg():
        import json as _json
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
                capture_output=True, text=True, check=True,
            )
            data = _json.loads(result.stdout)
            return float(data["format"]["duration"])
        except Exception:
            pass

    return None


def split_audio_file(
    audio_path: Path,
    max_seconds: int,
    overlap_seconds: int = 0,
) -> Tuple[List[Path], Optional[tempfile.TemporaryDirectory[str]]]:
    """Split audio into chunks of max_seconds. Returns (chunk_paths, tmpdir)."""
    duration = get_audio_duration(audio_path)
    if duration is None or duration <= max_seconds:
        return [audio_path], None

    tmpdir = tempfile.TemporaryDirectory(prefix="vibevoice_chunks_")
    tmp_path = Path(tmpdir.name)

    # Try soundfile first
    try:
        import soundfile as sf  # type: ignore[reportMissingTypeStubs]
        import numpy as np
        import numpy.typing as npt

        _raw: tuple[npt.NDArray[np.float64], int] = sf.read(str(audio_path), always_2d=False)  # type: ignore[reportUnknownMemberType]
        audio: npt.NDArray[np.float64] = np.asarray(_raw[0])
        sr: int = _raw[1]
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        chunk_samples = int(max_seconds * sr)
        overlap_samples = int(max(0, overlap_seconds) * sr)
        step = max(1, chunk_samples - overlap_samples)

        num_chunks = math.ceil(max(1, len(audio) - overlap_samples) / step)
        logger.info(
            "Splitting audio: %.1fs total, %ds chunks, %ds overlap -> ~%d chunks",
            len(audio) / sr, max_seconds, overlap_seconds, num_chunks,
        )

        chunk_paths: List[Path] = []
        start = 0
        idx = 0
        while start < len(audio):
            end = min(len(audio), start + chunk_samples)
            out = tmp_path / f"chunk_{idx:04d}.wav"
            sf.write(str(out), audio[start:end], sr)  # type: ignore[reportUnknownMemberType]
            chunk_paths.append(out)
            if end >= len(audio):
                break
            start += step
            idx += 1

        return chunk_paths, tmpdir
    except Exception as e:
        logger.warning("soundfile chunking failed (%s), trying ffmpeg", e)

    # Fallback: ffmpeg segment
    if _have_ffmpeg():
        num_chunks = math.ceil(duration / max_seconds)
        logger.info("Splitting with ffmpeg: %.1fs total -> ~%d chunks", duration, num_chunks)

        chunk_paths = []
        for idx in range(num_chunks):
            start_time = idx * max_seconds
            out = tmp_path / f"chunk_{idx:04d}.wav"
            cmd = [
                "ffmpeg", "-y",
                "-i", str(audio_path),
                "-ss", str(start_time),
                "-t", str(max_seconds),
                "-ac", "1", "-ar", "24000",
                str(out),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            chunk_paths.append(out)

        return chunk_paths, tmpdir

    logger.warning("Cannot split audio (no soundfile or ffmpeg). Processing as single file.")
    tmpdir.cleanup()
    return [audio_path], None


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

@torch.inference_mode()
def transcribe_chunk(
    model: VibeVoiceAsrForConditionalGeneration,
    processor: VibeVoiceAsrProcessor,
    device: torch.device,
    dtype: torch.dtype,
    audio_path: Path,
    prompt: Optional[str],
    max_new_tokens: int,
    tokenizer_chunk_size: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Transcribe a single audio file. Returns (raw_text, parsed_segments)."""
    inputs = processor.apply_transcription_request(audio=str(audio_path), prompt=prompt)
    inputs = inputs.to(device, dtype)  # type: ignore[reportUnknownMemberType]

    generate_kwargs: dict[str, int | bool] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    # tokenizer_chunk_size controls internal audio chunking (samples at 24kHz).
    # Only pass it if the model's generate() actually accepts it (added in
    # newer transformers builds); otherwise the default (60s) is used.
    if tokenizer_chunk_size != 1440000:
        generate_kwargs["tokenizer_chunk_size"] = tokenizer_chunk_size

    try:
        output_ids = cast(torch.Tensor, model.generate(**inputs, **generate_kwargs))  # type: ignore[reportAttributeAccessIssue]
    except ValueError as exc:
        if "tokenizer_chunk_size" in str(exc):
            logger.warning(
                "tokenizer_chunk_size not supported by this transformers version; "
                "using model default."
            )
            generate_kwargs.pop("tokenizer_chunk_size", None)
            output_ids = cast(torch.Tensor, model.generate(**inputs, **generate_kwargs))  # type: ignore[reportAttributeAccessIssue]
        else:
            raise

    input_len: int = inputs["input_ids"].shape[1]  # type: ignore[reportUnknownMemberType]
    generated_ids: torch.Tensor = output_ids[:, input_len:]

    # Try parsed format first
    try:
        parsed: list[dict[str, Any]] = processor.decode(generated_ids, return_format="parsed")[0]  # type: ignore[reportUnknownMemberType]
    except Exception:
        parsed = []

    raw_text: str = processor.decode(generated_ids, return_format="transcription_only")[0]  # type: ignore[reportUnknownMemberType]

    del output_ids, inputs
    cleanup("after transcribe_chunk")

    return raw_text, parsed


def transcribe_audio(
    model: VibeVoiceAsrForConditionalGeneration,
    processor: VibeVoiceAsrProcessor,
    device: torch.device,
    dtype: torch.dtype,
    audio_path: Path,
    prompt: Optional[str],
    max_new_tokens: int,
    tokenizer_chunk_size: int,
    max_chunk_seconds: int,
    chunk_overlap_seconds: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Transcribe audio, chunking if necessary."""
    logger.info("STAGE: transcription")

    if max_chunk_seconds > 0:
        chunk_paths, tmpdir = split_audio_file(
            audio_path, max_seconds=max_chunk_seconds, overlap_seconds=chunk_overlap_seconds,
        )
    else:
        chunk_paths, tmpdir = [audio_path], None

    try:
        all_raw: List[str] = []
        all_segments: List[Dict[str, Any]] = []
        time_offset = 0.0

        for i, p in enumerate(chunk_paths, start=1):
            logger.info("Transcribing chunk %d/%d: %s", i, len(chunk_paths), p.name)
            raw, segments = transcribe_chunk(
                model=model,
                processor=processor,
                device=device,
                dtype=dtype,
                audio_path=p,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                tokenizer_chunk_size=tokenizer_chunk_size,
            )
            all_raw.append(raw)

            # Offset timestamps for multi-chunk
            if time_offset > 0 and segments:
                for seg in segments:
                    seg["Start"] = seg.get("Start", 0) + time_offset
                    seg["End"] = seg.get("End", 0) + time_offset
            all_segments.extend(segments)

            # Compute offset for next chunk
            if max_chunk_seconds > 0:
                chunk_dur = get_audio_duration(p)
                if chunk_dur:
                    time_offset += chunk_dur - chunk_overlap_seconds

        transcript = "\n".join(t for t in all_raw if t)
        logger.info("Transcription complete (chars=%d, segments=%d)", len(transcript), len(all_segments))
        return transcript, all_segments
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_diarized_transcript(segments: List[Dict[str, Any]]) -> str:
    """Format parsed segments as [Speaker N]: text lines."""
    lines: list[str] = []
    for seg in segments:
        speaker = seg.get("Speaker", "?")
        content = seg.get("Content", "")
        start = seg.get("Start", 0)
        end = seg.get("End", 0)
        lines.append(f"[{start:.1f}s-{end:.1f}s] [Speaker {speaker}]: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summarisation (text-only, using the same model's LLM backbone)
# ---------------------------------------------------------------------------

def load_summary_prompt(prompt_path: Optional[Path]) -> str:
    """Load a summary prompt from a file. Falls back to a built-in default."""
    if prompt_path is None:
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


@torch.inference_mode()
def generate_text_summary(
    model: VibeVoiceAsrForConditionalGeneration,
    processor: VibeVoiceAsrProcessor,
    device: torch.device,
    dtype: torch.dtype,
    prompt_text: str,
    max_new_tokens: int,
) -> str:
    """Generate text using the model's chat template (text-only, no audio)."""
    conversation: list[list[dict[str, str]]] = [
        [{"role": "user", "content": prompt_text}]
    ]
    inputs = cast(BatchFeature, processor.apply_chat_template(  # type: ignore[reportUnknownMemberType]
        conversation,
        tokenize=True,
        return_dict=True,
    )).to(device, dtype)  # type: ignore[reportUnknownMemberType]

    outputs: torch.Tensor = model.generate(  # type: ignore[reportAttributeAccessIssue]
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.2,
        do_sample=True,
    )

    input_len_summary: int = inputs["input_ids"].shape[1]  # type: ignore[reportUnknownMemberType]
    _decoded = processor.decode(outputs[:, input_len_summary:], return_format="transcription_only")  # type: ignore[reportUnknownMemberType]
    text: str = cast(str, _decoded[0]).strip()

    del outputs, inputs
    cleanup("after generate_text_summary")

    return text


def summarise(
    model: VibeVoiceAsrForConditionalGeneration,
    processor: VibeVoiceAsrProcessor,
    device: torch.device,
    dtype: torch.dtype,
    transcript: str,
    max_new_tokens: int,
    chunk_chars: int,
    chunk_overlap: int,
    summary_prompt: str,
) -> str:
    logger.info("STAGE: summarisation")
    chunks = chunk_text(transcript, chunk_chars, chunk_overlap)

    if not chunks:
        return "(No transcript text to summarise.)"

    if len(chunks) == 1:
        prompt = (
            f"{summary_prompt}\n\n"
            f"TRANSCRIPT:\n{chunks[0]}"
        )
        return generate_text_summary(model, processor, device, dtype, prompt, max_new_tokens)

    partials: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        logger.info("Summarising chunk %d/%d", i, len(chunks))
        prompt = (
            f"{summary_prompt}\n\n"
            f"Summarise part {i}/{len(chunks)} of this transcript "
            "based on the instructions above. "
            "Return concise bullet points focusing on facts, names, dates, and key events.\n\n"
            f"TRANSCRIPT PART:\n{chunk}"
        )
        partials.append(generate_text_summary(model, processor, device, dtype, prompt, max_new_tokens))

    combined = "\n\n".join(f"PART {i+1} SUMMARY:\n{p}" for i, p in enumerate(partials))
    final_prompt = (
        f"{summary_prompt}\n\n"
        "You are given summaries of transcript chunks. "
        "Produce a final consolidated summary based on the instructions above.\n\n"
        f"CHUNK SUMMARIES:\n{combined}"
    )
    return generate_text_summary(model, processor, device, dtype, final_prompt, max_new_tokens)


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------

def write_output_file(
    out_path: Path,
    audio_path: Path,
    model_id: str,
    transcript_diarized: str,
    transcript_raw: str,
    summary: Optional[str],
    has_segments: bool,
) -> None:
    """Write a structured Markdown report to *out_path*."""
    sep = "─" * 60
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections = [
        "# Transcription Report",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| **Audio file** | `{audio_path.name}` |",
        f"| **Model** | `{model_id}` |",
        f"| **Date** | {now} |",
        f"| **Diarization** | Built-in (VibeVoice) |",
        "",
        sep,
        "",
        "## Transcript (diarized)",
        "",
        transcript_diarized,
    ]

    if not has_segments and transcript_raw != transcript_diarized:
        sections += [
            "",
            sep,
            "",
            "## Transcript (raw)",
            "",
            transcript_raw,
        ]

    if summary:
        sections += [
            "",
            sep,
            "",
            "## Summary",
            "",
            summary,
        ]

    sections.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sections), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe + diarize + summarise with VibeVoice-ASR")
    ap.add_argument("audio_path", type=Path, help="Path to audio file (wav/mp3/m4a/...)")
    ap.add_argument("--model", default="microsoft/VibeVoice-ASR-HF", help="HF model id")
    ap.add_argument(
        "--prompt", default=None,
        help="Optional context/hotwords to guide transcription (e.g. proper nouns).",
    )

    # Generation
    ap.add_argument("--max-new-tokens", type=int, default=32768, help="Max tokens for transcription")
    ap.add_argument("--max-new-tokens-summary", type=int, default=512, help="Max tokens per summary chunk")

    # Audio chunking (VibeVoice handles up to 60 min natively)
    ap.add_argument(
        "--max-chunk-seconds", type=int, default=3600,
        help="Split audio into chunks of this many seconds. VibeVoice handles up to 60 min. Use 0 to disable.",
    )
    ap.add_argument("--chunk-overlap-seconds", type=int, default=0, help="Overlap between audio chunks (seconds).")

    # Tokenizer chunk size (for MPS OOM)
    ap.add_argument(
        "--tokenizer-chunk-size", type=int, default=1440000,
        help="Internal tokenizer chunk size (samples at 24kHz). Reduce if OOM. Default=1440000 (60s).",
    )

    # Summarisation
    ap.add_argument("--no-summary", action="store_true", help="Skip summarisation, only transcribe.")
    ap.add_argument("--chunk-chars", type=int, default=4000, help="Chars per summary text chunk")
    ap.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between summary text chunks")
    ap.add_argument(
        "--summary-prompt",
        type=Path,
        default=None,
        help="Path to summary prompt file. Defaults to assets/call_summarisation_prompt.md",
    )

    # Device
    ap.add_argument(
        "--device", default=None, choices=["mps", "cuda", "cpu"],
        help="Device override. Auto-detects if omitted.",
    )

    # Attention
    ap.add_argument(
        "--attn-implementation", default="auto", choices=["flash_attention_2", "sdpa", "eager", "auto"],
        help="Attention implementation. 'auto' picks best for device.",
    )

    # Output
    ap.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file path. Defaults to <audio_stem>_vibevoice_<timestamp>.md",
    )

    # Logging
    ap.add_argument("--log-level", default="INFO", help="Logging level")

    args = ap.parse_args()
    setup_logging(args.log_level)

    audio_path: Path = args.audio_path.expanduser().resolve()
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    # Device + dtype
    explicit_device = args.device is not None
    if args.device:
        device = torch.device(args.device)
    else:
        device = pick_device()
    dtype = pick_dtype(device)

    # Guard: VibeVoice-ASR is ~8B params → ~32GB float32 / ~16GB float16.
    # On machines with ≤16GB unified memory MPS cannot hold the model.
    if device.type == "mps":
        mem_gb = get_system_memory_gb()
        if 0 < mem_gb < 24:
            if explicit_device:
                logger.warning(
                    "%.0fGB unified memory — VibeVoice-ASR (~8B params) needs "
                    "~20GB+ on MPS. OOM is very likely. Consider --device cpu.",
                    mem_gb,
                )
            else:
                logger.warning(
                    "%.0fGB unified memory detected — not enough for MPS with "
                    "VibeVoice-ASR (~8B params). Falling back to CPU (float16). "
                    "Use --device mps to override.",
                    mem_gb,
                )
                device = torch.device("cpu")
                dtype = torch.float16

    # Attention implementation
    attn_impl = args.attn_implementation
    if attn_impl == "auto":
        if device.type == "cuda":
            try:
                import flash_attn  # type: ignore[reportMissingImports]  # noqa: F401
                attn_impl = "flash_attention_2"
            except ImportError:
                attn_impl = "sdpa"
        else:
            attn_impl = "sdpa"

    logger.info("Device=%s dtype=%s attn=%s", device, dtype, attn_impl)

    # Load model
    logger.info("STAGE: load model")
    processor = VibeVoiceAsrProcessor.from_pretrained(args.model)  # type: ignore[reportUnknownMemberType]

    # NOTE: attn_implementation is applied per-submodel because the acoustic
    # tokenizer encoder does NOT support sdpa/flash_attention_2.
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(  # type: ignore[reportUnknownMemberType]
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    # Apply attention implementation only to the language model backbone,
    # which is the part that actually benefits from it.
    if attn_impl != "eager" and hasattr(model, "language_model"):
        model.language_model._attn_implementation = attn_impl  # type: ignore[reportUnknownMemberType]
    model.to(device)  # type: ignore[reportUnknownMemberType]
    model.eval()
    logger.info("Model loaded on %s", device)

    # Transcribe
    transcript, segments = transcribe_audio(
        model=model,
        processor=processor,
        device=device,
        dtype=dtype,
        audio_path=audio_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        tokenizer_chunk_size=args.tokenizer_chunk_size,
        max_chunk_seconds=args.max_chunk_seconds,
        chunk_overlap_seconds=args.chunk_overlap_seconds,
    )

    # Format output
    if segments:
        diarized = format_diarized_transcript(segments)
    else:
        diarized = transcript

    # Summarise
    summary = None
    if not args.no_summary:
        summary_prompt_text = load_summary_prompt(args.summary_prompt)
        summary = summarise(
            model=model,
            processor=processor,
            device=device,
            dtype=dtype,
            transcript=transcript,
            max_new_tokens=args.max_new_tokens_summary,
            chunk_chars=args.chunk_chars,
            chunk_overlap=args.chunk_overlap,
            summary_prompt=summary_prompt_text,
        )

    # Write output file
    if args.output:
        out_path = args.output.expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{audio_path.stem}_vibevoice_{ts}.md"

    write_output_file(
        out_path=out_path,
        audio_path=audio_path,
        model_id=args.model,
        transcript_diarized=diarized,
        transcript_raw=transcript,
        summary=summary,
        has_segments=bool(segments),
    )
    logger.info("Output written to %s", out_path)


if __name__ == "__main__":
    main()
