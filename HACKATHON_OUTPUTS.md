# Hackathon Outputs — Call Summarisation

| # | Output | Summary |
|---|--------|---------|
| 1 | Summarisation prompt & output format | Domain context, output schema (JSON/markdown), gold-standard examples, and a prompt iteration log |
| 2 | Diarisation comparison | Stereo channel split vs. pyannote vs. VibeVoice — head-to-head on the same audio with scored results |
| 3 | Model comparison matrix | Transcription and summarisation models evaluated separately on accuracy, speed, size, and licence |
| 4 | Synthetic data playbook | Audio format spec, TTS approach, public data sources, and a repeatable process to generate test calls |
| 5 | Infrastructure decision record | Latency target, inference server comparison, hardware sizing, cost estimate, and a reasoned recommendation |
| 6 | PII & data handling | PII categories in transcripts, redaction approach, data residency constraints, and retention policy |
| 7 | Working POC script | Single CLI entry point producing structured output, with measured performance on test files |
| 8 | Working end-to-end demo | Kafka consumer, processing pipeline, output storage, failure handling, and observability — the MVP finish line |

---

## 1. Summarisation prompt & output format

The prompt is the single highest-leverage thing to get right. A weak prompt makes every model look bad; a strong prompt can compensate for a weaker model.

What this output must include:

- **Sky domain context block**: A preamble the LLM receives on every call. It should define Sky-specific terminology (e.g. "Sky Glass", "Smart Home", "retention offer", "VIP & Rewards", "SkyID"), common call types (billing query, hardware fault, broadband outage, cancellation, upgrade), and the agent's role. Think of it as an AGENTS.md for the summarisation model — without it, the model will hallucinate generic telco language or miss domain nuance.
- **Output schema**: Decide JSON vs. markdown vs. plain text. JSON is better if downstream systems consume the summary programmatically (CRM, dashboards). Markdown is better if humans read it directly (e.g.: for quality checks by human call centre agents). Define the exact fields: executive summary, key topics, actions taken, follow-ups, customer sentiment, call outcome (resolved / escalated / callback).
- **Summary template with examples**: At least 3 example summaries written by hand from real or synthetic transcripts. These serve as few-shot examples in the prompt and as the ground truth for evaluating model output. Without these, you have no way to say whether a summary is "good enough."
- **Prompt iteration log**: Track prompt versions and what changed. Even a simple table: version, change made, observed effect. This is how you avoid circling back to prompts you already tried.

Deliverable: a versioned prompt file (e.g. `prompts/v1.md`) plus 3 gold-standard example outputs.

---

## 2. Diarisation comparison

Diarisation is the weakest link in your current POC. The `voxtral+pyannote.txt` output has agent and customer lines both attributed to `SPEAKER_01` — the summary is useless if you can't tell who said what.

What this output must include:

- **Stereo channel separation vs. model-based diarisation**: Many call centre recordings are stereo with agent on one channel and customer on the other. If that's the case for Sky's recordings, diarisation is a solved problem (just split channels) and you can skip pyannote entirely. You need to confirm the recording format with the team — this one fact may eliminate an entire class of complexity.
- **Head-to-head on the same audio**: Run the test file through (a) pyannote 3.1 + Voxtral transcription, (b) VibeVoice's built-in diarisation, and (c) stereo channel split if applicable. Compare outputs side by side. Score each on: correct speaker attribution rate, number of false speaker switches, handling of overlapping speech.
- **Edge cases to test**: Hold music / IVR, silence gaps, cross-talk, agent putting customer on hold, transferred calls (3+ speakers).
- **Configuration notes**: pyannote needs `num_speakers` hint for best results — document whether auto-detection is reliable enough or if the pipeline should always pass `num_speakers=2`.

Deliverable: a comparison table with annotated transcript excerpts showing where each approach succeeds or fails.

---

## 3. Model comparison matrix

You're comparing transcription and summarisation together, but they're separable. A model might be great at ASR but mediocre at summarisation, or vice versa. Split the comparison.

What this output must include:

**Transcription model comparison:**

| Criterion | Why it matters |
|-----------|---------------|
| License | Can Sky deploy it commercially? GPL/AGPL models are risky. |
| Model size / variants | Smaller models = faster inference, lower cost. Is there a 1B/3B/7B ladder? |
| Word Error Rate (WER) on test audio | The core accuracy metric. Measure on your own audio, not published benchmarks. |
| Supported languages | Does it handle Welsh, Scottish accents, code-switching? |
| Streaming support | Can it transcribe in real-time, or only post-call? |
| CPU/Memory footprint | Measured on target hardware, not claimed. Your POC already runs on MPS — measure there and on a comparable GPU. |
| Time to transcribe per minute of audio | Measure on 1-min, 5-min, and 15-min clips. Report real-time factor (e.g. 0.3x = 18s to transcribe 1 min). |

**Summarisation model comparison:**

| Criterion | Why it matters |
|-----------|---------------|
| Factual accuracy | Does it invent details not in the transcript? |
| Completeness | Does it miss key actions, amounts, dates? |
| Hallucination rate | Does it fabricate customer names, amounts, or commitments? |
| Action item extraction | Can it reliably pull out next steps? |
| Time to summarise | Per call, at what call length does it become too slow? |

**Candidates to test (corrected from your original list):**

- Voxtral Mini 3B (transcription + summarisation, your current default)
- Voxtral + pyannote (transcription pipeline with external diarisation)
- VibeVoice-ASR 8B (transcription with built-in diarisation)
- Whisper large-v3 (strong ASR baseline, no summarisation)
- A hosted API as control (e.g. Mistral API or OpenAI Whisper API) to benchmark against

Note: "VibeCode" in the original list appears to be a typo for VibeVoice. Clarify before the hackathon.

Deliverable: a filled-in matrix with measured numbers, not estimates. Run each candidate on the same 3-5 test audio files.

---

## 4. Synthetic data playbook

You need test data but probably can't use real customer calls on hackathon day. This output defines how to create realistic substitutes.

What this output must include:

- **Audio format spec**: Confirm with the team what format real call recordings arrive in. Sample rate (8kHz telephony? 16kHz? 48kHz?), bit depth, codec (PCM, G.711 μ-law, Opus, MP3), mono vs. stereo, file container (WAV, FLAC, MP3, raw). Your POC resamples everything to 16kHz mono — document whether that loses information and whether stereo is needed for channel-based diarisation.
- **Stereo channel mapping**: If recordings are stereo, which channel is agent and which is customer? Is this consistent? Does the telephony system document this? This directly affects whether you can skip model-based diarisation.
- **TTS generation approach**: If you're synthesising audio from transcripts, document which TTS engine (e.g. Bark, XTTS, Coqui, ElevenLabs), how to get two distinct voices, how to simulate realistic call quality (compression artefacts, background noise, hold music). Be honest about whether synthetic audio is realistic enough to validate the pipeline — if it isn't, flag this as a risk.
- **Alternative: semi-real data**: Can you use publicly available call centre recordings (e.g. from the CallHome corpus, or YouTube customer service calls)? These are more realistic than TTS and avoid PII issues. List sources.
- **Volume needed**: How many test calls do you need? For a hackathon POC, 5-10 is enough. For model comparison, you need at least 3 per "call type" (billing, fault, cancellation).

Deliverable: a repeatable process to generate N test audio files with known transcripts (so you can measure WER).

---

## 5. Infrastructure decision record

Where the compute runs, what hardware it needs, and what latency is acceptable.

What this output must include:

- **Latency target**: Is this real-time (during the call), near-real-time (seconds after hang-up), or batch (nightly)? This is the single biggest architectural driver. Real-time requires streaming ASR and GPU always-on. Batch can use spot instances and CPU. Get this answered before comparing anything.
- **Inference server comparison**: Compare the serving layer options on equal footing:
  - **HuggingFace Transformers (in-process)**: What you have now. Simple, no infra, but no batching, no scaling, and blocks the process during inference.
  - **vLLM**: Continuous batching, PagedAttention, OpenAI-compatible API. Best throughput for concurrent requests. Requires GPU.
  - **ollama**: Easy local setup, good for dev/testing. Not designed for production throughput.
  - **Triton / TensorRT-LLM**: NVIDIA's stack. Best raw performance but highest operational complexity.
  - **Hosted API (Mistral, OpenAI)**: Zero infra, pay-per-call. What are the data residency implications for Sky?
- **Hardware sizing**: For each model candidate, document: minimum GPU VRAM, recommended GPU, whether it fits on CPU-only (and how slow), whether quantisation (GPTQ, AWQ, GGUF) is viable and what accuracy it costs.
- **Cost estimate**: Even rough. "A single A100 on AWS costs ~$X/hr and can process ~Y calls/hr" gives stakeholders something to react to.
- **Recommendation with rationale**: Not just "use vLLM" but "use vLLM because we need to handle N concurrent calls with <Xs latency, which requires continuous batching, which rules out in-process Transformers."

Deliverable: a one-page decision record with the recommendation, alternatives considered, and key assumptions.

---

## 6. PII & data handling

The sample transcript contains a full name, address, postcode, password reminder, and partial bank details. This isn't optional — it determines what you can build and where you can run it.

What this output must include:

- **What PII appears in call transcripts**: List the categories: customer name, address, postcode, date of birth, account number, bank details, passwords/security answers, phone numbers. Your test transcript already shows most of these.
- **Redaction requirements**: Must PII be stripped from the summary? From the transcript? Before or after model inference? If before, you need a PII detection step in the pipeline (e.g. Presidio, spaCy NER, regex patterns for UK postcodes/phone numbers). If after, the model has already "seen" the PII — is that acceptable?
- **Data residency**: Can audio files or transcripts leave Sky's network? This rules out hosted APIs if the answer is no. Can they be sent to HuggingFace model endpoints? To a cloud GPU in eu-west-2?
- **Retention policy**: How long do audio files, transcripts, and summaries persist? Are they stored, or processed and discarded?
- **Impact on architecture**: If PII must never leave the network, you need on-prem or private-cloud GPU. If redaction is required pre-inference, add a pipeline stage. Document how this changes the architecture diagram.

Deliverable: a clear statement of constraints that the rest of the architecture must respect. Even if the answer is "we don't know yet," document the questions that need answering.

---

## 7. Working POC script

The current `main.py` and `vibevoice.py` both work end-to-end. The remaining 20% is making the output usable.

What this output must include:

- **Single entry point**: One script or CLI command that takes an audio file and produces a structured output file. Decide whether to consolidate `main.py` and `vibevoice.py` or keep them separate with a shared interface.
- **Correct diarisation**: The output must reliably distinguish agent from customer. If stereo channel separation works, use that instead of or alongside pyannote.
- **Structured output**: The summary file should match the output schema defined in Output 1. If JSON, it should be parseable. If markdown, it should be consistent.
- **Measured performance**: Run the script on 3-5 test files and record: total wall-clock time, peak memory usage, transcript accuracy (manual spot-check), summary quality (manual spot-check against the rubric from Output 1).
- **Known limitations**: Document what doesn't work: file formats that fail, call lengths that OOM, diarisation errors, summary hallucinations. This is more valuable than a polished demo.

Deliverable: `uv run summarise ./audio.wav` → structured output file, plus a short table of results on test files.

---

## 8. Working end-to-end demo -> Basis for MVP

This is the finish line for the POC: a running service that receives a call event, fetches the audio, transcribes, diarises, summarises, and stores the result — with no manual intervention. It's unlikely to be completed on hackathon day itself. Outputs 1–7 are the prep work; this output is what the follow-up days are for. Having it defined now means the team knows exactly what "done" looks like.

What this output must include:

- **Kafka consumer**: A service that subscribes to the call-recording topic, deserialises the message, and kicks off processing. Must handle consumer group rebalancing, offset commits (commit after successful processing, not before), and graceful shutdown. Document the topic name, consumer group ID, and deserialisation format.
- **Kafka message schema**: Define or confirm the schema for incoming messages. At minimum: call ID, audio file location (S3 path, URL, or file path), call metadata (agent ID, call start time, call duration, call type if known). If the schema already exists, document it. If it doesn't, propose one and get agreement before building.
- **Audio retrieval**: The consumer fetches the audio file from wherever recordings are stored (S3, NFS, a call recording platform API). Document: the storage system, auth mechanism (IAM role, API key, service account), expected file format (per Output 4), and what happens if the file isn't available yet (retry with backoff vs. re-queue).
- **Processing pipeline**: Wire up the POC script from Output 7 as the core processing step. The interface contract is `process_call(audio_path, metadata) -> Summary`. The pipeline should: load audio, run diarisation, transcribe, summarise using the prompt from Output 1, apply any PII handling from Output 6, and return a structured summary object matching the output schema.
- **Output storage**: Summaries need to land somewhere downstream systems can consume them. Options: write to a database (Postgres, DynamoDB), publish to another Kafka topic, call a CRM API, or write to an object store. Define the destination, schema, and write mechanism. If the destination isn't decided yet, implement two: file system (for local testing) and one real target (for demo).
- **Failure handling**: This is what separates a script from a service. Cover: audio file not found or corrupt (dead-letter topic + alert), model OOM or timeout (retry once with smaller chunk size, then dead-letter), transient infra errors (exponential backoff), poison messages (skip after N retries, log for manual review). Every failure path must be observable — log the call ID, failure reason, and stage where it failed.
- **Observability**: Structured logging with call ID as correlation key. Metrics: calls processed per minute, processing time per call (broken down by stage: fetch, diarise, transcribe, summarise, store), error rate by failure type, queue lag. Even if you don't ship a dashboard, instrument the code so metrics are emittable.
- **Architecture diagram**: A Mermaid or draw.io diagram showing the full flow: Kafka topic → consumer service → audio fetcher → diarisation → transcription → summarisation → PII redaction (if applicable) → output store. Label each box with the technology choice from Output 5 and mark which components exist (from the POC) vs. which are new.
- **Deployment notes**: How does this run? A Docker container? A Kubernetes job? A systemd service? Document the minimum viable deployment: Dockerfile, environment variables (Kafka brokers, model path, output destination, HF token), and health check endpoint.

Deliverable: a deployable service that processes call recordings from a Kafka topic end-to-end, with structured logging, failure handling, and a documented architecture diagram. Validated by processing at least 10 test calls without manual intervention.
