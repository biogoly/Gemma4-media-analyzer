# Gemma-4 Media Analyzer

Gemma-4 Media Analyzer turns audio or video of arbitrary length into one chronological Markdown script and
one machine-readable JSON document. FFmpeg normalizes the source into resumable windows of no more than 30
seconds. Adjacent windows overlap by 2 seconds by default so words at an artificial boundary retain context;
their transcripts are reconciled deterministically before output. Audio inputs receive complete transcription
and optional translation. Video inputs receive that same audio analysis plus a grounded visual-description pass
that uses the current transcript and a compact continuity record from earlier chunks.

The target and preferred model is **Gemma 4 12B instruction-tuned**. The recommended deployment is Google's
QAT `Q4_0` GGUF through llama.cpp. It avoids loading PyTorch in the analyzer process and substantially reduces
model-weight VRAM. Server-side streaming keeps long completions responsive. The full-precision Transformers
backend remains available, and an experimental Ollama adapter is included.

## Model compatibility: use Gemma 4 12B

This project is built and tested around `google/gemma-4-12B-it`. The 12B Unified model is the preferred—and
currently only recommended—choice for the complete audio-and-video workflow.

| Gemma 4 variant | Native audio | Recommendation for this project |
| --- | --- | --- |
| **12B Unified** | Yes | **Preferred.** Best balance of transcription, visual analysis, structured-output quality, speed, and VRAM use. |
| E2B / E4B | Yes | Not recommended for final analysis. They are useful for lightweight experiments, but in project testing the 2B/4B-class models are not consistently accurate enough for detailed transcription, cross-chunk continuity, and scene description. |
| 26B A4B / 31B | No | Not compatible with the full pipeline. These variants accept text and images but do not process audio, including the audio pass required for videos with sound. |

Passing another model through `--model` does not add a missing modality. A 31B model may be useful in a
separate vision-only workflow, but it is not a drop-in quality upgrade for this analyzer. See Google's
[Gemma 4 model card](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf#models-overview) for the
official modality matrix.

## Why llama.cpp is the preferred quantized backend

- llama.cpp receives native `input_audio` blocks and ordered `image_url` frames on `/v1/chat/completions`.
- `-hf` retrieves both the 6.98 GB model GGUF and its multimodal projector from Google's official repository.
- The analyzer uses schema-constrained JSON, disables thinking, and accumulates streamed SSE responses.
- Base64 transport works across machines. `file-url` mode avoids base64 copies when client and server share a
  filesystem.

Quantization reduces weight memory and is often faster when inference is memory-bandwidth-bound, but speed is
hardware-dependent. KV cache, multimodal processing, and the projector still consume memory. Q4 can also lose
some accuracy, so validate transcription and fine OCR against representative media before production use.

## Prerequisites

- Python 3.11+
- FFmpeg and ffprobe on `PATH`
- A recent llama.cpp build for the supported GGUF path

Install the lightweight client (no Torch required):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev]"
```

## Recommended: Gemma 4 12B QAT Q4_0 with llama.cpp

Start the server. Gemma 4 supports several visual-token budgets; 140 is a practical quality/performance point
for this pipeline's 30 one-frame-per-second samples. The analyzer sends those exact samples as ordered images,
avoiding llama.cpp's higher internal sampling rate for native video inputs.

```powershell
llama-server `
  -hf google/gemma-4-12B-it-qat-q4_0-gguf:Q4_0 `
  --alias gemma4-media `
  --n-gpu-layers all `
  --ctx-size 16384 `
  --parallel 1 `
  --flash-attn on `
  --image-min-tokens 70 `
  --image-max-tokens 140 `
  --reasoning off
```

Analyze a video:

```powershell
gemma4-media analyze "C:\media\meeting.mp4" `
  --backend llama-cpp `
  --output-dir "C:\analysis-output" `
  --context "A quarterly engineering planning meeting" `
  --source-language English `
  --translate-to Spanish
```

`--context` is optional authoritative background for resolving visual ambiguity across chunks. Keep it factual
and concise. Without it, the analyzer still grounds each visual request with the current transcript and a
bounded record of opening/recent speech, the prior visual summary, final action, objects, and setting. Earlier
visual conclusions remain explicitly provisional so a mistaken classification is less likely to propagate.

Analyze an audio recording with the same pipeline. Audio-only sources skip visual preprocessing and inference:

```powershell
gemma4-media analyze "C:\media\interview.flac" `
  --backend llama-cpp `
  --output-dir "C:\analysis-output" `
  --translate-to English
```

The default server is `http://127.0.0.1:8080`. Use `--server-url` for another host. The default base64 media
transport is portable. For a same-machine server, local-file transport uses less host memory:

```powershell
llama-server `
  -hf google/gemma-4-12B-it-qat-q4_0-gguf:Q4_0 `
  --alias gemma4-media `
  --n-gpu-layers all `
  --ctx-size 16384 `
  --image-min-tokens 70 `
  --image-max-tokens 140 `
  --reasoning off `
  --media-path "C:\analysis-output"

gemma4-media analyze "C:\media\meeting.mp4" `
  --backend llama-cpp `
  --output-dir "C:\analysis-output" `
  --media-mode file-url `
  --media-root "C:\analysis-output"
```

For an authenticated server, put the token in an environment variable and pass only its name, for example
`--api-key-env GEMMA_SERVER_TOKEN`. The secret is never written to the cache manifest.

## Experimental: Ollama

Ollama's documented chat schema exposes an `images` collection rather than typed audio/video content. The
adapter forwards WAV and MP4 bytes through that collection because current multimodal builds can inspect such
media, but this route is version-dependent. Use llama.cpp when reliable audio and video ingestion matters.

```powershell
ollama pull gemma4:12b-it-qat
ollama serve

gemma4-media analyze "C:\media\meeting.mp4" `
  --backend ollama `
  --model gemma4:12b-it-qat `
  --output-dir "C:\analysis-output"
```

You can also import Google's repository with
`ollama run hf.co/google/gemma-4-12B-it-qat-q4_0-gguf:Q4_0`; if you do, pass that exact model name through
`--model`.

## Native Transformers backend

Install the heavier optional dependencies:

```powershell
pip install -e ".[transformers,dev]"
```

Then run:

```powershell
gemma4-media analyze "C:\media\meeting.mp4" `
  --backend transformers `
  --model google/gemma-4-12B-it `
  --output-dir "C:\analysis-output"
```

The Transformers backend loads `AutoModelForMultimodalLM` lazily, so server-only installations never import
Torch or download the full-precision checkpoint.

## Preprocessing and outputs

Defaults are deliberately conservative:

- chunks: 30 seconds (hard upper bound enforced)
- overlap: 2 seconds, producing a 28-second stride; configure with `--chunk-overlap`
- video inputs: 1 FPS, longest edge 720 px, H.264/YUV420p, no audio track
- llama.cpp visual transport: ordered JPEG frames at the configured FPS, not native `input_video`
- visual output: consecutive frames are grouped into at most 10 shots; repeated OCR text is deduplicated
- visual continuity: current audio plus a bounded deterministic record; no extra model request is required
- visual grounding: uncertain object identities and OCR are described neutrally or omitted rather than guessed
- malformed or truncated llama.cpp visual JSON: one automatic compact retry with a larger safe token allowance
- audio from either source type: mono, 16 kHz, float32 PCM WAV
- timestamps: duration-bounded affine normalization preserves model order and prevents chunk overruns
- transcript completeness: the full chunk transcript is canonical; missing timestamped tails or heads are
  restored with inferred timestamps, with a whole-chunk fallback for more complex segment disagreements
- inference: audio request for every audio chunk; an ordered-frame visual request only for video chunks
- generation: temperature 1.0, top-p 0.95, top-k 64, thinking disabled

At each overlap, the reconciler compares the earlier transcript suffix with the later transcript prefix. Exact
duplicates and clearly completed partial words or sentences use the later rendition. Fuzzy conflicts retain
the earlier wording because a chunk-opening word is especially vulnerable to clipping. If text alignment is
inconclusive, both renditions are preserved: Gemma's generated timestamps are not reliable enough to justify
deleting otherwise unmatched dialogue. This can leave a duplicate, but prevents silent omissions.

The complete chunk transcript is canonical. If Gemma omits a transcript tail or head from its timestamped
segment array, the analyzer restores it with an inferred timestamp. More complex disagreement falls back to
rendering the complete chunk transcript rather than silently dropping words. Normalized per-chunk analysis
records stay in the cache; reconciliation is applied to final outputs. At timestamp overflow, the model's
relative timeline is proportionally compressed into the real FFmpeg chunk duration. Retained dialogue is then
made monotonic across overlap seams. These timestamps remain approximate model estimates, but they cannot
extend beyond the source window or run backward between successive retained statements.

Segment trimming follows the canonical transcript span rather than deleting an assumed number of words from
the end of an incomplete segment array. Existing duration-bounded audio cache entries are repaired in place;
this segment-coverage upgrade does not require another inference pass.
Set `--chunk-overlap 0` to restore non-overlapping behavior.

Run preprocessing without a model:

```powershell
gemma4-media prepare "C:\media\meeting.mp4" --output-dir "C:\analysis-output"
gemma4-media prepare "C:\media\interview.mp3" --output-dir "C:\analysis-output"
```

Output files are `<media>.analysis.md` and `<media>.analysis.json`. The JSON includes a top-level
`media_type` value of `audio` or `video`, plus top-level `transcript` and `translation` fields containing the
reconciled full-media text. Intermediate chunks and normalized per-pass JSON are
stored under `.gemma4-cache`. Cache keys include the source fingerprint, preprocessing settings, backend,
model, language, generation settings, transcript evidence, rolling continuity, and user context. Contextual
visual cache entries are isolated from older independent-vision entries while compatible audio transcripts are
reused. A stopped run therefore resumes at the first unfinished audio or visual pass rather than starting over.

The former `gemma4-video` executable and `gemma4_video_analyzer` Python namespace remain available as
compatibility aliases. New integrations should use `gemma4-media` and `gemma4_media_analyzer`.

## Known limitations

- Gemma 4 12B is a general multimodal model, not a dedicated speech recognizer. Spoken dialogue can be very
  accurate, but names, specialized terminology, dense crosstalk, and especially sung lyrics may require manual
  correction or comparison with a dedicated STT system such as Whisper.
- Timestamps are normalized model estimates rather than forced audio alignments. They are chronological and
  bounded to the source duration, but should not be treated as frame-accurate subtitles.
- The compact continuity record reduces cross-chunk visual misclassification but cannot eliminate it. Review
  uncertain object identities, OCR, and subtle actions against the source video.
- Quantization can reduce transcription, OCR, and fine visual accuracy. Validate the selected GGUF and visual
  token budget on representative material before relying on unattended output.

## Tests

```powershell
pytest
ruff check .
```

The unit tests do not require a model, GPU, FFmpeg invocation, or running inference server.

## License

Gemma-4 Media Analyzer is released under the [MIT License](LICENSE). Model weights, llama.cpp, FFmpeg, and
other dependencies remain subject to their respective licenses and terms.

## Upstream references

- [Official Google Gemma 4 QAT Q4_0 GGUF model card](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf)
- [llama.cpp multimodal documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)
- [llama.cpp server API documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [Ollama Gemma 4 tags](https://ollama.com/library/gemma4/tags)
