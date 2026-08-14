# Changelog

All notable changes to **supertonic-py** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project broadly follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html);
**see the heads-up note below 1.2.2 — that release carries default-value
shifts in `total_steps` and `lang` that are semver-minor in nature but are
shipping under a patch bump on purpose.**

## [Unreleased]

### Added
- **Realtime streaming synthesis** (`supertonic.streaming`), for voice agents
  that need audio before the text is finished. `synthesize_stream()` — also
  exposed as `TTS.synthesize_stream()` — accepts a string *or* an iterable of
  fragments (an LLM's streamed tokens) and yields an `AudioChunk` per clause.
  `ClauseBuffer` does the incremental splitting: it releases text at sentence
  and phrase boundaries as soon as one is complete, holding back a period only
  until the next character confirms it is not `3.14`, `Dr.`, or an initial.
  The vocoder consumes a whole latent, so a clause — not an audio frame — is
  the smallest unit that can be emitted; the first clause therefore uses a
  shorter character target and fewer diffusion steps (`first_chunk_steps=4`),
  since it is the only chunk the listener waits on.
- **`WS /v1/realtime`** on `supertonic serve`: the client pushes
  `input.text.delta` events, the server answers with an `audio.chunk` metadata
  event followed by exactly one binary frame of raw PCM (`pcm_s16le` default,
  `pcm_f32le` available) at the model's native 44.1 kHz. Supports
  `session.update` (voice/lang/speed/steps/format/chunk sizing, also settable
  as query parameters), `speak`, `input.text.done`, `ping`, and `cancel` for
  barge-in. Cancellation drops queued clauses and discards the chunk currently
  in the ONNX session — inference cannot be interrupted, so barge-in takes
  effect within one chunk.
- `examples/test10_streaming.py` and `examples/test11_realtime_websocket.py`.

### Changed
- `supertonic.server.routes._resolve_voice` renamed to `resolve_voice`; it is
  now shared with the realtime endpoint, which resolves a voice once per
  WebSocket session instead of once per request.
- `supertonic serve` startup output lists the realtime WebSocket URL.

## [1.3.1] — 2026-05-18

### Added
- **Python 3.13 is officially supported.** The CI matrix runs against 3.9
  through 3.13 on Ubuntu/macOS/Windows; 3.14 enters the matrix as
  *experimental* (best-effort, `continue-on-error`, `allow-prereleases`)
  and will be promoted once all native dependencies (notably
  `onnxruntime`, `uvloop`) ship 3.14 wheels.

### Changed
- Raise the `onnxruntime` floor to `>=1.20.0`. 1.20.0 is the first release
  with a Python 3.13 wheel — older floors force a source build (and
  almost-certain failure) on 3.13 environments. This preserves the
  numpy 2.x C-ABI compatibility that motivated the 1.19.0 floor in 1.2.3.
- CI test job now installs `pip install -e ".[dev,serve]"`. Previously
  only `[dev]` was installed, so `tests/test_server_routes.py` was
  skipped via `pytest.importorskip("fastapi")` and the new server code
  shipped without a real CI gate. The full server test suite now runs
  across every matrix cell.
- `[tool.black].target-version` narrowed to `py39…py311` to match the
  lint job's Python (3.11). Including `py312` produced a noisy "Python
  3.11 cannot parse code formatted for Python 3.12" warning on every CI
  run without changing any actual formatting.

### Fixed
- Apply Black formatting to `supertonic/server/{routes,styles_store}.py`
  and `tests/test_server_routes.py` (whitespace + ternary parenthesization
  drift; no behavior change).
- Drop unused imports (`sys` in `tests/test_cli_serve.py`, `io` in
  `tests/test_server_routes.py`) flagged by Ruff.
- Remove the unused `styles_store.resolve()` helper and its trailing
  `Optional` import (dead code from the initial server cut).
- **`SUPERTONIC_CACHE_DIR` env var now overrides the cache directory on
  every code path.** Previously it was silently ignored on the most common
  call shape — `TTS()` constructed with a model name (the default, since
  `model: str = "supertonic-3"`). `get_model_cache_dir()` now consults the
  env var at call time, and `get_cache_dir()` no longer has a two-branch
  fork. The same fix transitively repairs `default_custom_styles_dir()`
  in the server package. Resolution is **lazy** (call-time) instead of
  eager (import-time), so late-set env vars are honored. Tests cover the
  reproduction from the original bug report plus a lazy-evaluation guard.
- **Restore Python 3.9 installability after the `onnxruntime>=1.20.0`
  bump.** onnxruntime 1.20.0 added a 3.13 wheel but dropped the 3.9 wheel,
  so a single floor couldn't satisfy both ends of the supported range.
  The dependency is now split with PEP 508 markers:
  `onnxruntime>=1.19.0; python_version<'3.13'` and
  `onnxruntime>=1.20.0; python_version>='3.13'`. 3.9-3.12 keep the 1.19.0
  floor (3.9 wheel available, numpy 2.x C-ABI safe); 3.13 gets the 1.20.0
  floor (3.13 wheel available).
- **CI lint job now installs through the `[dev]` extra** instead of a
  bare `pip install black ruff`, so contributor and CI formatting/linting
  versions cannot drift. The `[dev]` floors are raised to `black>=25.0`
  and `ruff>=0.5` accordingly.

## [1.3.0] — 2026-05-18

### Added
- **`supertonic serve` — local HTTP server.** A thin, loopback-only FastAPI
  wrapper around the same TTS engine, installed via the new `[serve]`
  extra (`pip install supertonic[serve]`). Designed for environments where
  embedding a Python interpreter is awkward — n8n, browser extensions,
  Electron apps, Unity, Home Assistant, robotics devices — and for clients
  that already speak the OpenAI Audio Speech API.

  Endpoints:
  - `GET  /v1/health` — readiness + loaded model info
  - `GET  /v1/styles` — built-in + imported custom voices
  - `POST /v1/styles/import` — multipart or JSON upload of a Voice Builder
    style JSON; persisted per-model under
    `~/.cache/<model>/custom_styles/`
  - `POST /v1/tts` — native synthesis (full parameter set)
  - `POST /v1/audio/speech` — OpenAI Audio Speech-compatible alias
  - `POST /v1/tts/batch` — up to 64 items in a single JSON request,
    base64-encoded audio response

  Defaults: bind `127.0.0.1:7788`, no auth, no CORS, single uvicorn worker.
  Binding beyond loopback is opt-in and emits a stderr warning. Errors use
  the OpenAI-shaped `{"error":{"message","type","code"}}` envelope so
  existing OpenAI-SDK error parsers keep working.

- **`supertonic.server` Python package.** `create_app()` and `ServerState`
  are exposed for users who want to mount the FastAPI app inside a larger
  ASGI service (e.g. behind a reverse proxy with auth).

- **Per-model custom-voice storage.** Imported voice JSONs are stored
  under `<model cache dir>/custom_styles/` (e.g.
  `~/.cache/supertonic3/custom_styles/`) so the same name cannot collide
  across model versions. `$SUPERTONIC_CUSTOM_STYLES_DIR` overrides this
  with a single shared directory.

- **`Content-Length` pre-flight middleware** for `POST /v1/styles/import`.
  Requests exceeding 1 MiB are rejected at the headers stage before the
  body is buffered.

- **Docs + CLI reference for `supertonic serve`** — `docs/cli/serve.md`,
  `docs/api/server.md`, a new *Local Server* section in `docs/quickstart.md`,
  and a *Local Server (HTTP)* section in `README.md`.

### Changed
- README *Key Features* section renamed and refreshed as *✨ Highlights*
  with Supertonic-3 facts (99M-parameter open-weight model, 31-language
  multilingual, 44.1 kHz native output, Expression Tags, multi-runtime
  SDKs). The same update lands in `docs/index.md`.

### Notes
- `supertonic serve` ships with no MP3, AAC, or Opus encoding. Supported
  `response_format` values are `wav`, `flac`, and `ogg` (Vorbis) —
  everything libsndfile encodes natively at 44.1 kHz. MP3/AAC would add
  encoder dependencies; Opus needs an 8/12/16/24/48 kHz resampling step
  and is deferred.
- The existing SDK and CLI (`say`, `tts`, `list-voices`, `info`,
  `download`, `version`) are untouched. Core dependencies are unchanged —
  FastAPI/uvicorn install only when the `[serve]` extra is requested.

## [1.2.3] — 2026-05-15

### Fixed
- Raise the `onnxruntime` lower bound to `>=1.19.0`. `onnxruntime` 1.18
  and earlier shipped wheels built against the numpy 1.x C ABI, so
  installing them alongside `numpy>=2.0` — which became possible in
  1.2.2 via #1 ("Drop numpy<2.0 upper bound") — produced `ImportError`
  or segfault at module import time. `onnxruntime` 1.19.0 (Aug 2024) is
  the first release with explicit numpy 2.x support
  ([release notes](https://github.com/microsoft/onnxruntime/releases/tag/v1.19.0):
  *"Numpy support for 2.x has been added"*), so this bump unblocks the
  numpy 2.x co-install scenario that motivated #1 (e.g. running side by
  side with `kokoro-onnx>=0.5`).
- Drop the residual `numpy<2.0` pin in `requirements.txt` that #1
  missed — only `pyproject.toml` was touched in that PR, leaving
  `requirements.txt` inconsistent with the package spec.

## [1.2.2] — Unreleased

### Heads-up — soft behavior changes in this patch

This release intentionally changes the **default value** of two public
parameters. Code copy-pasted from older docs or tutorials will keep
working, but the audio it produces will be subtly different.

- `TTS.synthesize(total_steps=...)` default `5` → `8`. Output quality
  goes up; synthesis is slightly slower. To keep the old behavior, pass
  `total_steps=5` explicitly.
- `TTS.synthesize(lang=...)` default no longer hard-coded to `"en"`.
  When unset, supertonic-2 / supertonic-3 now resolve to the language-
  agnostic `"na"` fallback (so non-English text "just works" without the
  user picking a code), and supertonic v1 still resolves to `"en"`. To
  keep the old behavior on v3, pass `lang="en"` explicitly.

### Added
- `MODEL_CONFIGS` entries now pin the Hugging Face Hub revision SHA per
  model. `pip install supertonic==1.2.2` will always download the same
  ONNX weights, even if the upstream HF repos are updated later. The
  pinned SHAs:
  - `Supertone/supertonic` → `b6856d033f622c63ea29441795be266a1133e227`
  - `Supertone/supertonic-2` → `75e6727618a02f323c720cba9478152d4bc16ca4`
  - `Supertone/supertonic-3` → `724fb5abbf5502583fb520898d45929e62f02c0b`
- `supertonic.config.get_model_revision(model_name)` helper; the
  `SUPERTONIC_MODEL_REVISION` env var continues to act as a manual
  override (primarily for development).
- `SECURITY.md` with a responsible-disclosure contact.
- `CHANGELOG.md` (this file).

### Changed
- **Default `total_steps`** raised from `5` to `8`
  (`supertonic/config.py:DEFAULT_TOTAL_STEPS`). Affects
  `TTS.synthesize`, `TTS.__call__`, and the `--steps` CLI option.
- **Default `lang`** is now model-aware. The `synthesize` /  `__call__`
  signatures changed from `lang: str = "en"` to
  `lang: Optional[str] = None`. When `None`, the value is resolved at
  synthesis time:
  - Multilingual models (supertonic-2 / supertonic-3) → `"na"` fallback
  - English-only supertonic v1 → `"en"`
- CLI `--lang` default also becomes `None`; `--help` now reads
  *"Default: 'na' for multilingual models (supertonic-2/3), 'en' for
  supertonic v1."*
- Model download size reference updated from `~305 MB` to `~400 MB`
  across README and docs.
- README and `docs/index.md` restructured: Python Quick Start moved
  above CLI, the snippet annotated inline so it doubles as
  copy-paste-to-an-LLM documentation, new "Custom voices (Voice
  Builder)" subsection covers `get_voice_style_from_path()` and the
  `~/.cache/supertonic3/voice_styles/{M1..M5,F1..F5}.json` layout.
- Documentation now consistently states there are **10 built-in voices**
  (M1–M5, F1–F5) — previously some surfaces said only four.
- `docs/quickstart.md`, `docs/api/index.md`, `docs/cli/README.md`,
  `supertonic/__init__.py` example, and the `get_voice_style()`
  docstring updated to match the new defaults.

### Fixed
- `cli.py` first-run progress line no longer prints `lang=None` when
  the language is unspecified; shows `lang=auto` instead.
- `examples/test2_voices.py` docstring now accurately reflects the
  full 10-voice set the script iterates over.

## [1.2.1] — Unreleased

`1.2.1` was bumped in source but never tagged. Its work has been folded
into `1.2.2`. Notable items from that interim:

- **feat**: Supertonic-3 multilingual support (31 ISO languages plus
  `"na"` fallback for unknown / unsupported text). Adds the new
  `Supertone/supertonic-3` HF Hub repo and corresponding entries in
  `MODEL_CONFIGS`. Commit `9d5373d`.
- **fix**: preserve diacritics in text preprocessing. Commit `d02b8b5`.
- **chore**: package authors updated. Commit `074eca5`.

## [1.2.0] — 2026-01-24

- Documentation refresh: multilingual section in quickstart, new banner
  image, README links pointing to raw GitHub for image assets, license
  format normalized in metadata. Commits `87ed50a`, `7f35d03`,
  `db57854`, `0f473a0`.

## [1.1.2] — 2026-01-24

- Banner URL fix and patch bump. Commit `1e3e202`. Banner image and
  URLs updated for supertonic-2. Commit `5a8eeae`.

## [1.1.1] — 2026-01-24

- First multilingual release (`supertonic-2`). Commit `ef1c39f`.

## [1.0.0] — 2025-11-24 → 2025-12-10

- Initial public release: English-only supertonic v1 on ONNX Runtime,
  HF Hub model download, CLI (`supertonic say` / `tts` / `list-voices`
  / `info`), 10 built-in voices.

[1.2.2]: https://github.com/supertone-inc/supertonic-py/compare/v1.2.0...v1.2.2
[1.2.1]: https://github.com/supertone-inc/supertonic-py/compare/v1.2.0...v1.2.2
[1.2.0]: https://github.com/supertone-inc/supertonic-py/compare/v1.1.2...v1.2.0
[1.1.2]: https://github.com/supertone-inc/supertonic-py/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/supertone-inc/supertonic-py/compare/v1.0.0...v1.1.1
[1.0.0]: https://github.com/supertone-inc/supertonic-py/releases/tag/v1.0.0
