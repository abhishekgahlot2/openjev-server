# Changelog

## 0.2.0 (2026-09-22)
- Modules by job: `profile.py` (calibration constants), `prompt.py` (state text and prompt layouts), `readout.py` (scores to answers), `backends/base.py` (the backend contract), `backends/make_backend` (the only place that knows backend names). Answers unchanged: the compat tests against the released helper still hold.
- Server: `/v1` routes on one router with bearer auth as a router dependency; operations endpoints on their own router.
- vLLM backend: `VllmOptions` replaces ten constructor arguments; one response parser for the chat and raw-completion paths; tests against a fake OpenAI-compatible server.
- `/v1/version.readout_sha256` now hashes `prompt.py` + `readout.py` (the answer-determining code).
- Tests fetch the public OpenJev tokenizer and the released helper from the Hub when no local path is set, so CI needs nothing local.
- Landing page: links open in the top window (the Space iframe could not frame huggingface.co).

## 0.1.0 (2026-09-22)
- First release: the OpenJev decision API as an installable server. vLLM and MLX backends, profiles, calibration fitter, bench, metrics, bearer auth.
- Answers identical to the helper shipped with the OpenJev model (`helper/shim.py`) under the `openjev` profile.
- Any-model support: label tokenization detection, exact-readout detection, raw prompting without a chat template, assistant prefix for reasoning models, startup probe that refuses to serve noise.
- MLX: the shared part of a request is read once and reused across its questions.
