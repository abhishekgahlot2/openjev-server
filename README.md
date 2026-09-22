# openjev-server

One-pass decisions from any open language model, served as an API.

[![ci](https://img.shields.io/github/actions/workflow/status/abhishekgahlot2/openjev-server/ci.yml?label=CI)](https://github.com/abhishekgahlot2/openjev-server/actions/workflows/ci.yml)
[![weights](https://img.shields.io/badge/weights-16bit%20%7C%20FP8%20%7C%20MLX-blue)](https://huggingface.co/openjev)
[![space](https://img.shields.io/badge/Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/openjev/openjev-server)
[![licence](https://img.shields.io/badge/licence-Apache%202.0-green)](LICENSE)

Send a situation and your questions. Get back one choice, a yes/no probability or a score per question, with a probability for every option. The answer is read off the model's first output position: one forward pass per question, nothing generated, no chain of thought to wait for.

It is the server behind [OpenJev](https://huggingface.co/openjev), an open 27B decision model for browser and desktop agents. Point it at OpenJev, or at your own model: anything served by vLLM, or an MLX model on a Mac.

## What you get

- **Three question types in one request.** `choice`, `noul` (yes/no) and `score`. Questions share the state but each is answered on its own.
- **Probabilities, not just labels.** Every option comes back with its probability and a confidence, so your code can branch on how sure the model is.
- **Any model.** A startup probe finds the label tokens, checks for exact score readout and vision, and refuses to serve a model that would return noise.
- **Honest probabilities for your model.** `openjev calibrate` fits the two calibration constants on your own development rows.
- **Fast where it matters.** Exact candidate scores by token id on vLLM. On a Mac the shared part of a request is read once and reused for every question.
- **Drop-in for clients of the hosted API.** Request and response shapes follow Jev's `/v1/systemone`, so a client written for it can point at your server.
- **Built to run.** Health and readiness, Prometheus metrics, bearer auth, a bench command, Docker Compose, CI.

## Why OpenJev

Most small "decision models" are fast because they are narrow. **The point is that it has to be fast and it has to be a general classifier at the same time**: any situation, any set of labels, given at request time, one pass.

**Same 10,000 questions, same options, same order for every model** (34 public datasets, gold answers from the datasets):

| model | right of 10,000 | accuracy |
|---|---|---|
| Jev, hosted | 8,540 | **85.4%** |
| **OpenJev**, open weights, one GPU or a Mac | **8,403** | **84.0%** |
| Qwen3.8-27B base, untuned | 8,036 | 80.4% |
| a fine-tuned 9B decision model | 7,574 | 75.7% |

**137 questions apart out of 10,000** from the hosted model, and **829 ahead of the 9B** that is sold as the fast alternative. On 2,000 desktop screenshot next-action steps OpenJev picks the right element **88.0%** of the time. A text decision takes about **125 ms** on one H100 and about **150 ms** on a Mac with the 4-bit build.

The comparison is not a controlled one: the hosted model's training data is unknown. The full method, the caveats and every number are on the [model card](https://huggingface.co/openjev/openjev#results).

## Demos

Real runs, every decision logged. The scripts ran on a MacBook Pro with the 4-bit build; five recordings were made on an H100 with the FP8 build. Scripts, outputs and provenance are in [`demos/`](demos/README.md).

| demo | what happens | result |
|---|---|---|
| [Chess](demos/README.md#chess) | legal moves are the options, one call per ply | checkmated a random mover in 16 moves, 1.2 s a move |
| [Poker](demos/README.md#poker) | heads-up no-limit, duplicate deals against an equity bot | 118 chips up on a random bot over 50 hands; loses slowly to an exact-equity bot; its strength rating tracks true equity (Spearman 0.67 to 0.75) |
| [Returns policy](demos/README.md#returns-policy-enterprise) | nine rules, 200 generated cases, exact ground truth | **93.5%** right, **88.0%** when it must count calendar days; majority baseline 43.5% |
| [Agent guardrail](demos/README.md#agent-guardrail-enterprise) | seven tool rules, 200 generated calls, exact ground truth | **200 of 200** right; 0 denied calls executed |
| [Row filters](demos/README.md#natural-language-row-filters-enterprise) | eight English filters per ticket row in one call | **92.1%** over 480 answers, 5 of 8 filters perfect |
| [Prompt injection](demos/README.md#prompt-injection-public-dataset) | 662 public messages, one yes/no each | **95.0%** accuracy, ROC-AUC 0.995, 97.7% recall at 2% false alarms |
| [Intent classification](demos/README.md#intent-classification-public-dataset) | Banking77, all 77 intents as options, two passes | **78.0%** zero shot, top-3 92.3%, 300 messages |
| [Generative UI](demos/README.md#generative-ui) | one sentence, ten layout decisions in one call, shadcn-style render | six distinct, sensible dashboards |
| [Browser agent](demos/README.md#browser-agent) | jev-ultrafast on the local server, page text and controls only | on the GPU build: Google Flights and a shopping task completed, recorded; on the laptop: Hacker News and Wikipedia done, two heavy pages not |
| [Inbox triage](demos/README.md#recorded-on-the-gpu-build) | 24 messages get a team, an urgency score and a review flag | 72 typed fields in 6.41 s on one H100, recorded |
| [Snake](demos/README.md#snake) | live play, one call per move | survived all three 200-move games, scores 14, 2, 17; a random player scores 0.3 |

<p><img src="demos/out/recorded_gpu/google-flights-colocated.gif" width="420" alt="browser agent on Google Flights"> <img src="demos/out/chess_openjev_vs_random/chess.gif" width="260" alt="chess"></p>
<p><img src="demos/out/recorded_gpu/inbox-classification.gif" width="420" alt="inbox triage"> <img src="demos/out/genui/saas_growth.png" width="420" alt="dashboard laid out by the model"></p>

## Models

| build | runs on | link |
|---|---|---|
| OpenJev 16 bit | two 80 GB GPUs, or one with offload | [openjev/openjev](https://huggingface.co/openjev/openjev) |
| OpenJev FP8 | one H100 or any 80 GB GPU | [openjev/openjev-FP8](https://huggingface.co/openjev/openjev-FP8) |
| OpenJev MLX 8 bit | Apple silicon, 64 GB | [openjev/openjev-MLX](https://huggingface.co/openjev/openjev-MLX) |
| OpenJev MLX 4 bit | Apple silicon, 32 GB | [openjev/openjev-MLX-4bit](https://huggingface.co/openjev/openjev-MLX-4bit) |

Accuracy, calibration and latency numbers are on the model cards and the [org page](https://huggingface.co/openjev).

## Install

```bash
pip install "openjev-server @ git+https://github.com/abhishekgahlot2/openjev-server"
```

## Quick start

**One GPU, vLLM**

```bash
vllm serve openjev/openjev-FP8 --served-model-name qwen --port 8000 --enable-prefix-caching \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image":1}' \
  --trust-remote-code --max-num-seqs 64 --max-logprobs 64 --gdn-prefill-backend triton --quantization fp8

openjev serve --backend vllm --model openjev/openjev-FP8 --vllm-url http://localhost:8000/v1 --profile openjev --port 3000
```

**Mac, MLX**

```bash
pip install "openjev-server[mlx] @ git+https://github.com/abhishekgahlot2/openjev-server"
hf download openjev/openjev-MLX-4bit --local-dir openjev-MLX-4bit
openjev serve --backend mlx --model openjev-MLX-4bit --profile openjev --port 3000
```

The MLX build takes text, JSON and DOM states. The shared part of a request is read once and reused for every question of that request.

## Ask

```bash
curl -s http://localhost:3000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Customer message: I was charged twice for my order last week and nobody has replied.",
  "questions": {
    "route":   {"type": "choice", "instructions": "Which team should handle this?",
                "criteria": {"billing": null, "shipping": null, "technical": null}},
    "angry":   {"type": "noul",   "instructions": "Is the customer angry?"},
    "urgency": {"type": "score",  "instructions": "How urgent is this?",
                "criteria": ["can wait", "this week", "today", "right now"]}
  }
}'
```

```json
{
  "answers": {
    "route":   {"type": "choice", "choice": "billing",
                "probabilities": {"billing": 0.9996, "shipping": 0.0002, "technical": 0.0002}, "confidence": 0.9995},
    "angry":   {"type": "noul", "noul": 0.56},
    "urgency": {"type": "score", "score": 2.04,
                "legend": {"0": "can wait", "1": "this week", "2": "today", "3": "right now"},
                "probabilities": {"0": 0.02, "1": 0.13, "2": 0.64, "3": 0.21}, "confidence": 0.71}
  },
  "usage": {"input_tokens": 239, "output_tokens": 0}
}
```

The message is about a double charge, and the routing probability says so with almost nothing left for the other teams. The urgency answer is split between "today" and "right now": that spread is the signal you get from probabilities that a single label would hide.

**Python**

```python
import httpx

r = httpx.post("http://localhost:3000/v1/systemone", json={
    "state": "Customer message: I was charged twice for my order last week and nobody has replied.",
    "questions": {
        "route": {"type": "choice", "instructions": "Which team should handle this?",
                  "criteria": {"billing": None, "shipping": None, "technical": None}},
        "angry": {"type": "noul", "instructions": "Is the customer angry?"},
    },
}, timeout=60)
answers = r.json()["answers"]
print(answers["route"]["choice"], answers["route"]["probabilities"], answers["angry"]["noul"])
```

| type | you send | you get |
|---|---|---|
| `choice` | `criteria`: map of option to description or `null` | the chosen option, a probability per option, confidence |
| `noul` | optional `criteria.true` / `criteria.false` descriptions | probability that the statement is true |
| `score` | `criteria`: ordered list of levels | expected level, probability per level, confidence |

- `instructions` may be a string or an object, for example a goal plus rules.
- Up to 52 options answer in one pass. More are handled in two passes.
- A screenshot goes in `state.screenshot` as a data URL, on models with a vision tower.
- Request and response shapes follow the hosted Jev API. A client written for it can be pointed at your own server.

## Any model

At start the server probes the model and prints what it found:

```json
{"model_probe": {"served_model": "qwen", "exact_readout": true, "chat_template": true,
                 "letter_prefix": "", "vision": true, "letter_mass_at_readout_position": 0.9994}}
```

- **Labels.** Options are labelled with letters. The server uses whichever single-token form the tokenizer has, `A` or ` A`.
- **Exact scores.** With vLLM it asks for the scores of exactly the candidate tokens and reads them back by id. Servers without that extension fall back to matching the top-K by text.
- **Templates.** Models without a chat template are prompted raw. Thinking is turned off where the template supports it. For models that always reason first, `--assistant-prefix` puts an empty think block before the readout position.
- **Refuses noise.** If the model puts almost none of its probability on the option letters at the readout position, the server refuses to start and says why. `--force` overrides.
- **Vision.** Detected from the model config. Screenshot requests to a text-only model return a clear error.

## Calibrate a new model

Probabilities are only useful if they are honest. Start a new model in the `uncalibrated` profile, then fit its two constants on your own development rows, never on a test set:

```bash
openjev serve --backend vllm --model your/model --profile uncalibrated --port 3000
openjev calibrate --dev dev.jsonl --endpoint http://localhost:3000 --out profiles/your-model.json
openjev serve --backend vllm --model your/model --profile profiles/your-model.json
```

`dev.jsonl` rows look like `{"state": ..., "questions": {"q": {...}}, "gold": "<option key>" | true | false}`. The `openjev` profile holds the constants behind every number on the OpenJev model cards.

## Operate

| what | how |
|---|---|
| liveness, readiness, metrics | `GET /healthz`, `GET /readyz`, `GET /metrics` (Prometheus) |
| what is running | `GET /v1/version`: model, profile, probe, hash of the readout code |
| auth | `--token` requires `Authorization: Bearer` on `/v1/*` |
| long states | `POST /v1/prewarm` prefills a state before its questions arrive |
| accuracy mode | `--perms 4` averages each answer over four option orders, at higher latency |
| benchmark | `openjev bench --endpoint http://localhost:3000 --tokens 1400 --conc 32` |
| settings | CLI flags, then `OPENJEV_*` variables, then the profile file. The original helper's `READOUT_*` names still work |
| containers | `Dockerfile` and `docker-compose.yml` for one GPU |

## Same answers as the published helper

With the `openjev` profile this server produces the same answers as the helper shipped in the model repository (`helper/shim.py`). The prompt layouts, calibration and confidence formulas are the same code paths, held by `tests/test_compat.py` and checked on the real model.

## Layout

| module | job |
|---|---|
| `openjev_server/profile.py` | the constants that turn scores into answers |
| `openjev_server/prompt.py` | state text and prompt layouts, byte-identical to the released helper |
| `openjev_server/readout.py` | scores to calibrated probabilities to answers; two-stage fallback above 52 options |
| `openjev_server/backends/base.py` | the contract: log-probabilities of given token ids at the first output position |
| `openjev_server/backends/vllm.py`, `mlx.py` | the two backends; `make_backend` builds one from settings |
| `openjev_server/backends/letters.py`, `probe.py` | label tokens for any tokenizer; the startup check |
| `openjev_server/server.py` | the HTTP surface |
| `calibrate.py`, `bench.py`, `cli.py` | the commands |

## Development

```bash
pip install -e ".[dev]"
ruff check openjev_server tests && ruff format --check openjev_server tests
pytest -q tests
```

The tests fetch the public OpenJev tokenizer and the released helper from the Hub. `python scripts/publish_space.py` mirrors the repository to the Hugging Face Space (needs `HF_TOKEN` in the environment).

## Licence

Apache 2.0. Model weights have their own licences (OpenJev: CC BY-NC 4.0).

OpenJev is an independent project, not affiliated with TypeSafe; Jev is their product.
