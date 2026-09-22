"""Intent classification on a public dataset with every label as an option. Banking77 has 77 intents, so this is also the
two-pass path: 77 options split into two chunks, then one pass over the chunk winners.

    python demos/classify.py --endpoint http://localhost:3009 --limit 300 --out demos/out/classify_banking77

Reports accuracy, top-3 accuracy from the returned probabilities, macro-F1, the majority baseline and throughput.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from common import Player


def load(limit: int, seed: int) -> tuple[list[dict], list[str]]:
    from datasets import load_dataset

    ds = load_dataset("mteb/banking77", split="test")  # parquet mirror of PolyAI/banking77 with label names
    names = sorted({r["label_text"] for r in ds})
    rows = [{"text": r["text"], "label": r["label_text"]} for r in ds]
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows, names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="http://localhost:3000")
    ap.add_argument("--model", default="openjev")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="demos/out/classify_banking77")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, names = load(a.limit, a.seed)
    criteria = {n: n.replace("_", " ") for n in names}
    question = {"intent": {"type": "choice", "instructions": "Which banking intent does this customer message express?", "criteria": criteria}}
    player = Player(a.endpoint, a.model, name="model")
    right = top3 = 0
    tp, fp, fn = Counter(), Counter(), Counter()
    with (out / "decisions.jsonl").open("w") as log:
        for i, r in enumerate(rows):
            answers, ms = player.ask(f"Customer message to the bank's support: {r['text']}", question)
            choice = answers["intent"]["choice"]
            ranked = sorted(answers["intent"]["probabilities"].items(), key=lambda kv: -kv[1])
            right += choice == r["label"]
            top3 += r["label"] in [k for k, _ in ranked[:3]]
            if choice == r["label"]:
                tp[choice] += 1
            else:
                fp[choice] += 1
                fn[r["label"]] += 1
            log.write(
                json.dumps(
                    {
                        "i": i,
                        "text": r["text"],
                        "gold": r["label"],
                        "choice": choice,
                        "top3": ranked[:3],
                        "confidence": answers["intent"]["confidence"],
                        "ms": round(ms),
                    }
                )
                + "\n"
            )
            if i % 50 == 0:
                print(f"{i}/{len(rows)} acc so far {right / (i + 1):.3f}", flush=True)
    f1s = []
    for n in names:
        p = tp[n] / (tp[n] + fp[n]) if tp[n] + fp[n] else 0.0
        r_ = tp[n] / (tp[n] + fn[n]) if tp[n] + fn[n] else 0.0
        if tp[n] + fn[n]:
            f1s.append(2 * p * r_ / (p + r_) if p + r_ else 0.0)
    gold_counts = Counter(r["label"] for r in rows)
    summary = {
        "dataset": "banking77 test (mteb/banking77 mirror)",
        "messages": len(rows),
        "labels": len(names),
        "accuracy": round(right / len(rows), 4),
        "top3_accuracy": round(top3 / len(rows), 4),
        "macro_f1": round(sum(f1s) / len(f1s), 4),
        "majority_baseline": round(gold_counts.most_common(1)[0][1] / len(rows), 4),
        "avg_ms": round(player.avg_ms()),
        "messages_per_second": round(1000 / player.avg_ms(), 2),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    confusions = defaultdict(int)
    for line in (out / "decisions.jsonl").open():
        d = json.loads(line)
        if d["choice"] != d["gold"]:
            confusions[(d["gold"], d["choice"])] += 1
    print("top confusions:", sorted(confusions.items(), key=lambda kv: -kv[1])[:5])


if __name__ == "__main__":
    main()
