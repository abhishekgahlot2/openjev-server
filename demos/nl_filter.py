"""Natural-language filters over table rows, like a WHERE clause written in English. One call per row carries ALL the predicates
as yes/no questions, so 8 filters over a row cost one forward pass of the row plus one short readout each.

    python demos/nl_filter.py --endpoint http://localhost:3009 --rows 60 --out demos/out/nl_filter

Ground truth: each predicate is also written as Python over the row's fields. Reports precision / recall per predicate.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from pathlib import Path

from common import Player

TODAY = dt.date(2026, 9, 22)
ITEMS = [
    ("wireless headphones", "audio", 149),
    ("4K monitor", "displays", 489),
    ("mirrorless camera", "cameras", 1499),
    ("robot vacuum", "home", 399),
    ("gaming laptop", "computers", 1899),
    ("smart thermostat", "home", 179),
    ("mechanical keyboard", "accessories", 129),
    ("home projector", "displays", 1099),
    ("USB-C dock", "accessories", 89),
    ("noise-cancelling earbuds", "audio", 249),
]
ISSUES = [
    "arrived damaged",
    "never arrived",
    "wrong item sent",
    "stopped working after a week",
    "wants a different colour",
    "charged twice",
    "cannot connect to wifi",
    "missing accessory",
]
COUNTRIES = ["Germany", "France", "United States", "India", "Brazil", "Japan"]

PREDICATES = {
    "damaged_or_broken": ("The customer reports the item damaged or not working.", lambda r: r["issue"] in ("arrived damaged", "stopped working after a week")),
    "premium_over_500": ("Premium customer and the order is over $500.", lambda r: r["premium"] and r["price"] > 500),
    "delivery_problem": (
        "The problem is about delivery, not the product itself.",
        lambda r: r["issue"] in ("never arrived", "wrong item sent", "missing accessory"),
    ),
    "recent_30d": ("The ticket was opened in the last 30 days.", lambda r: (TODAY - r["opened"]).days <= 30),
    "eu_customer": ("The customer is in the European Union.", lambda r: r["country"] in ("Germany", "France")),
    "billing": ("It is a billing problem.", lambda r: r["issue"] == "charged twice"),
    "electronics_under_200": ("Audio or accessories item under $200.", lambda r: r["category"] in ("audio", "accessories") and r["price"] < 200),
    "needs_replacement": (
        "A replacement shipment is the likely fix.",
        lambda r: r["issue"] in ("arrived damaged", "wrong item sent", "missing accessory", "stopped working after a week"),
    ),
}


def make_row(rng: random.Random, i: int) -> dict:
    item, cat, price = rng.choice(ITEMS)
    return {
        "id": i,
        "item": item,
        "category": cat,
        "price": price,
        "issue": rng.choice(ISSUES),
        "country": rng.choice(COUNTRIES),
        "premium": rng.random() < 0.35,
        "opened": TODAY - dt.timedelta(days=rng.randint(0, 90)),
    }


def describe(r: dict) -> str:
    return (
        f"Support ticket #{r['id'] + 7000}. Opened {r['opened']:%B %d, %Y} (today is {TODAY:%B %d, %Y}). "
        f"Customer in {r['country']}, {'Premium' if r['premium'] else 'standard'} account. Item: {r['item']} ({r['category']}), ${r['price']:,}. "
        f"Customer says: the item {r['issue']}."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="http://localhost:3000")
    ap.add_argument("--model", default="openjev")
    ap.add_argument("--rows", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="demos/out/nl_filter")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    rows = [make_row(rng, i) for i in range(a.rows)]
    player = Player(a.endpoint, a.model, name="model")
    questions = {k: {"type": "noul", "instructions": text} for k, (text, _) in PREDICATES.items()}
    counts = {k: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for k in PREDICATES}
    with (out / "decisions.jsonl").open("w") as log:
        for r in rows:
            answers, ms = player.ask(describe(r), questions)
            rec = {"id": r["id"], "ms": round(ms), "predicates": {}}
            for k, (_, fn) in PREDICATES.items():
                gold, pred = fn(r), answers[k]["noul"] >= 0.5
                counts[k]["tp" if gold and pred else "fp" if pred else "fn" if gold else "tn"] += 1
                rec["predicates"][k] = {"gold": gold, "p": answers[k]["noul"]}
            log.write(json.dumps(rec) + "\n")
    report = {}
    for k, c in counts.items():
        prec = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 1.0
        rec_ = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 1.0
        report[k] = {"precision": round(prec, 3), "recall": round(rec_, 3), "accuracy": round((c["tp"] + c["tn"]) / a.rows, 3), "positives": c["tp"] + c["fn"]}
    total_right = sum(c["tp"] + c["tn"] for c in counts.values())
    summary = {
        "rows": a.rows,
        "predicates": len(PREDICATES),
        "row_predicate_accuracy": round(total_right / (a.rows * len(PREDICATES)), 4),
        "per_predicate": report,
        "avg_ms_per_row": round(player.avg_ms()),
        "predicates_per_second": round(len(PREDICATES) * 1000 / player.avg_ms(), 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
