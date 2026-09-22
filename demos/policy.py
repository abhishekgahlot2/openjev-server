"""Enterprise policy decisions: a returns policy with nine rules, applied to generated cases with a known right answer.

    python demos/policy.py --endpoint http://localhost:3009 --cases 200 --out demos/out/policy

Every case is one /v1/systemone call with the policy text and the case in the state: a `choice` over the five outcomes and a `noul`
"needs manager approval". The right answers come from the rule engine below, so accuracy is exact and there is no dataset to leak.
Two versions of every case: dates given as calendar dates (the model must count the days itself) and days stated outright.
Baselines: majority class and uniform random. `--seed` fixes the cases.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from collections import Counter
from pathlib import Path

from common import Player

POLICY = """Returns policy of Northwind Electronics (applies to every order):
1. Unopened items returned within 30 days of delivery get a full refund to the original payment method.
2. Opened, non-defective items returned within 30 days of delivery get a partial refund: the price minus a 15% restocking fee.
3. Defective items are covered for 90 days after delivery in any condition: the customer gets a replacement if they ask for one, otherwise a full refund.
4. After the return window has passed, the request is denied, except that Premium members returning an unopened item between 31 and 60 days after delivery get store credit.
5. Without an order number the only possible outcome is store credit, and only for unopened items within 30 days of delivery; otherwise the request is denied.
6. Clearance items are final sale: denied unless defective (rule 3 still applies to defective clearance items).
7. Items received as a gift are never refunded to a card: where rules 1 or 2 would refund, the customer gets store credit of the same amount instead.
8. Any outcome other than a denial needs manager approval when the item price is above $1,000.
9. Where several rules apply, the earliest rule in this list that fits the case decides, except that rule 8 always applies on top."""

OUTCOMES = {
    "full_refund": "full refund to the original payment method",
    "partial_refund": "refund minus the 15% restocking fee",
    "replacement": "send a replacement item",
    "store_credit": "store credit for the amount",
    "deny": "deny the return",
}
ITEMS = [
    ("wireless headphones", 149),
    ("4K monitor", 489),
    ("mirrorless camera body", 1499),
    ("robot vacuum", 399),
    ("gaming laptop", 1899),
    ("smart thermostat", 179),
    ("mechanical keyboard", 129),
    ("home projector", 1099),
]


def truth(c: dict) -> tuple[str, bool]:
    """The policy applied by hand, in the order of the rules."""
    days, price = c["days"], c["price"]
    if c["defective"] and days <= 90:
        out = "replacement" if c["wants_replacement"] else "full_refund"
    elif not c["has_order_number"]:
        out = "store_credit" if (c["unopened"] and days <= 30) else "deny"
    elif c["clearance"]:
        out = "deny"
    elif c["unopened"] and days <= 30:
        out = "full_refund"
    elif not c["unopened"] and days <= 30:
        out = "partial_refund"
    elif c["premium"] and c["unopened"] and 31 <= days <= 60:
        out = "store_credit"
    else:
        out = "deny"
    if c["gift"] and out in ("full_refund", "partial_refund"):
        out = "store_credit"
    return out, out != "deny" and price > 1000


def make_case(rng: random.Random, i: int) -> dict:
    item, price = rng.choice(ITEMS)
    days = rng.choice([rng.randint(1, 30), rng.randint(1, 30), rng.randint(1, 30), rng.randint(31, 60), rng.randint(61, 90), rng.randint(91, 140)])
    today = dt.date(2026, 9, 22) - dt.timedelta(days=rng.randint(0, 60))
    return {
        "id": i,
        "item": item,
        "price": price,
        "days": days,
        "today": today,
        "delivered": today - dt.timedelta(days=days),
        "unopened": rng.random() < 0.45,
        "defective": rng.random() < 0.3,
        "wants_replacement": rng.random() < 0.5,
        "has_order_number": rng.random() < 0.9,
        "clearance": rng.random() < 0.1,
        "premium": rng.random() < 0.35,
        "gift": rng.random() < 0.2,
    }


def describe(c: dict, dates: bool) -> str:
    when = f"delivered on {c['delivered']:%A, %B %d, %Y}; today is {c['today']:%A, %B %d, %Y}" if dates else f"delivered {c['days']} days ago"
    bits = [
        f"Customer requests a return of a {c['item']} bought for ${c['price']:,}",
        when,
        "the item is unopened" if c["unopened"] else "the item has been opened and used",
        ("the customer reports it is defective and " + ("asks for a replacement" if c["wants_replacement"] else "wants their money back"))
        if c["defective"]
        else "the customer simply changed their mind",
        "the order number is on the request" if c["has_order_number"] else "the customer cannot find the order number",
        "it was a clearance item" if c["clearance"] else "it was a regular-price item",
        "the customer is a Premium member" if c["premium"] else "the customer has a standard account",
        "the item was a gift to the customer" if c["gift"] else "the customer bought it themselves",
    ]
    return ". ".join(bits) + "."


def run(player: Player, cases: list[dict], dates: bool, log) -> dict:
    right_out = right_appr = 0
    for c in cases:
        gold_out, gold_appr = truth(c)
        state = POLICY + "\n\nCase: " + describe(c, dates)
        answers, ms = player.ask(
            state,
            {
                "outcome": {"type": "choice", "instructions": "Apply the policy to the case. What is the outcome?", "criteria": OUTCOMES},
                "approval": {"type": "noul", "instructions": "Does this outcome need manager approval under the policy?"},
            },
        )
        out, appr = answers["outcome"]["choice"], answers["approval"]["noul"] >= 0.5
        right_out += out == gold_out
        right_appr += appr == gold_appr
        log.write(
            json.dumps(
                {
                    "id": c["id"],
                    "dates": dates,
                    "gold": gold_out,
                    "gold_approval": gold_appr,
                    "choice": out,
                    "approval_p": answers["approval"]["noul"],
                    "confidence": answers["outcome"]["confidence"],
                    "ms": round(ms),
                }
            )
            + "\n"
        )
    n = len(cases)
    return {"n": n, "outcome_accuracy": round(right_out / n, 4), "approval_accuracy": round(right_appr / n, 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="http://localhost:3000")
    ap.add_argument("--model", default="openjev")
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="demos/out/policy")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    cases = [make_case(rng, i) for i in range(a.cases)]
    golds = Counter(truth(c)[0] for c in cases)
    majority = golds.most_common(1)[0]
    player = Player(a.endpoint, a.model, name="model")
    result = {
        "cases": a.cases,
        "gold_distribution": dict(golds),
        "baselines": {"majority_class": {"label": majority[0], "accuracy": round(majority[1] / a.cases, 4)}, "uniform_random": round(1 / len(OUTCOMES), 4)},
    }
    with (out / "decisions.jsonl").open("w") as log:
        for dates in (False, True):
            r = run(player, cases, dates, log)
            result["calendar_dates" if dates else "days_stated"] = r
            print(("calendar dates" if dates else "days stated  ").ljust(15), json.dumps(r), flush=True)
    result["avg_ms"] = round(player.avg_ms())
    (out / "summary.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
