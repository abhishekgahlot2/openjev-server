"""Does the model's poker play track the cards? For every logged decision: Monte-Carlo equity of the hand against a random hand
(hidden from the model) versus the model's own strength rating and its action.

    python demos/poker_analysis.py demos/out/poker_openjev_vs_rulebot
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from treys import Card

sys.path.insert(0, str(Path(__file__).parent))
from poker_game import equity

SUITS = {"♠": "s", "♥": "h", "♦": "d", "♣": "c"}


def parse(cards: str) -> list[int]:
    return [Card.new(c[:-1] + SUITS[c[-1]]) for c in cards.split()] if cards else []


def spearman(xs, ys) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else 0.0


def main(game_dir: str) -> None:
    rows = [json.loads(line) for line in (Path(game_dir) / "decisions.jsonl").open()]
    rows = [r for r in rows if r["player"] != "rulebot"]
    eq = [equity(parse(r["hole"]), parse(r["board"]), samples=300) for r in rows]
    strength = [r["strength"] for r in rows]
    buckets = {"weak (<40%)": [], "middle (40-60%)": [], "strong (>60%)": []}
    for r, e in zip(rows, eq, strict=True):
        key = "weak (<40%)" if e < 0.4 else "middle (40-60%)" if e < 0.6 else "strong (>60%)"
        buckets[key].append(r["action"])
    out = {
        "decisions": len(rows),
        "spearman_strength_vs_equity": round(spearman(strength, eq), 3),
        "action_by_equity": {
            k: {
                "n": len(v),
                "fold": round(sum(a == "fold" for a in v) / len(v), 2),
                "raise_or_all_in": round(sum(a.startswith("raise") or a == "all_in" for a in v) / len(v), 2),
            }
            for k, v in buckets.items()
            if v
        },
    }
    (Path(game_dir) / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
