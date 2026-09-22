"""Heads-up no-limit hold'em between two decision models, played as duplicate poker: every deal is played twice with the seats
swapped, so both players see the same cards and luck cancels out over the session.

    python demos/poker_game.py --a http://localhost:3009 --a-name OpenJev --b http://localhost:8009 --b-model kev-latest --b-name Kev-9B --deals 25

Each decision is one /v1/systemone call: a `choice` over the legal actions (with the amounts spelled out), a `score` for hand strength
and, when facing a bet, a `noul` "is the opponent bluffing?" (logged, not used). `rulebot` is a fixed-strategy baseline that bets its
hand strength. Stacks reset to 200 every hand (blinds 1/2); results are chips won over the session.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from common import Player
from treys import Card, Deck, Evaluator

EVAL = Evaluator()
STRENGTH = ["very weak", "weak", "medium", "strong", "very strong"]
STACK, SB, BB, MAX_RAISES = 200, 1, 2, 3


def cards(cs) -> str:
    return " ".join(Card.int_to_pretty_str(c).strip("[] ") for c in cs)


EQUITY_RNG = random.Random(7)


def equity(hole, board, samples=120, rng=EQUITY_RNG) -> float:
    """Monte-Carlo win probability against a random hand (the rulebot's only input; models never see it)."""
    wins = 0.0
    for _ in range(samples):
        deck = [c for c in Deck().cards if c not in hole and c not in board]
        rng.shuffle(deck)
        opp, rest = deck[:2], deck[2:]
        full = board + rest[: 5 - len(board)]
        a, b = EVAL.evaluate(full, hole), EVAL.evaluate(full, opp)
        wins += 1.0 if a < b else 0.5 if a == b else 0.0
    return wins / samples


class Hand:
    def __init__(self, deck: list[int], button: int, deal: int = 0):
        self.deck = deck
        self.deal = deal
        self.hole = [deck[0:2], deck[2:4]]
        self.board_all = deck[4:9]
        self.button = button  # posts the small blind, acts first preflop, second after
        self.stacks = [STACK, STACK]
        self.bets = [0, 0]
        self.pot = 0
        self.history: list[str] = []
        self.street = 0
        self.folded = None

    @property
    def board(self):
        return self.board_all[: [0, 3, 4, 5][self.street]]

    def street_name(self):
        return ["preflop", "flop", "turn", "river"][self.street]

    def options(self, p: int, raises: int) -> dict[str, str]:
        diff = self.bets[1 - p] - self.bets[p]
        pot_now = self.pot + sum(self.bets)
        opts = {}
        if diff > 0:
            opts["fold"] = "fold the hand"
            opts["call"] = f"call {min(diff, self.stacks[p])} (pot becomes {pot_now + min(diff, self.stacks[p])})"
        else:
            opts["check"] = "check, no chips"
        if raises < MAX_RAISES and self.stacks[p] > diff:
            for label, frac in (("half_pot", 0.5), ("pot", 1.0)):
                raise_by = max(BB, round((pot_now + diff) * frac))
                total = diff + raise_by
                if total < self.stacks[p]:
                    opts[f"raise_{label}"] = f"{'raise' if diff else 'bet'} {raise_by} more, to {self.bets[p] + total} ({int(frac * 100)}% of pot)"
            opts["all_in"] = f"all in for {self.stacks[p]} (to {self.bets[p] + self.stacks[p]})"
        return opts

    def apply(self, p: int, action: str):
        diff = self.bets[1 - p] - self.bets[p]
        pot_now = self.pot + sum(self.bets)
        if action == "fold":
            self.folded = p
            return
        if action in ("check", "call"):
            amt = min(diff, self.stacks[p])
        elif action == "all_in":
            amt = self.stacks[p]
        else:
            frac = 0.5 if action == "raise_half_pot" else 1.0
            amt = min(self.stacks[p], diff + max(BB, round((pot_now + diff) * frac)))
        self.stacks[p] -= amt
        self.bets[p] += amt

    def state_text(self, p: int) -> str:
        pos = "button (small blind)" if p == self.button else "big blind"
        return (
            "Heads-up no-limit Texas hold'em, one hand. You started with 200 chips, blinds 1/2.\n"
            f"Your position: {pos}. Street: {self.street_name()}.\n"
            f"Your hole cards: {cards(self.hole[p])}. Board: {cards(self.board) or '(none yet)'}.\n"
            f"Pot: {self.pot + sum(self.bets)}. Your stack: {self.stacks[p]}. Opponent stack: {self.stacks[1 - p]}. "
            f"To call: {self.bets[1 - p] - self.bets[p]}.\nAction so far: {'; '.join(self.history) or '(none)'}\n"
            "Choose your action."
        )


def rulebot(hand: Hand, p: int, opts: dict) -> str:
    e = equity(hand.hole[p], hand.board)
    if e > 0.72 and "raise_pot" in opts:
        return "raise_pot"
    if e > 0.58 and "raise_half_pot" in opts:
        return "raise_half_pot"
    if "check" in opts:
        return "check"
    diff = hand.bets[1 - p] - hand.bets[p]
    pot_odds = diff / (hand.pot + sum(hand.bets) + diff)
    return "call" if e > pot_odds + 0.05 else "fold"


RANDOM_RNG = random.Random(11)


def decide(player: Player | str, hand: Hand, p: int, opts: dict, log) -> str:
    if player == "rulebot":
        return rulebot(hand, p, opts)
    if player == "random":
        return RANDOM_RNG.choice(list(opts))
    questions = {
        "action": {"type": "choice", "instructions": "Which action do you take?", "criteria": opts},
        "strength": {"type": "score", "instructions": "How strong is your hand right now, against a typical opponent range?", "criteria": STRENGTH},
    }
    if "call" in opts:
        questions["bluff"] = {"type": "noul", "instructions": "Is the opponent probably bluffing with this bet?"}
    answers, ms = player.ask(hand.state_text(p), questions)
    action = answers["action"]["choice"]
    rec = {
        "deal": hand.deal,
        "player": player.name,
        "street": hand.street_name(),
        "hole": cards(hand.hole[p]),
        "board": cards(hand.board),
        "options": list(opts),
        "action": action,
        "confidence": answers["action"]["confidence"],
        "strength": answers["strength"]["score"],
        "bluff": answers.get("bluff", {}).get("noul"),
        "ms": round(ms),
    }
    log.write(json.dumps(rec) + "\n")
    return action


def seat(p: int) -> str:
    return "A" if p == 0 else "B"


def betting_round(players, hand: Hand, order: list[int], log) -> None:
    """One street: players act in `order` until the bets match, someone folds, or an all-in is matched."""
    acted, raises, i = 0, 0, 0
    while True:
        p = order[i % 2]
        if hand.stacks[p] == 0 and hand.bets[p] >= hand.bets[1 - p]:
            return  # all in and matched
        opts = hand.options(p, raises)
        action = decide(players[p], hand, p, opts, log)
        if action not in opts:
            action = "call" if "call" in opts else "check"
        hand.apply(p, action)
        hand.history.append(f"{seat(p)} {action.replace('_', ' ')}" + (f" (bet {hand.bets[p]})" if hand.bets[p] else ""))
        if hand.folded is not None:
            return
        if action.startswith("raise") or action == "all_in":
            raises, acted = raises + 1, 1
        else:
            acted += 1
        if hand.bets[0] == hand.bets[1] and (acted >= 2 or min(hand.stacks) == 0):
            return
        i += 1


def settle(hand: Hand) -> list[int]:
    """Chips won per seat after the last street (sum zero)."""
    hand.pot += sum(hand.bets)
    if hand.folded is not None:
        winner = 1 - hand.folded
    else:
        a, b = EVAL.evaluate(hand.board_all, hand.hole[0]), EVAL.evaluate(hand.board_all, hand.hole[1])
        winner = 0 if a < b else 1 if b < a else None
    won = [hand.pot // 2, hand.pot - hand.pot // 2] if winner is None else [hand.pot if i == winner else 0 for i in range(2)]
    return [hand.stacks[i] + won[i] - STACK for i in range(2)]


def play_hand(players, hand: Hand, log) -> list[int]:
    sb, bb = hand.button, 1 - hand.button
    hand.stacks[sb] -= SB
    hand.stacks[bb] -= BB
    hand.bets = [0, 0]
    hand.bets[sb], hand.bets[bb] = SB, BB
    hand.history = [f"{seat(sb)} posts small blind 1", f"{seat(bb)} posts big blind 2"]
    for street in range(4):
        hand.street = street
        if street > 0:
            hand.pot += sum(hand.bets)
            hand.bets = [0, 0]
            hand.history.append(f"--- {hand.street_name()}: {cards(hand.board)}")
        betting_round(players, hand, [sb, bb] if street == 0 else [bb, sb], log)
        if hand.folded is not None:
            break
    return settle(hand)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="endpoint, 'rulebot' (bets its equity) or 'random' (uniform over legal actions)")
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-model", default="openjev")
    ap.add_argument("--b-model", default="openjev")
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--deals", type=int, default=20, help="each deal is played twice with seats swapped")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="demos/out/poker")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pa = a.a if a.a in ("rulebot", "random") else Player(a.a, a.a_model, name=a.a_name)
    pb = a.b if a.b in ("rulebot", "random") else Player(a.b, a.b_model, name=a.b_name)
    rng = random.Random(a.seed)
    total = {a.a_name: 0, a.b_name: 0}
    hands = []
    with (out / "decisions.jsonl").open("w") as log:
        for deal in range(a.deals):
            deck = Deck().cards[:]
            rng.shuffle(deck)
            for swap in (0, 1):
                players = [pa, pb] if swap == 0 else [pb, pa]
                names = [a.a_name, a.b_name] if swap == 0 else [a.b_name, a.a_name]
                hand = Hand(list(deck), button=deal % 2, deal=deal)
                res = play_hand(players, hand, log)
                for i in range(2):
                    total[names[i]] += res[i]
                result = dict(zip(names, res, strict=True))
                hole = [cards(h) for h in hand.hole]
                hands.append({"deal": deal, "swap": swap, "board": cards(hand.board_all), "hole": hole, "result": result, "history": hand.history})
                print(f"deal {deal:2d}{'ab'[swap]}  {names[0]} {res[0]:+4d}  {names[1]} {res[1]:+4d}   running {total}", flush=True)
    summary = {"deals": a.deals, "hands": 2 * a.deals, "chips": total}
    for name, pl in ((a.a_name, pa), (a.b_name, pb)):
        if isinstance(pl, Player):
            summary[name] = {"decisions": pl.calls, "avg_ms": round(pl.avg_ms())}
    (out / "hands.json").write_text(json.dumps(hands, indent=1) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
