"""Snake, played live by the model: every move is one /v1/systemone call with the non-suicidal directions as options.

    python demos/snake_game.py --endpoint http://localhost:3009 --games 3 --out demos/out/snake

The state is the grid as text plus head, apple and direction. Nothing steers the model toward the apple: the option descriptions only
say which direction each move is. `--player greedy` plays shortest path to the apple (a strong baseline); `--player random` picks any safe move.
Frames are written as SVG and turned into a GIF with rsvg-convert + ffmpeg when available.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import Player

W = H = 12
DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
LEVELS = ["very unsafe", "unsafe", "fine", "good", "very good"]


class Snake:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.body = deque([(W // 2, H // 2), (W // 2 - 1, H // 2), (W // 2 - 2, H // 2)])
        self.direction = "right"
        self.apple = self.place()
        self.score = 0
        self.alive = True

    def place(self):
        free = [(x, y) for x in range(W) for y in range(H) if (x, y) not in self.body]
        return self.rng.choice(free)

    def safe_moves(self) -> list[str]:
        hx, hy = self.body[0]
        out = []
        for d, (dx, dy) in DIRS.items():
            if d == OPPOSITE[self.direction]:
                continue
            nx, ny = hx + dx, hy + dy
            tail_free = set(self.body) - {self.body[-1]}  # the tail moves away unless we eat
            if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in tail_free:
                out.append(d)
        return out

    def step(self, d: str):
        dx, dy = DIRS[d]
        hx, hy = self.body[0]
        nx, ny = hx + dx, hy + dy
        self.direction = d
        if not (0 <= nx < W and 0 <= ny < H) or (nx, ny) in set(self.body) - {self.body[-1]}:
            self.alive = False
            return
        self.body.appendleft((nx, ny))
        if (nx, ny) == self.apple:
            self.score += 1
            self.apple = self.place()
        else:
            self.body.pop()

    def grid(self) -> str:
        rows = []
        for y in range(H):
            row = ""
            for x in range(W):
                row += "H" if (x, y) == self.body[0] else "o" if (x, y) in self.body else "A" if (x, y) == self.apple else "."
            rows.append(row)
        return "\n".join(rows)

    def state_text(self) -> str:
        hx, hy = self.body[0]
        ax, ay = self.apple
        rel = f"{abs(ay - hy)} {'up' if ay < hy else 'down'}" if ay != hy else ""
        rel += (", " if rel else "") + (f"{abs(ax - hx)} {'left' if ax < hx else 'right'}" if ax != hx else "")
        return (
            f"Snake on a {W}x{H} grid. H = your head, o = your body, A = the apple, . = empty. Row 0 is the top; up means a smaller row number.\n"
            f"{self.grid()}\nHead at column {hx}, row {hy}. Moving {self.direction}. Length {len(self.body)}. Score {self.score}.\n"
            f"The apple is {rel or 'here'}. Eat apples; do not hit the wall or your body."
        )

    def greedy(self, moves: list[str]) -> str:
        """Shortest path to the apple through free cells; falls back to the move with the most room."""
        start = self.body[0]
        blocked = set(self.body) - {self.body[-1]}
        prev = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == self.apple:
                while prev[cur] != start:
                    cur = prev[cur]
                for d, (dx, dy) in DIRS.items():
                    if (start[0] + dx, start[1] + dy) == cur and d in moves:
                        return d
                break
            for dx, dy in DIRS.values():
                nxt = (cur[0] + dx, cur[1] + dy)
                if 0 <= nxt[0] < W and 0 <= nxt[1] < H and nxt not in blocked and nxt not in prev:
                    prev[nxt] = cur
                    q.append(nxt)
        return moves[0]


def svg(game: Snake, score_text: str) -> str:
    cell = 28
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W * cell}" height="{H * cell + 34}" viewBox="0 0 {W * cell} {H * cell + 34}"><rect width="100%" height="100%" fill="#101418"/>'
    ]
    parts.extend(f'<rect x="{x * cell + 1}" y="{y * cell + 1}" width="{cell - 2}" height="{cell - 2}" fill="#1b2128"/>' for x in range(W) for y in range(H))
    ax, ay = game.apple
    parts.append(f'<circle cx="{ax * cell + cell / 2}" cy="{ay * cell + cell / 2}" r="{cell * 0.36}" fill="#e5484d"/>')
    for i, (x, y) in enumerate(game.body):
        parts.append(f'<rect x="{x * cell + 2}" y="{y * cell + 2}" width="{cell - 4}" height="{cell - 4}" rx="6" fill="{"#65dba4" if i == 0 else "#2f9e6a"}"/>')
    parts.append(f'<text x="8" y="{H * cell + 23}" fill="#e8e8e3" font-family="-apple-system,Helvetica,Arial" font-size="15">{score_text}</text></svg>')
    return "".join(parts)


@dataclass
class Run:
    rng: random.Random
    out: Path
    max_moves: int
    log: Any


def play(player, game_no: int, run: Run) -> dict:
    rng, out, max_moves, log = run.rng, run.out, run.max_moves, run.log
    game = Snake(rng)
    frames = out / f"frames_{game_no}"
    frames.mkdir(exist_ok=True)
    moves = 0
    ms_total = 0.0
    (frames / "000.svg").write_text(svg(game, f"game {game_no + 1}  score 0"))
    while game.alive and moves < max_moves:
        safe = game.safe_moves()
        if not safe:
            game.alive = False
            break
        options = {d: f"move {d}" for d in safe}
        if isinstance(player, Player):
            answers, ms = player.ask(
                game.state_text(),
                {
                    "move": {"type": "choice", "instructions": "Which way do you move next?", "criteria": options},
                    "outlook": {"type": "score", "instructions": "How is this game going for you?", "criteria": LEVELS},
                },
            )
            d = answers["move"]["choice"]
            log.write(
                json.dumps(
                    {
                        "game": game_no,
                        "move": moves,
                        "choice": d,
                        "options": safe,
                        "confidence": answers["move"]["confidence"],
                        "outlook": answers["outlook"]["score"],
                        "score": game.score,
                        "ms": round(ms),
                    }
                )
                + "\n"
            )
            ms_total += ms
        elif player == "greedy":
            d = game.greedy(safe)
        else:
            d = rng.choice(safe)
        game.step(d)
        moves += 1
        (frames / f"{moves:03d}.svg").write_text(
            svg(game, f"game {game_no + 1}  score {game.score}  move {moves}" + (f"  {round(ms_total / moves)} ms/move" if ms_total else ""))
        )
    return {"game": game_no, "score": game.score, "moves": moves, "alive": game.alive, "ms_per_move": round(ms_total / moves) if moves and ms_total else None}


def gif(out: Path, games: int) -> str | None:
    if not (shutil.which("rsvg-convert") and shutil.which("ffmpeg")):
        return None
    i = 0
    pngs = out / "png"
    pngs.mkdir(exist_ok=True)
    for g in range(games):
        for f in sorted((out / f"frames_{g}").glob("*.svg")):
            subprocess.run(["rsvg-convert", "-o", str(pngs / f"{i:04d}.png"), str(f)], check=True)
            i += 1
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            "6",
            "-i",
            str(pngs / "%04d.png"),
            "-vf",
            "split[a][b];[a]palettegen=max_colors=32[p];[b][p]paletteuse",
            str(out / "snake.gif"),
        ],
        check=True,
    )
    shutil.rmtree(pngs)
    for g in range(games):
        shutil.rmtree(out / f"frames_{g}")
    return str(out / "snake.gif")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="http://localhost:3000")
    ap.add_argument("--model", default="openjev")
    ap.add_argument("--player", default="model", help="model | greedy | random")
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--max-moves", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="demos/out/snake")
    ap.add_argument("--no-gif", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    player = Player(a.endpoint, a.model, name="model") if a.player == "model" else a.player
    rng = random.Random(a.seed)
    results = []
    t0 = time.perf_counter()
    with (out / "decisions.jsonl").open("w") as log:
        for g in range(a.games):
            r = play(player, g, Run(rng, out, a.max_moves, log))
            results.append(r)
            print(json.dumps(r), flush=True)
    summary = {
        "player": a.player,
        "games": a.games,
        "scores": [r["score"] for r in results],
        "mean_score": round(sum(r["score"] for r in results) / a.games, 2),
        "moves": sum(r["moves"] for r in results),
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    if isinstance(player, Player):
        summary["avg_ms_per_move"] = round(player.avg_ms())
        summary["moves_per_second"] = round(1000 / player.avg_ms(), 2)
    summary["gif"] = None if a.no_gif else gif(out, a.games)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
