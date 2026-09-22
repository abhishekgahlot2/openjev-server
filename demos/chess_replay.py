"""Build a self-contained replay page for a recorded chess game (the data is inlined, so the page works from a file or a static host).

python demos/chess_replay.py demos/out/chess_openjev_vs_random demos/out/chess_openjev_vs_random/replay.html
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import chess

TEMPLATE = Path(__file__).with_name("replay_chess.html")


def game_data(game_dir: Path) -> dict:
    rows = [json.loads(line) for line in (game_dir / "decisions.jsonl").open()]
    summary = json.loads((game_dir / "summary.json").read_text())
    board = chess.Board()
    plies = []
    for r in rows:
        mv = board.parse_san(r["choice"])
        board.push(mv)
        squares = [(7 - chess.square_rank(sq)) * 8 + chess.square_file(sq) for sq in (mv.from_square, mv.to_square)]
        plies.append({**r, "fen_after": board.fen(), "squares": squares})
    return {"result": summary["result"], "white": summary["white"], "black": summary["black"], "plies": plies}


def main(game_dir: str, out: str) -> None:
    data = json.dumps(game_data(Path(game_dir)))
    page = TEMPLATE.read_text().replace("__GAME_DATA__", data.replace("</", "<\\/"))
    Path(out).write_text(page)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
