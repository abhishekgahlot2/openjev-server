"""Chess between two decision models. The legal moves are the options; every ply is one /v1/systemone call.

    python demos/chess_game.py --white http://localhost:3009 --black http://localhost:8009 --black-model kev-latest --out demos/out/chess

Nothing is scripted: the model sees the board, the move list and the legal moves, and picks one. A `score` question rates the
position from the mover's side, logged for the record. `--white random` plays uniformly random legal moves (sanity baseline).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import chess
import chess.pgn
from common import Player

LEVELS = ["losing", "worse", "equal", "better", "winning"]
PIECE = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop", chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}


def describe(board: chess.Board, move: chess.Move) -> str:
    piece = PIECE[board.piece_type_at(move.from_square)]
    text = f"{piece} {chess.square_name(move.from_square)} to {chess.square_name(move.to_square)}"
    if board.is_castling(move):
        text = "castle kingside" if chess.square_file(move.to_square) == 6 else "castle queenside"
    captured = board.piece_type_at(move.to_square)
    if captured or board.is_en_passant(move):
        text += f", captures {PIECE[captured] if captured else 'pawn'}"
    if move.promotion:
        text += f", promotes to {PIECE[move.promotion]}"
    board.push(move)
    if board.is_checkmate():
        text += ", checkmate"
    elif board.is_check():
        text += ", check"
    board.pop()
    return text


def material(board: chess.Board) -> str:
    v = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    w = sum(v[p] * len(board.pieces(p, chess.WHITE)) for p in v)
    b = sum(v[p] * len(board.pieces(p, chess.BLACK)) for p in v)
    return f"material White {w}, Black {b}"


def state_text(board: chess.Board, sans: list[str]) -> str:
    side = "White" if board.turn else "Black"
    moves = " ".join(f"{i // 2 + 1}.{'' if i % 2 == 0 else '..'} {s}" for i, s in enumerate(sans[-16:]))
    return (
        f"You are playing chess as {side}. It is your move.\n"
        f"Board (uppercase = White, lowercase = black, rank 8 at the top):\n{board}\n"
        f"FEN: {board.fen()}\n{material(board)}\nRecent moves: {moves or '(game start)'}\n"
        f"{'You are in check. ' if board.is_check() else ''}Pick the best legal move."
    )


def play(white: Player, black: Player, max_plies: int, seed: int, out: Path) -> dict:
    rng = random.Random(seed)
    board = chess.Board()
    sans: list[str] = []
    log = (out / "decisions.jsonl").open("w")
    while not board.is_game_over(claim_draw=True) and len(sans) < max_plies:
        player = white if board.turn else black
        legal = list(board.legal_moves)
        options = {board.san(m): describe(board, m) for m in legal}
        rec = {"ply": len(sans) + 1, "side": "white" if board.turn else "black", "player": player.name, "fen": board.fen(), "legal": len(legal)}
        if player.random:
            san = board.san(rng.choice(legal))
            rec.update(choice=san, ms=0)
        else:
            questions = {
                "move": {"type": "choice", "instructions": "Which legal move should you play?", "criteria": options},
                "eval": {"type": "score", "instructions": "How good is your position right now?", "criteria": LEVELS},
            }
            answers, ms = player.ask(state_text(board, sans), questions)
            san = answers["move"]["choice"]
            probs = answers["move"]["probabilities"]
            top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
            rec.update(choice=san, confidence=answers["move"]["confidence"], top3=top, eval=answers["eval"]["score"], ms=round(ms))
        board.push_san(san)
        sans.append(san)
        log.write(json.dumps(rec) + "\n")
        line = f"{rec['ply']:3d} {rec['side']:5s} {san:7s} {rec.get('ms', 0):5d} ms  conf {rec.get('confidence', '-')}  eval {rec.get('eval', '-')}"
        print(f"{line}  ({rec['legal']} legal)", flush=True)
    log.close()
    result = board.result(claim_draw=True)
    game = chess.pgn.Game.from_board(board)
    game.headers.update(Event="openjev-server demo", White=white.name, Black=black.name, Result=result, Date=time.strftime("%Y.%m.%d"))
    (out / "game.pgn").write_text(str(game) + "\n")
    summary = {
        "result": result,
        "plies": len(sans),
        "termination": "checkmate" if board.is_checkmate() else "stalemate" if board.is_stalemate() else "draw claim / ply cap",
        "white": {"name": white.name, "calls": white.calls, "avg_ms": round(white.avg_ms())},
        "black": {"name": black.name, "calls": black.calls, "avg_ms": round(black.avg_ms())},
        "final_fen": board.fen(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--white", required=True, help="endpoint or 'random'")
    ap.add_argument("--black", required=True)
    ap.add_argument("--white-model", default="openjev")
    ap.add_argument("--black-model", default="openjev")
    ap.add_argument("--white-name", default="")
    ap.add_argument("--black-name", default="")
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="demos/out/chess")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    play(Player(a.white, a.white_model, name=a.white_name), Player(a.black, a.black_model, name=a.black_name), a.max_plies, a.seed, out)


if __name__ == "__main__":
    main()
