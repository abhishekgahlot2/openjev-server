"""Write demos/out/index.html: one page that links every recorded demo output in demos/out (GIFs, replay pages, rendered UIs, summaries)."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "out"
TITLES = {
    "chess": "Chess: OpenJev vs a random mover",
    "poker": "Poker: heads-up no-limit hold'em, OpenJev vs a rule bot",
    "browser": "Browser agent on the laptop",
    "policy": "Returns policy: 200 generated cases, exact ground truth",
    "genui": "Generative UI: ten dashboard decisions in one call",
    "recorded_gpu": "Recorded on one H100 with the FP8 build (inbox triage, dashboard assembly, Google Flights, shopping, page QA)",
    "snake": "Snake, played live by the model",
    "guardrail": "Agent guardrail: is this tool call allowed under the policy?",
    "nl_filter": "Natural-language filters over table rows, eight per call",
}


def section(d: Path) -> str:
    kind = next((k for k in TITLES if d.name.startswith(k)), d.name)
    items = []
    for f in sorted(d.iterdir()):
        if f.suffix == ".mp4":
            items.append(
                f'<figure><video src="{d.name}/{f.name}" controls muted playsinline preload="metadata" style="max-width:100%;border-radius:8px"></video><figcaption>{f.name}</figcaption></figure>'
            )
        elif f.suffix == ".gif" and not (f.with_suffix(".mp4")).exists():
            items.append(f'<figure><img src="{d.name}/{f.name}" alt="{f.stem}"><figcaption>{f.name}</figcaption></figure>')
        elif f.suffix == ".png":
            items.append(
                f'<figure><a href="{d.name}/{f.stem}.html"><img src="{d.name}/{f.name}" alt="{f.stem}"></a><figcaption>{f.stem} (click for the page)</figcaption></figure>'
            )
        elif f.name in ("replay.html",):
            items.append(f'<p><a class="btn" href="{d.name}/{f.name}">Open the interactive replay</a></p>')
        elif f.name in ("summary.json", "run.json", "analysis.json"):
            items.append(f"<pre>{json.dumps(json.loads(f.read_text()), indent=1)}</pre>")
        elif f.suffix == ".pgn":
            items.append(f"<pre>{f.read_text()}</pre>")
    return f"<section><h2>{TITLES.get(kind, d.name)} <small>{d.name}</small></h2>{''.join(items)}</section>"


def main():
    dirs = [d for d in sorted(OUT.iterdir()) if d.is_dir() and d.name != "frames"]
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenJev demos</title>
<style>:root{{color-scheme:light dark}}body{{margin:0;padding:24px 16px;font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:light-dark(#f7f7f4,#0f1115);color:light-dark(#1c1c1c,#e8e8e3)}}
main{{max-width:1000px;margin:0 auto}}h1{{font-size:26px}}h2{{font-size:20px;margin:32px 0 8px}}h2 small{{opacity:.6;font-weight:400;font-size:13px;margin-left:8px}}section{{background:light-dark(#fff,#181b22);border-radius:10px;padding:16px 20px;margin-bottom:20px}}
figure{{margin:12px 0}}img{{max-width:100%;border-radius:8px;border:1px solid rgba(127,127,127,.3)}}figcaption{{font-size:13px;opacity:.7}}pre{{overflow-x:auto;background:rgba(127,127,127,.12);padding:12px;border-radius:8px;font-size:12.5px}}
.btn{{display:inline-block;padding:8px 14px;border-radius:8px;background:#247653;color:#fff;text-decoration:none;font-weight:600}}</style></head><body><main>
<h1>OpenJev demos</h1><p>Every output here was produced by the real model on this machine (OpenJev MLX 4-bit, Apple M5 Max). Nothing is scripted: each decision is one <code>/v1/systemone</code> call, logged in <code>decisions.jsonl</code> next to the outputs.</p>
{"".join(section(d) for d in dirs)}
</main></body></html>"""
    (OUT / "index.html").write_text(html)
    print(f"wrote {OUT / 'index.html'} with {len(dirs)} demos")


if __name__ == "__main__":
    main()
