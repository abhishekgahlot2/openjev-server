import math
import random

from openjev_server.calibrate import fit


def test_fit_recovers_temperature_and_scale():
    rng = random.Random(1)
    rows = []
    for _ in range(600):  # choice rows whose probabilities were sharpened by 1/0.7: the fit should undo that (T about 0.7)
        raw = [rng.gauss(0, 1.5) for _ in range(4)]
        gold = max(range(4), key=lambda i: raw[i] + rng.gauss(0, 1.0))
        z = [v / 0.7 for v in raw]
        m = max(z)
        p = [math.exp(v - m) for v in z]
        s = sum(p)
        rows.append({"gold": str(gold), "answer": {"type": "choice", "probabilities": {str(i): round(pi / s, 4) for i, pi in enumerate(p)}}})
    for _ in range(400):
        lg = rng.gauss(0, 2.0)
        y = rng.random() < 1 / (1 + math.exp(-lg / 1.5))
        rows.append({"gold": y, "answer": {"type": "noul", "noul": round(1 / (1 + math.exp(-lg)), 4)}})
    r = fit(rows)
    assert r["n_choice"] == 600 and r["n_noul"] == 400
    assert r["choice_nll_after"] <= r["choice_nll_before"] and r["noul_nll_after"] <= r["noul_nll_before"]
    assert 1.0 < r["noul_total_scale"] < 2.5
