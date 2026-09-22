from fastapi.testclient import TestClient

from openjev_server.profile import Profile
from openjev_server.readout import Readout
from openjev_server.server import build_app

REQ = {
    "model": "x",
    "state": "I was charged twice.",
    "questions": {"route": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "shipping": None}}},
}


def client(fake_backend, token=""):
    return TestClient(build_app(Readout(fake_backend, Profile()), token=token, model_dir="/models/openjev", backend_name="fake"))


def test_systemone_shape(fake_backend):
    r = client(fake_backend).post("/v1/systemone", json=REQ)
    assert r.status_code == 200
    d = r.json()
    assert d["id"].startswith("oj-") and d["usage"]["output_tokens"] == 0 and d["usage"]["input_tokens"] > 0
    a = d["answers"]["route"]
    assert a["type"] == "choice" and a["choice"] in ("billing", "shipping") and abs(sum(a["probabilities"].values()) - 1) < 0.01 and 0 <= a["confidence"] <= 1


def test_auth(fake_backend):
    c = client(fake_backend, token="secret")
    assert c.post("/v1/systemone", json=REQ).status_code == 401
    assert c.get("/v1/version").status_code == 401
    assert c.post("/v1/systemone", json=REQ, headers={"Authorization": "Bearer secret"}).status_code == 200
    assert c.get("/healthz").status_code == 200  # never behind auth


def test_validation(fake_backend):
    c = client(fake_backend)
    assert c.post("/v1/systemone", json={"state": "s", "questions": {}}).status_code == 422
    assert c.post("/v1/systemone", json={"state": "s", "questions": {"q": {"type": "essay"}}}).status_code == 422
    r = c.post("/v1/systemone", json={"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {}}}})
    assert r.status_code == 422 and "criteria" in r.json()["error"]["message"]


def test_version_health_metrics_prewarm(fake_backend):
    c = client(fake_backend)
    v = c.get("/v1/version").json()
    assert v["model_dir"] == "openjev" and v["T"] == 0.85 and v["flags"]["targeted"] is True and len(v["readout_sha256"]) == 64
    assert c.get("/readyz").json()["ok"] is True
    c.post("/v1/systemone", json=REQ)
    m = c.get("/metrics").text
    assert "openjev_requests_total" in m and "openjev_request_seconds" in m
    assert c.post("/v1/prewarm", json={"state": REQ["state"]}).json()["ok"] is True
