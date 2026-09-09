import json
import os

import pytest

import app as app_module


@pytest.fixture
def client(tmp_path, monkeypatch):
    data_file = tmp_path / "data.json"
    monkeypatch.setattr(app_module, "DATA_FILE", str(data_file))
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as client:
        yield client


def test_index_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Soccer Coach" in resp.data


def test_add_and_list_player(client):
    resp = client.post(
        "/api/players",
        data=json.dumps({"name": "Alex", "number": 7, "position": "ST"}),
        content_type="application/json",
    )
    assert resp.status_code == 201
    player = resp.get_json()
    assert player["name"] == "Alex"

    resp = client.get("/api/players")
    players = resp.get_json()
    assert len(players) == 1
    assert players[0]["name"] == "Alex"


def test_add_player_requires_name(client):
    resp = client.post(
        "/api/players", data=json.dumps({}), content_type="application/json"
    )
    assert resp.status_code == 400


def test_delete_player_clears_lineup_assignment(client):
    resp = client.post(
        "/api/players",
        data=json.dumps({"name": "Sam"}),
        content_type="application/json",
    )
    player_id = resp.get_json()["id"]

    client.post(
        "/api/lineup",
        data=json.dumps({"formation": "4-4-2", "assignments": {"0": player_id}}),
        content_type="application/json",
    )

    client.delete(f"/api/players/{player_id}")

    lineup = client.get("/api/lineup").get_json()
    assert lineup["assignments"] == {}


def test_lineup_rejects_unknown_formation(client):
    resp = client.post(
        "/api/lineup",
        data=json.dumps({"formation": "5-5-5"}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_match_lifecycle(client):
    resp = client.post(
        "/api/matches",
        data=json.dumps({"opponent": "Riverside FC", "date": "2026-09-01"}),
        content_type="application/json",
    )
    assert resp.status_code == 201
    match_id = resp.get_json()["id"]

    resp = client.put(
        f"/api/matches/{match_id}",
        data=json.dumps({"our_score": 3, "opponent_score": 1}),
        content_type="application/json",
    )
    assert resp.get_json()["our_score"] == 3

    resp = client.delete(f"/api/matches/{match_id}")
    assert resp.get_json()["status"] == "deleted"

    matches = client.get("/api/matches").get_json()
    assert matches == []
