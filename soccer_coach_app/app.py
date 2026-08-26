import json
import os
import uuid
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

DATA_FILE = os.path.join(os.path.dirname(__file__), "data.json")

DEFAULT_DATA = {
    "players": [],
    "matches": [],
    "lineup": {"formation": "4-4-2", "assignments": {}},
}

FORMATIONS = {
    "4-4-2": ["GK", "LB", "CB", "CB", "RB", "LM", "CM", "CM", "RM", "ST", "ST"],
    "4-3-3": ["GK", "LB", "CB", "CB", "RB", "CM", "CM", "CM", "LW", "ST", "RW"],
    "3-5-2": ["GK", "CB", "CB", "CB", "LM", "CM", "CM", "CM", "RM", "ST", "ST"],
    "4-2-3-1": ["GK", "LB", "CB", "CB", "RB", "CDM", "CDM", "LW", "CAM", "RW", "ST"],
}


def load_data():
    if not os.path.exists(DATA_FILE):
        return json.loads(json.dumps(DEFAULT_DATA))
    with open(DATA_FILE, "r") as f:
        return json.load(f)


def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


@app.route("/")
def index():
    return render_template("index.html", formations=list(FORMATIONS.keys()))


@app.route("/api/formations")
def get_formations():
    return jsonify(FORMATIONS)


@app.route("/api/players", methods=["GET", "POST"])
def players():
    data = load_data()
    if request.method == "POST":
        body = request.get_json(force=True)
        name = (body.get("name") or "").strip()
        if not name:
            return jsonify({"error": "name is required"}), 400
        player = {
            "id": str(uuid.uuid4()),
            "name": name,
            "number": body.get("number"),
            "position": body.get("position", ""),
        }
        data["players"].append(player)
        save_data(data)
        return jsonify(player), 201
    return jsonify(data["players"])


@app.route("/api/players/<player_id>", methods=["DELETE"])
def delete_player(player_id):
    data = load_data()
    data["players"] = [p for p in data["players"] if p["id"] != player_id]
    data["lineup"]["assignments"] = {
        slot: pid
        for slot, pid in data["lineup"]["assignments"].items()
        if pid != player_id
    }
    save_data(data)
    return jsonify({"status": "deleted"})


@app.route("/api/lineup", methods=["GET", "POST"])
def lineup():
    data = load_data()
    if request.method == "POST":
        body = request.get_json(force=True)
        formation = body.get("formation")
        assignments = body.get("assignments")
        if formation is not None:
            if formation not in FORMATIONS:
                return jsonify({"error": "unknown formation"}), 400
            data["lineup"]["formation"] = formation
        if assignments is not None:
            data["lineup"]["assignments"] = assignments
        save_data(data)
    return jsonify(data["lineup"])


@app.route("/api/matches", methods=["GET", "POST"])
def matches():
    data = load_data()
    if request.method == "POST":
        body = request.get_json(force=True)
        opponent = (body.get("opponent") or "").strip()
        if not opponent:
            return jsonify({"error": "opponent is required"}), 400
        match = {
            "id": str(uuid.uuid4()),
            "opponent": opponent,
            "date": body.get("date", ""),
            "location": body.get("location", ""),
            "our_score": body.get("our_score"),
            "opponent_score": body.get("opponent_score"),
        }
        data["matches"].append(match)
        save_data(data)
        return jsonify(match), 201
    return jsonify(data["matches"])


@app.route("/api/matches/<match_id>", methods=["PUT", "DELETE"])
def match_detail(match_id):
    data = load_data()
    if request.method == "DELETE":
        data["matches"] = [m for m in data["matches"] if m["id"] != match_id]
        save_data(data)
        return jsonify({"status": "deleted"})

    body = request.get_json(force=True)
    for m in data["matches"]:
        if m["id"] == match_id:
            for field in ("opponent", "date", "location", "our_score", "opponent_score"):
                if field in body:
                    m[field] = body[field]
            save_data(data)
            return jsonify(m)
    return jsonify({"error": "match not found"}), 404


if __name__ == "__main__":
    app.run(debug=True, port=5050)
