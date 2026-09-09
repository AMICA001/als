# Soccer Coach App

A small Flask app for youth/club soccer coaches to manage a roster, build
a starting lineup on a visual pitch, and track match results.

## Features

- **Roster** — add/remove players with number and preferred position.
- **Lineup** — pick a formation (4-4-2, 4-3-3, 3-5-2, 4-2-3-1) and assign
  roster players to each position on a visual pitch; unassigned players
  show up on the bench automatically.
- **Matches** — schedule matches against opponents and record scores.

Data is persisted to `data.json` in this directory (no database required).

## Running

```bash
cd soccer_coach_app
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5050

## Tests

```bash
cd soccer_coach_app
pytest
```
