const FORMATION_COORDS = {
  "4-4-2": [
    [50, 92],
    [15, 72], [38, 75], [62, 75], [85, 72],
    [15, 45], [38, 48], [62, 48], [85, 45],
    [35, 15], [65, 15],
  ],
  "4-3-3": [
    [50, 92],
    [15, 72], [38, 75], [62, 75], [85, 72],
    [30, 45], [50, 50], [70, 45],
    [15, 18], [50, 12], [85, 18],
  ],
  "3-5-2": [
    [50, 92],
    [30, 75], [50, 78], [70, 75],
    [10, 45], [30, 50], [50, 53], [70, 50], [90, 45],
    [35, 15], [65, 15],
  ],
  "4-2-3-1": [
    [50, 92],
    [15, 72], [38, 75], [62, 75], [85, 72],
    [38, 55], [62, 55],
    [15, 30], [50, 32], [85, 30],
    [50, 12],
  ],
};

let state = {
  players: [],
  formations: {},
  lineup: { formation: "4-4-2", assignments: {} },
};

async function api(path, opts) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  return res.json();
}

function switchTab(name) {
  document.querySelectorAll(".tab-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.tab === name)
  );
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("active", t.id === `tab-${name}`)
  );
}

function renderFormationSelect() {
  const sel = document.getElementById("formation-select");
  sel.innerHTML = "";
  Object.keys(state.formations).forEach((f) => {
    const opt = document.createElement("option");
    opt.value = f;
    opt.textContent = f;
    if (f === state.lineup.formation) opt.selected = true;
    sel.appendChild(opt);
  });
}

function playerLabel(p) {
  return `${p.number ? "#" + p.number + " " : ""}${p.name}`;
}

function renderPitch() {
  const pitch = document.getElementById("pitch");
  pitch.innerHTML = "";
  const formation = state.lineup.formation;
  const labels = state.formations[formation] || [];
  const coords = FORMATION_COORDS[formation] || [];
  const assignments = state.lineup.assignments || {};

  labels.forEach((label, i) => {
    const coord = coords[i] || [50, 50];
    const slot = document.createElement("div");
    slot.className = "slot";
    slot.style.left = coord[0] + "%";
    slot.style.top = coord[1] + "%";

    const labelEl = document.createElement("div");
    labelEl.className = "slot-label";
    labelEl.textContent = label;
    slot.appendChild(labelEl);

    const select = document.createElement("select");
    const emptyOpt = document.createElement("option");
    emptyOpt.value = "";
    emptyOpt.textContent = "-- empty --";
    select.appendChild(emptyOpt);

    state.players.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = playerLabel(p);
      if (assignments[String(i)] === p.id) opt.selected = true;
      select.appendChild(opt);
    });

    select.addEventListener("change", () => {
      const newAssignments = { ...state.lineup.assignments };
      if (select.value) {
        // remove player from any other slot first
        Object.keys(newAssignments).forEach((slotKey) => {
          if (newAssignments[slotKey] === select.value) delete newAssignments[slotKey];
        });
        newAssignments[String(i)] = select.value;
      } else {
        delete newAssignments[String(i)];
      }
      state.lineup.assignments = newAssignments;
      saveLineup();
      renderPitch();
      renderBench();
    });

    slot.appendChild(select);
    pitch.appendChild(slot);
  });
}

function renderBench() {
  const benchList = document.getElementById("bench-list");
  benchList.innerHTML = "";
  const assignedIds = new Set(Object.values(state.lineup.assignments || {}));
  state.players
    .filter((p) => !assignedIds.has(p.id))
    .forEach((p) => {
      const li = document.createElement("li");
      li.textContent = playerLabel(p) + (p.position ? ` (${p.position})` : "");
      benchList.appendChild(li);
    });
}

async function saveLineup() {
  await api("/api/lineup", {
    method: "POST",
    body: JSON.stringify(state.lineup),
  });
}

function renderRoster() {
  const tbody = document.querySelector("#roster-table tbody");
  tbody.innerHTML = "";
  state.players.forEach((p) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${p.number ?? ""}</td>
      <td>${p.name}</td>
      <td>${p.position || ""}</td>
      <td><button class="delete-btn" data-id="${p.id}">Remove</button></td>
    `;
    tbody.appendChild(tr);
  });
  tbody.querySelectorAll(".delete-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      await api(`/api/players/${btn.dataset.id}`, { method: "DELETE" });
      await loadPlayers();
      renderRoster();
      renderPitch();
      renderBench();
    });
  });
}

async function loadPlayers() {
  state.players = await api("/api/players");
}

async function loadFormations() {
  state.formations = await api("/api/formations");
}

async function loadLineup() {
  state.lineup = await api("/api/lineup");
  if (!state.lineup.assignments) state.lineup.assignments = {};
}

function renderMatches() {
  const tbody = document.querySelector("#matches-table tbody");
  tbody.innerHTML = "";
  state.matches.forEach((m) => {
    const tr = document.createElement("tr");
    const score =
      m.our_score != null && m.opponent_score != null
        ? `${m.our_score} - ${m.opponent_score}`
        : "";
    tr.innerHTML = `
      <td>${m.date || ""}</td>
      <td>${m.opponent}</td>
      <td>${m.location || ""}</td>
      <td>${score}</td>
      <td><button class="delete-btn" data-id="${m.id}">Remove</button></td>
    `;
    tbody.appendChild(tr);
  });
  tbody.querySelectorAll(".delete-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      await api(`/api/matches/${btn.dataset.id}`, { method: "DELETE" });
      await loadMatches();
      renderMatches();
    });
  });
}

async function loadMatches() {
  state.matches = await api("/api/matches");
}

async function init() {
  document.querySelectorAll(".tab-btn").forEach((btn) =>
    btn.addEventListener("click", () => switchTab(btn.dataset.tab))
  );

  await Promise.all([loadFormations(), loadPlayers(), loadLineup(), loadMatches()]);

  renderFormationSelect();
  renderPitch();
  renderBench();
  renderRoster();
  renderMatches();

  document.getElementById("formation-select").addEventListener("change", async (e) => {
    state.lineup.formation = e.target.value;
    state.lineup.assignments = {};
    await saveLineup();
    renderPitch();
    renderBench();
  });

  document.getElementById("player-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = document.getElementById("player-name").value.trim();
    const number = document.getElementById("player-number").value;
    const position = document.getElementById("player-position").value.trim();
    if (!name) return;
    await api("/api/players", {
      method: "POST",
      body: JSON.stringify({ name, number: number ? Number(number) : null, position }),
    });
    e.target.reset();
    await loadPlayers();
    renderRoster();
    renderPitch();
    renderBench();
  });

  document.getElementById("match-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const opponent = document.getElementById("match-opponent").value.trim();
    const date = document.getElementById("match-date").value;
    const location = document.getElementById("match-location").value.trim();
    if (!opponent) return;
    await api("/api/matches", {
      method: "POST",
      body: JSON.stringify({ opponent, date, location }),
    });
    e.target.reset();
    await loadMatches();
    renderMatches();
  });
}

init();
