// Phase 4 dashboard — per-run view. Plots E-Star and latent drift,
// then subscribes to the run's WebSocket for live updates.
//
// Security note: the API key is held in a JS variable for the lifetime
// of the page only. We deliberately do not persist it (see
// dashboard.js for the full rationale).
(function () {
  "use strict";

  // Defensive: reject IDs that don't look like a UUID-ish slug. This
  // prevents weird characters from being smuggled into the API path
  // even though encodeURIComponent already handles escaping correctly.
  const RUN_ID_RE = /^[A-Za-z0-9_-]{1,64}$/;

  const params = new URLSearchParams(location.search);
  const runId = params.get("id");

  const labelEl = document.getElementById("run-label");
  const liveBadge = document.getElementById("m-live");
  const specEl = document.getElementById("latest-spec");

  if (!runId || !RUN_ID_RE.test(runId)) {
    labelEl.textContent = "invalid run id — return to dashboard";
    return;
  }
  labelEl.textContent = runId;

  // Prompt for the API key (the dashboard intentionally doesn't share
  // it across pages). `window.prompt` keeps the value in memory only.
  const apiKey = (window.prompt("API key for run " + runId.slice(0, 8) + "…") || "").trim();
  if (!apiKey) {
    labelEl.textContent = "no API key supplied — return to dashboard";
    return;
  }

  // -------------------- chart scaffolding --------------------
  const estarCtx = document.getElementById("estar-chart");
  const driftCtx = document.getElementById("drift-chart");

  const estarChart = new Chart(estarCtx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "E★",
          data: [],
          borderColor: "#7c5cff",
          backgroundColor: "rgba(124,92,255,0.15)",
          fill: true,
          tension: 0.25,
          pointRadius: 2,
        },
      ],
    },
    options: chartOptions("step (t)", "E★"),
  });

  const driftChart = new Chart(driftCtx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "‖state‖",
          data: [],
          borderColor: "#2dd4bf",
          backgroundColor: "rgba(45,212,191,0.10)",
          fill: false,
          tension: 0.25,
          pointRadius: 2,
        },
        {
          label: "Δ (drift)",
          data: [],
          borderColor: "#f59e0b",
          backgroundColor: "rgba(245,158,11,0.10)",
          fill: false,
          tension: 0.25,
          pointRadius: 2,
          spanGaps: false,
        },
      ],
    },
    options: chartOptions("step (t)", "norm / Δ"),
  });

  // -------------------- initial fetch --------------------
  let prevState = null;
  fetchMetrics().then(connectWs).catch((err) => {
    labelEl.textContent = "failed to load: " + err.message;
  });

  async function fetchMetrics() {
    const resp = await fetch(
      "/runs/" + encodeURIComponent(runId) + "/metrics",
      { headers: { "X-API-Key": apiKey } },
    );
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const data = await resp.json();

    for (const p of data.e_star || []) {
      pushPoint(estarChart, 0, p.t, p.e_star);
    }
    for (const p of data.drift || []) {
      pushPoint(driftChart, 0, p.t, p.state_norm);
      pushPoint(driftChart, 1, p.t, p.state_delta);
    }
    estarChart.update("none");
    driftChart.update("none");
    renderSummary(data.summary);

    // Fetch the latest spec from /trace once (the WS will replace it later).
    const traceResp = await fetch(
      "/runs/" + encodeURIComponent(runId) + "/trace",
      { headers: { "X-API-Key": apiKey } },
    );
    if (traceResp.ok) {
      const traceData = await traceResp.json();
      const trace = (traceData.run && traceData.run.trace) || [];
      if (trace.length) {
        renderSpec(trace[trace.length - 1].spec);
      }
    }
  }

  // -------------------- websocket --------------------
  function connectWs() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const url =
      proto +
      "//" +
      location.host +
      "/runs/" +
      encodeURIComponent(runId) +
      "/stream?" +
      new URLSearchParams({ x_api_key: apiKey }).toString();

    const ws = new WebSocket(url);
    ws.onopen = () => setLive(true);
    ws.onclose = () => setLive(false);
    ws.onerror = () => setLive(false);
    ws.onmessage = (msg) => {
      let event;
      try {
        event = JSON.parse(msg.data);
      } catch (err) {
        // Surface to the console so malformed server payloads are
        // visible during debugging rather than silently dropped.
        console.warn("dashboard: failed to parse WS message", err);
        return;
      }
      if (event.type === "snapshot") {
        // Already loaded the trace via REST; nothing to do.
        return;
      }
      if (event.type !== "step") return;

      pushPoint(estarChart, 0, event.t, event.e_star);
      const stateVec = Array.isArray(event.state) ? event.state : null;
      const norm = stateVec ? l2(stateVec) : 0;
      const delta = stateVec && prevState ? l2Delta(prevState, stateVec) : null;
      pushPoint(driftChart, 0, event.t, norm);
      pushPoint(driftChart, 1, event.t, delta);
      // Reset prevState to null when the server omits a state vector so
      // the next delta reading is honestly "unknown" rather than a
      // stale comparison against an older step.
      prevState = stateVec ? stateVec.slice() : null;

      estarChart.update("none");
      driftChart.update("none");
      renderSpec(event.spec);
    };
  }

  // -------------------- helpers --------------------
  function pushPoint(chart, datasetIdx, label, value) {
    if (!chart.data.labels.includes(label)) chart.data.labels.push(label);
    chart.data.datasets[datasetIdx].data.push(value);
  }

  function chartOptions(xTitle, yTitle) {
    return {
      animation: false,
      responsive: true,
      plugins: {
        legend: { labels: { color: "#e6edf3" } },
      },
      scales: {
        x: { title: { display: true, text: xTitle, color: "#8b949e" }, ticks: { color: "#8b949e" }, grid: { color: "#30363d" } },
        y: { title: { display: true, text: yTitle, color: "#8b949e" }, ticks: { color: "#8b949e" }, grid: { color: "#30363d" } },
      },
    };
  }

  function l2(vec) {
    if (!Array.isArray(vec)) return 0;
    let s = 0;
    for (let i = 0; i < vec.length; i++) s += Number(vec[i]) * Number(vec[i]);
    return Math.sqrt(s);
  }

  function l2Delta(a, b) {
    const n = Math.min(a.length, b.length);
    let s = 0;
    for (let i = 0; i < n; i++) {
      const d = Number(a[i]) - Number(b[i]);
      s += d * d;
    }
    return Math.sqrt(s);
  }

  function fmt(v, d) {
    if (v === null || v === undefined) return "—";
    return Number(v).toFixed(d == null ? 3 : d);
  }

  function renderSummary(s) {
    if (!s) return;
    document.getElementById("m-steps").textContent = String(s.steps ?? 0);
    document.getElementById("m-latest").textContent = fmt(s.latest_e_star);
    document.getElementById("m-mean").textContent = fmt(s.mean_e_star);
    document.getElementById("m-norm").textContent = fmt(s.final_state_norm);
    document.getElementById("m-drift").textContent = fmt(s.mean_state_delta);
  }

  function renderSpec(spec) {
    // textContent guards against injection from spec contents.
    specEl.textContent = JSON.stringify(spec || {}, null, 2);
  }

  function setLive(on) {
    liveBadge.textContent = on ? "live" : "offline";
    liveBadge.classList.toggle("live", on);
  }
})();
