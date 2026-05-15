// Phase 4 dashboard — landing page. Loads the caller's runs and links
// to per-run dashboards.
(function () {
  "use strict";

  const KEY = "pae_api_key";
  const form = document.getElementById("auth-form");
  const input = document.getElementById("api-key");
  const runsCard = document.getElementById("runs-card");
  const tbody = document.querySelector("#runs-table tbody");
  const empty = document.getElementById("no-runs");

  // Preload from sessionStorage so a reload doesn't ask for the key again.
  const saved = sessionStorage.getItem(KEY);
  if (saved) {
    input.value = saved;
    loadRuns(saved);
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const key = input.value.trim();
    if (!key) return;
    sessionStorage.setItem(KEY, key);
    loadRuns(key);
  });

  function fmt(value, digits) {
    if (value === null || value === undefined) return "—";
    return Number(value).toFixed(digits == null ? 3 : digits);
  }

  async function loadRuns(key) {
    runsCard.hidden = false;
    tbody.innerHTML = "";
    empty.hidden = true;
    try {
      const resp = await fetch("/runs", { headers: { "X-API-Key": key } });
      if (!resp.ok) {
        const detail = await resp.text();
        throw new Error("HTTP " + resp.status + ": " + detail);
      }
      const data = await resp.json();
      const runs = data.runs || [];
      if (runs.length === 0) {
        empty.hidden = false;
        return;
      }
      for (const r of runs) {
        const tr = document.createElement("tr");
        // textContent everywhere — no innerHTML with user data.
        const tdId = document.createElement("td");
        const codeEl = document.createElement("code");
        codeEl.textContent = (r.run_id || "").slice(0, 8);
        tdId.appendChild(codeEl);

        const tdGoal = document.createElement("td");
        tdGoal.textContent = r.brief_goal || "—";

        const tdSteps = document.createElement("td");
        tdSteps.textContent = String(r.steps ?? 0);

        const tdLatest = document.createElement("td");
        tdLatest.textContent = fmt(r.latest_e_star);

        const tdMean = document.createElement("td");
        tdMean.textContent = fmt(r.mean_e_star);

        const tdOpen = document.createElement("td");
        const a = document.createElement("a");
        // URLSearchParams escapes properly so a malicious run_id can't
        // inject markup or open-redirect us anywhere.
        const params = new URLSearchParams({ id: r.run_id });
        a.href = "/dashboard/run.html?" + params.toString();
        a.textContent = "Open →";
        tdOpen.appendChild(a);

        tr.append(tdId, tdGoal, tdSteps, tdLatest, tdMean, tdOpen);
        tbody.appendChild(tr);
      }
    } catch (err) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 6;
      td.textContent = "Failed to load runs: " + err.message;
      tr.appendChild(td);
      tbody.appendChild(tr);
    }
  }
})();
