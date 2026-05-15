// Phase 6 studio — Community Prompt Registry browser.
//
// Read-only and unauthenticated; the /prompt-packs endpoints are
// public so the studio doesn't need an API key.
(function () {
  "use strict";

  const form = document.getElementById("search-form");
  const qInput = document.getElementById("q");
  const domainInput = document.getElementById("domain");
  const tbody = document.querySelector("#packs-table tbody");
  const empty = document.getElementById("no-packs");
  const detailCard = document.getElementById("pack-detail-card");
  const detailTitle = document.getElementById("pack-detail-title");
  const detailMeta = document.getElementById("pack-detail-meta");
  const detailDesc = document.getElementById("pack-detail-description");
  const detailTpls = document.getElementById("pack-detail-templates");

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    loadPacks();
  });

  function escapeHtml(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  async function loadPacks() {
    const params = new URLSearchParams();
    const q = qInput.value.trim();
    const dom = domainInput.value.trim();
    if (q) params.set("q", q);
    if (dom) params.set("domain", dom);

    const res = await fetch(`/prompt-packs?${params.toString()}`);
    if (!res.ok) {
      tbody.innerHTML = "";
      empty.hidden = false;
      empty.textContent = `Failed to load packs (HTTP ${res.status}).`;
      return;
    }
    const body = await res.json();
    const packs = body.packs || [];
    tbody.innerHTML = "";
    if (!packs.length) {
      empty.hidden = false;
      empty.textContent = "No matching packs.";
      return;
    }
    empty.hidden = true;
    for (const p of packs) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><a href="#" data-pack="${escapeHtml(p.id)}"><code>${escapeHtml(p.id)}</code></a></td>
        <td>${escapeHtml(p.title)}</td>
        <td>${escapeHtml(p.domain)}</td>
        <td>${(p.tags || []).map(escapeHtml).join(", ")}</td>
        <td>${Number(p.prompt_count) || 0}</td>
      `;
      tr.querySelector("a").addEventListener("click", (e) => {
        e.preventDefault();
        loadPackDetail(p.id);
      });
      tbody.appendChild(tr);
    }
  }

  async function loadPackDetail(id) {
    const res = await fetch(`/prompt-packs/${encodeURIComponent(id)}`);
    if (!res.ok) {
      detailCard.hidden = false;
      detailTitle.textContent = "Not found";
      detailDesc.textContent = `HTTP ${res.status}`;
      detailTpls.innerHTML = "";
      return;
    }
    const pack = await res.json();
    detailCard.hidden = false;
    detailTitle.textContent = `${pack.title} (${pack.id} @ ${pack.version})`;
    detailMeta.textContent = `domain: ${pack.domain} · author: ${pack.author} · license: ${pack.license || "—"} · tags: ${(pack.tags || []).join(", ")}`;
    detailDesc.textContent = pack.description || "";
    detailTpls.innerHTML = "";
    for (const tmpl of pack.prompts || []) {
      const wrap = document.createElement("div");
      wrap.className = "template";
      const inputs = (tmpl.inputs || []).join(", ");
      wrap.innerHTML = `
        <h4><code>${escapeHtml(tmpl.name)}</code> <small class="muted">[${escapeHtml(tmpl.role || "user")}]</small></h4>
        <p class="muted">${escapeHtml(tmpl.description || "")}${inputs ? ` · inputs: ${escapeHtml(inputs)}` : ""}</p>
        <pre></pre>
      `;
      // Use textContent on the <pre> so the template body is rendered
      // verbatim and never interpreted as HTML.
      wrap.querySelector("pre").textContent = tmpl.body || "";
      detailTpls.appendChild(wrap);
    }
  }

  // First load.
  loadPacks();
})();
