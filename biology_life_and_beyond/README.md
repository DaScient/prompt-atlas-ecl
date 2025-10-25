# Biology, Life, and Beyond — Interactive Micro‑Exhibit (v1)

This is an **offline‑first** interactive exhibit that turns your essay into a living lab:
- Toy simulations for **ecosystems**, **evolutionary search**, **symbiosis resilience**, and **pandemic mutation risk**.
- A clean FastAPI backend + zero‑dependency front-end (no CDNs) with a tiny custom plotting utility.
- Copy‑safe, non‑diagnostic, and educational.

## Quickstart
```bash
pip install -r requirements.txt
uvicorn biology_app:app --reload --port 8820
# open http://127.0.0.1:8820
```

## Modules
- **Ecosystem (Lotka–Volterra)** — predator–prey with adjustable rates & shocks
- **Evolutionary Design** — a toy GA that “designs” sequences under a simple fitness oracle
- **Symbiosis (Coral Whisperer)** — coral–algae resilience under heat stress
- **Pandemic Watch** — branching‑process toy model for mutation risk

Everything runs locally; no outbound calls.

## Files
```
biology_life_and_beyond/
 ├─ README.md
 ├─ requirements.txt
 ├─ biology_app.py            # FastAPI app
 ├─ simulation_engine.py      # models & RK4 integrator
 ├─ static/
 │  ├─ index.html             # front-end
 │  ├─ css/styles.css
 │  └─ js/app.js             # UI + plotting
 └─ content/
    ├─ essay.md              # Your essay content
    └─ prompts.md           # Prompt slate for exploration
```
