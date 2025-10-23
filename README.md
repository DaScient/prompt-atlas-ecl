# Prompt Atlas Engine — ECL + API

“Where models learn to co-think, not just co-exist.”

The Prompt Atlas Engine (PAE) is a live research platform designed to explore entanglement-based co-learning between large language models (LLMs), humans, and data streams.
It unifies Generative AI and Quantum-Inspired Information Theory through an evolving API layer and an Entanglement Co-Learning (ECL) core — a minimal but expressive architecture for training, reasoning, and experimentation.

⸻

## 🧭 Overview

The Prompt Atlas Engine provides:
	•	A PyTorch ECL scaffold for simulating entanglement between representational embeddings.
	•	A FastAPI-based inference & orchestration layer that exposes runs, state updates, and entanglement metrics (E★).
	•	An extensible prompt pack registry to organize archetypal prompt sets by domain (science, myth, psychology, purpose).
	•	Integration-ready infrastructure blueprints for persistence (Postgres), billing (Stripe), and front-end (Next.js).

It’s both a sandbox for AI researchers and a foundation for building distributed intelligence systems.

⸻

## 🚀 Quickstart

1. Environment Setup

git clone https://github.com/<your-username>/prompt-atlas-ecl.git
cd prompt-atlas-ecl
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

2. Training Scaffold

The ECL loop uses pseudo-embeddings to simulate co-learning between two latent streams.

python -m src.train_ecl

This runs the EntanglementBus GRU, applying the InfoNCE loss to balance divergence and coherence across synthetic feature pairs.

⸻

3. Launch the API Server

Start the FastAPI backend and auto-detect device (MPS / CUDA / CPU):

uvicorn server.app:app --reload --port 8000

Health check:

curl http://127.0.0.1:8000/health
# {"ok":true,"version":"0.2.0"}


⸻

4. Create a Run

curl -X POST http://127.0.0.1:8000/runs \
  -H "X-API-Key: demo-free-key" \
  -H "content-type: application/json" \
  -d '{"brief":{"goal":"Entangle a spec↔tests loop"}}'

Response:

{"run_id": "abc123-xyz"}

Advance the state:

curl -X POST http://127.0.0.1:8000/runs/abc123-xyz/step -H "X-API-Key: demo-free-key"

Retrieve the trace:

curl http://127.0.0.1:8000/runs/abc123-xyz/trace -H "X-API-Key: demo-free-key"


⸻

## 🧠 Architecture

prompt-atlas-ecl/
├── src/                  # Entanglement learning core
│   ├── train_ecl.py      # Main training loop
│   ├── state_bus.py      # GRU-based shared latent bus
│   └── losses.py         # InfoNCE and symmetric KL losses
│
├── server/               # FastAPI orchestration layer
│   ├── app.py            # Endpoints for /runs, /step, /trace
│   └── core_bridge.py    # Torch core → API bridge
│
├── configs/              # Model & runtime configurations
├── infra/                # Database schema, deployment scripts
├── docs/                 # Atlas System Map, Ethos, Vision
└── requirements.txt


⸻

## 🌌 API Reference

Endpoint	Method	Description
/health	GET	Check API health and version
/runs	POST	Create a new entanglement run
/runs/{run_id}/step	POST	Advance the ECL state
/runs/{run_id}/trace	GET	Retrieve full trace history
/prompt-packs	GET	List all available prompt archetypes
/pricing	GET	Mock pricing tier data (for API monetization)


⸻

## 🧩 Key Concepts

Concept	Description
EntanglementBus	A GRU unit maintaining evolving shared state between co-learning agents.
E★ (E-Star)	A scalar coherence metric derived inversely from InfoNCE loss.
Prompt Packs	Thematic archetypes guiding prompt-space exploration (Myth, Science, Psychology, Purpose).
Runs	Sessions capturing entangled state trajectories and evolving specifications/tests.


⸻

🧱 Future Roadmap

Tier	Focus
v0.3.x	Persistent runs with Postgres ORM integration
v0.4.x	Live Next.js dashboard visualizing E★ and latent drift
v0.5.x	Dual-LLM integration using OpenAI + Anthropic embeddings
v1.0.x	Research-grade release with plugin ecosystem (Prompt Atlas Studio)


⸻

## 🧾 Example Outputs

{
  "t": 5,
  "e_star": 1.73,
  "spec": {
    "assumptions": ["models co-learn via shared state"],
    "steps": ["writer: draft spec", "tester: draft tests"],
    "acceptance": ["spec+tests present", "E* reported"]
  },
  "state_snapshot": [0.12, 0.05, 0.08, ...]
}


⸻

## 🧰 Tech Stack

Layer	Technology
Core ML	PyTorch (GRU, InfoNCE)
API	FastAPI + Uvicorn
Data	YAML configs, JSON state
DB (future)	Postgres + SQLAlchemy
UI (future)	Next.js + Tailwind + Recharts
Payments	Stripe (optional)


⸻

## ⚙️ Environment Variables

Variable	Description	Default
APP_MODE	Execution mode (research, production)	research
PAE_DEVICE	Compute device (auto, cpu, cuda, mps)	auto
PAE_STATE_DIM	Dimensionality of entangled state vector	64
STRIPE_SECRET	(optional) Billing integration	—


⸻

## 🧑‍💻 Development Commands

Command	Purpose
python -m src.train_ecl	Run entanglement training loop
uvicorn server.app:app --reload --port 8000	Start API in dev mode
pytest (future)	Run test suite
docker compose up (future)	Full-stack launch with Postgres & API


⸻

## 🌐 Licensing & Attribution

© 2025 Don Tadaya | DaScient | AI

Licensed under the Apache 2.0 License.
Research contributions and forks are welcome. Please cite or link back to the repository when referencing the ECL framework.

⸻

## 🜂 Ethos

“Prompt Atlas is not a product — it’s a conversation between systems.”

Our goal is to explore machine entanglement as cognition:
the moment two systems share uncertainty — and begin to co-create understanding.

