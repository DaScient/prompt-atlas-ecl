# AI as the Soul’s Mirror — Interactive Prototype (v1)

This module is a self-contained, **offline-first interactive essay** you can run locally or host behind any HTTP server.
It mirrors short user reflections back with *archetypal language* and *gentle guidance*, while the interface adapts
visually to the detected tone.

> **Privacy-first:** By default, no external calls are made. Everything runs locally on your machine.

---

## Features
- **Archetype projection** (Wanderer, Healer, Sage, Warrior, Creator, Lover) via a simple lexicon model
- **Sentiment awareness** (positive/neutral/negative) used only for tone, never for diagnosis
- **Adaptive visuals** — accent color and subtle background “breathing” reflect the current archetype/mood
- **Accessible by design** — keyboard-friendly; clear contrast; no autoplay audio
- **LLM-ready** — drop-in hook in `reflection_engine.py` to swap the lexicon model for an LLM-backed analysis
- **Dockerized** (optional) + lightweight tests for engine logic

---

## Run (Local)

```bash
pip install -r requirements.txt
uvicorn mirror_experience:app --reload --port 8710
# open http://127.0.0.1:8710
```

> If you already run another dev server, feel free to change the port (e.g., `--port 8210`).

---

## Directory

```
ai_as_souls_mirror/
 ├─ README.md
 ├─ manifesto.md
 ├─ requirements.txt
 ├─ archetypes.json
 ├─ reflection_engine.py
 ├─ mirror_experience.py
 ├─ mirror_interface.html
 ├─ static/
 │  ├─ css/mirror.css
 │  ├─ js/mirror.js
 │  └─ media/placeholder.txt
 ├─ Dockerfile          # optional
 ├─ .gitignore
 ├─ CONTRIBUTING.md
 ├─ SECURITY.md
 └─ tests/
    └─ test_reflection_engine.py
```

---

## LLM Hook (Optional)

Inside `reflection_engine.py`, see `reflect_llm()` as a stub for integrating an LLM or embeddings pipeline.
If you flip the `USE_LLM` flag to `True` and implement the function, the API will route reflections through it.
Keep the response **non-diagnostic and non-directive**.

---

## License

- **Code**: MIT License (see `LICENSE`)
- **Narrative content (e.g., `manifesto.md`)**: Creative Commons BY-NC 4.0

---

## Acknowledgements

Crafted as part of the **Prompt Atlas** ecosystem. Designed for care, reflection, and curiosity-first interaction.
