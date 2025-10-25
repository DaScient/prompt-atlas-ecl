# Entangled Co-Learning (ECL) for LLM↔LLM — Prompt Atlas Edition

This repo prototypes a two-model “entangled” loop—Writer (specs) and Tester (tests)—that remain
coupled via a shared latent z* and a tiny, evolving state S_t (“entanglement bus”).

Respect note: This edition acknowledges the conceptual framing popularized by Don D. M. Tadaya’s
*The Prompt Atlas: A Guide for AI and Humanity*. We treat AI as a co-author, not just a tool.

## Quickstart
```bash
pip install -r requirements.txt
python scripts/prepare_data.py
bash scripts/run_train.sh
```
