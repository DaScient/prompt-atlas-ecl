from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List

# ---------------- Utilities ----------------
def rk4(f, y0, t, args=()):
    y = np.array(y0, dtype=float)
    out = np.zeros((len(t), len(y0)), dtype=float)
    out[0] = y
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        k1 = f(y, t[i-1], *args)
        k2 = f(y + 0.5*dt*k1, t[i-1] + 0.5*dt, *args)
        k3 = f(y + 0.5*dt*k2, t[i-1] + 0.5*dt, *args)
        k4 = f(y + dt*k3, t[i-1] + dt, *args)
        y = y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        out[i] = y
    return out

# ---------------- Models ----------------
@dataclass
class LVParams:
    alpha: float = 1.0   # prey growth
    beta: float = 0.1    # predation rate
    gamma: float = 1.5   # predator decay
    delta: float = 0.075 # predator reproduction per prey
    shock_t: float = -1.0
    shock_scale: float = 1.0

def lv_rhs(y, t, p: LVParams):
    x, ypred = y
    # pulse shock to prey at shock_t
    pulse = 0.0
    if p.shock_t >= 0 and abs(t - p.shock_t) < 0.01:
        pulse = p.shock_scale
    dx = p.alpha * x - p.beta * x * ypred + pulse
    dy = -p.gamma * ypred + p.delta * x * ypred
    return np.array([dx, dy])

def simulate_lv(t_end=50.0, dt=0.05, x0=10.0, y0=5.0, params: Dict = None):
    p = LVParams(**(params or {}))
    t = np.arange(0, t_end+1e-9, dt)
    traj = rk4(lv_rhs, [x0, y0], t, args=(p,))
    return {
        "t": t.tolist(),
        "prey": traj[:,0].tolist(),
        "predator": traj[:,1].tolist(),
        "params": asdict(p)
    }

# Evolutionary toy: sequences over alphabet, fitness prefers target motif & GC content proxy
def evaluate_sequence(seq: str, target: str = "GATTACA"):
    match = sum(1 for a,b in zip(seq, target) if a == b)
    gc = sum(1 for a in seq if a in "GC")
    return match*2 + 0.2*gc

def mutate(seq: str, alphabet="ACGT", rate=0.08, rng=None):
    rng = rng or np.random.default_rng()
    out = list(seq)
    for i in range(len(out)):
        if rng.random() < rate:
            choices = [c for c in alphabet if c != out[i]]
            out[i] = rng.choice(choices)
    return "".join(out)

def evolve_population(pop_size=64, length=12, generations=60, target="GATTACA", seed=42):
    rng = np.random.default_rng(seed)
    pop = ["".join(rng.choice(list("ACGT"), size=length)) for _ in range(pop_size)]
    best_hist = []
    avg_hist = []
    for g in range(generations):
        scores = np.array([evaluate_sequence(s, target) for s in pop])
        best_hist.append(float(scores.max()))
        avg_hist.append(float(scores.mean()))
        # selection: top quartile
        idx = scores.argsort()[::-1][: max(2, pop_size//4)]
        parents = [pop[i] for i in idx]
        # reproduce
        children = []
        while len(children) < pop_size:
            a, b = rng.choice(parents, size=2, replace=True)
            cut = rng.integers(1, length-1)
            child = a[:cut] + b[cut:]
            child = mutate(child, rate=0.08, rng=rng)
            children.append(child)
        pop = children
    return {"best": best_hist, "avg": avg_hist}

# Coral–algae (symbiosis) toy with heat stress factor H(t)
def coral_rhs(y, t, growth=0.8, decay=0.6, symb=0.5, heat_t=30.0, heat_amp=0.7):
    C, A = y  # coral cover, algae symbiont
    H = 1.0 + heat_amp * (1.0 if t >= heat_t else 0.0)  # step heat
    dC = growth*C*(1 - C) + symb*A*C - H*0.5*C
    dA = 0.4*A*(1 - A) + 0.3*C*A - H*0.6*A
    return np.array([dC, dA])

def simulate_coral(t_end=80.0, dt=0.1, C0=0.6, A0=0.6, growth=0.8, decay=0.6, symb=0.5, heat_t=30.0, heat_amp=0.7):
    t = np.arange(0, t_end+1e-9, dt)
    rhs = lambda y, tt: coral_rhs(y, tt, growth, decay, symb, heat_t, heat_amp)
    traj = rk4(rhs, [C0, A0], t)
    return {"t": t.tolist(), "coral": traj[:,0].tolist(), "algae": traj[:,1].tolist(),
            "params": {"growth":growth,"decay":decay,"symb":symb,"heat_t":heat_t,"heat_amp":heat_amp}}

# Pandemic mutation branching toy
def simulate_pandemic(t_end=60, R0=1.2, mut_prob=0.02, variant_boost=0.3, seed=7):
    rng = np.random.default_rng(seed)
    I = 10.0
    t = list(range(t_end+1))
    base, variant = [], []
    has_variant = False
    for day in t:
        base.append(float(I))
        if has_variant:
            variant.append(float(0.2*I))
        else:
            variant.append(0.0)
        # mutation chance
        if not has_variant and rng.random() < mut_prob*min(1.0, I/1000.0):
            has_variant = True
        growth = R0 + (variant_boost if has_variant else 0.0) - 1.0
        I = max(0.0, I*(1.0 + growth))
    return {"t": t, "base": base, "variant": variant, "params": {"R0":R0,"mut_prob":mut_prob,"variant_boost":variant_boost}}
