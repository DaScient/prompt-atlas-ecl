#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt Atlas — Hyperspectral Performance (self-running, interactive)
- Band viewer auto-animates through 64 bands
- PCA scatter slow-rotates in 3D
- E★ pulses in the title
- Click any PCA point to reveal its full spectrum (bottom-left)
- Space = play/pause, R = reset

Output: prompt_atlas_performance.html
Deps: numpy, plotly  (pip install numpy plotly)
"""

import os
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


# ---------- Synthetic cube ----------
def make_hyperspectral_cube(H=128, W=128, B=64, K=6, seed=42):
    rng = np.random.default_rng(seed)
    bands = np.linspace(400, 1000, B)

    means = rng.uniform(0.2, 0.9, size=(K, 1))
    widths = rng.uniform(80, 200, size=(K, 1))
    centers = rng.uniform(430, 970, size=(K, 1))
    base_specs = []
    for k in range(K):
        g = means[k] * np.exp(-0.5 * ((bands - centers[k]) / (widths[k] + 1e-9)) ** 2)
        ripple = 0.06 * np.sin(2 * np.pi * (bands - 400) / rng.uniform(140, 260))
        spec = np.clip(g + ripple + rng.normal(0, 0.008, size=bands.shape), 0, 1)
        base_specs.append(spec)
    base_specs = np.stack(base_specs, axis=0)  # [K,B]

    yy, xx = np.mgrid[0:H, 0:W]
    fields = []
    for _ in range(K):
        cx, cy = rng.uniform(0.2, 0.8) * W, rng.uniform(0.2, 0.8) * H
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        f = np.exp(-0.5 * (r / rng.uniform(W * 0.15, W * 0.35)) ** 2)
        fields.append(f)
    fields = np.stack(fields, axis=2) + rng.uniform(0, 0.08, size=(H, W, K))
    weights = fields / (1e-9 + fields.sum(axis=2, keepdims=True))

    cube = np.tensordot(weights, base_specs, axes=(2, 0))
    cube = np.clip(cube + rng.normal(0, 0.008, size=cube.shape), 0, 1).astype(np.float32)
    labels = np.argmax(weights, axis=2).astype(np.int32)
    return cube, labels, bands, base_specs


# ---------- PCA via SVD ----------
def pca_np(X, n_components=3):
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:n_components].T
    scores = Xc @ comps
    var = (S ** 2) / (X.shape[0] - 1)
    var_exp = var[:n_components] / var.sum()
    return comps, scores, var_exp


# ---------- k-means (numpy) ----------
def kmeans_np(X, k=6, iters=25, seed=123):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    centroids = X[rng.choice(N, size=k, replace=False)].copy()
    for _ in range(iters):
        d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        assign = np.argmin(d2, axis=1)
        for j in range(k):
            pts = X[assign == j]
            if len(pts) > 0:
                centroids[j] = pts.mean(axis=0)
    return assign, centroids


# ---------- E★ proxy ----------
def e_star_proxy(X, tau=0.1, seed=7, max_samples=4000):
    rng = np.random.default_rng(seed)
    if X.shape[0] > max_samples:
        X = X[rng.choice(X.shape[0], size=max_samples, replace=False)]
    Xn = X / (1e-12 + np.linalg.norm(X, axis=1, keepdims=True))
    logits = (Xn @ Xn.T) / tau
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True)
    idx = np.arange(X.shape[0])
    ce = -np.log(P[idx, idx] + 1e-12).mean()
    return float(max(0.0, 2.0 - ce))


# ---------- small utils ----------
def stretch(x):
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def smooth(y, w=5):
    if w <= 1:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same")


# ---------- dashboard ----------
def build_dashboard(cube, labels, bands, base_specs, sample_n=9000, seed=0, div_id="atlas_fig"):
    H, W, B = cube.shape
    rng = np.random.default_rng(seed)

    band_images = [(cube[:, :, b] * 255).astype(np.uint8) for b in range(B)]
    presets = {
        "Natural (R=650,G=560,B=470)": (
            int(np.argmin(np.abs(bands - 650))),
            int(np.argmin(np.abs(bands - 560))),
            int(np.argmin(np.abs(bands - 470))),
        ),
        "NIR (R=860,G=650,B=560)": (
            int(np.argmin(np.abs(bands - 860))),
            int(np.argmin(np.abs(bands - 650))),
            int(np.argmin(np.abs(bands - 560))),
        ),
        "SWIR-ish (R=950,G=860,B=650)": (
            int(np.argmin(np.abs(bands - 950))),
            int(np.argmin(np.abs(bands - 860))),
            int(np.argmin(np.abs(bands - 650))),
        ),
    }

    def make_rgb(idx_triplet):
        r, g, b = idx_triplet
        R = stretch(cube[:, :, r])
        G = stretch(cube[:, :, g])
        Bc = stretch(cube[:, :, b])
        return (np.stack([R, G, Bc], axis=2) * 255).astype(np.uint8)

    rgb_images = {name: make_rgb(idx) for name, idx in presets.items()}

    X = cube.reshape(-1, B)
    L = labels.reshape(-1)
    N = X.shape[0]
    sel = rng.choice(N, size=min(N, sample_n), replace=False)
    Xs, Ls = X[sel], L[sel]

    _, scores, var_exp = pca_np(Xs, n_components=3)
    pc1, pc2, pc3 = scores[:, 0], scores[:, 1], scores[:, 2]
    e_star = e_star_proxy(scores)

    Kc = min(7, max(3, len(np.unique(Ls))))
    assign, _ = kmeans_np(scores, k=Kc, iters=20, seed=seed)
    palette = ["#2563eb", "#16a34a", "#dc2626", "#f59e0b", "#7c3aed", "#0891b2", "#475569"]
    colors = [palette[i % len(palette)] for i in assign]

    means_by_cluster = {}
    for cid in np.unique(assign):
        m = Xs[assign == cid].mean(axis=0)
        means_by_cluster[int(cid)] = smooth(m, w=5)

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "image"}, {"type": "scene"}], [{"type": "xy"}, {"type": "domain"}]],
        column_widths=[0.48, 0.52],
        row_heights=[0.50, 0.50],
        subplot_titles=("Band Viewer", "3D PCA (sampled)", "Spectrum (click a point)", "Parallel Coordinates"),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # (A) Band viewer (trace 0)
    fig.add_trace(go.Image(z=band_images[0]), row=1, col=1)
    frames = [go.Frame(data=[go.Image(z=band_images[b])], name=f"band-{b}") for b in range(B)]
    fig.frames = frames

    slider_steps = [
        {
            "args": [[f"band-{b}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": f"{int(bands[b])}nm",
            "method": "animate",
        }
        for b in range(B)
    ]
    band_slider = {"active": 0, "y": 0.915, "x": 0.08, "len": 0.80, "steps": slider_steps}

    anim_buttons = {
        "type": "buttons",
        "showactive": False,
        "x": 0.08,
        "y": 1.045,
        "xanchor": "left",
        "buttons": [
            {"label": "Play", "method": "animate", "args": [None, {"fromcurrent": True, "frame": {"duration": 60}}]},
            {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}}]},
        ],
    }
    fc_buttons = [
        {
            "label": name,
            "method": "update",
            "args": [
                {"z": [rgb]},
                {"title": f"Prompt Atlas — Hyperspectral Performance • {name} • E★={e_star:.3f}"},
            ],
        }
        for name, rgb in rgb_images.items()
    ]
    falsecolor_menu = {"type": "dropdown", "direction": "down", "showactive": True, "x": 0.64, "y": 1.045, "buttons": fc_buttons}
    fig.update_layout(updatemenus=[anim_buttons, falsecolor_menu], sliders=[band_slider])

    # (B) 3D PCA
    fig.add_trace(
        go.Scatter3d(
            x=pc1,
            y=pc2,
            z=pc3,
            mode="markers",
            marker=dict(size=2, color=colors, opacity=0.65),
            hovertemplate="PC1=%{x:.2f}<br>PC2=%{y:.2f}<br>PC3=%{z:.2f}<extra></extra>",
            showlegend=False,
            name="pca_points",
            customdata=np.arange(Xs.shape[0]),
        ),
        row=1,
        col=2,
    )
    fig.update_scenes(
        dict(
            xaxis_title=f"PC1 ({var_exp[0]*100:.1f}%)",
            yaxis_title=f"PC2 ({var_exp[1]*100:.1f}%)",
            zaxis_title=f"PC3 ({var_exp[2]*100:.1f}%)",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            zaxis=dict(showgrid=False),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
        ),
        row=1,
        col=2,
    )

    # (C) Spectrum panel (starts empty; fills on click)
    fig.add_trace(
        go.Scatter(
            x=bands,
            y=[None] * len(bands),
            mode="lines",
            line=dict(width=3),
            name="selected spectrum",
            hovertemplate="λ=%{x:.0f}nm • r=%{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="Wavelength (nm)", row=2, col=1)
    fig.update_yaxes(title_text="Reflectance (a.u.)", row=2, col=1, range=[0, 1])

    # (D) Parallel coordinates — downsampled dims
    pick_idx = np.random.choice(X.shape[0], size=min(200, X.shape[0]), replace=False)
    Xp = X[pick_idx]
    Xpn = (Xp - Xp.min(axis=0, keepdims=True)) / (1e-9 + (Xp.max(axis=0, keepdims=True) - Xp.min(axis=0, keepdims=True)))
    dims_idx = np.linspace(0, B - 1, 16, dtype=int)
    dims = [
        dict(label=f"{int(bands[i])}nm", values=Xpn[:, i], tickvals=[0, 0.5, 1], ticktext=["0", "0.5", "1"])
        for i in dims_idx
    ]
    fig.add_trace(
        go.Parcoords(line=dict(color=np.linspace(0, 1, Xpn.shape[0]), colorscale="Viridis", showscale=False), dimensions=dims),
        row=2,
        col=2,
    )

    fig.update_layout(
        title=f"Prompt Atlas — Hyperspectral Performance • E★={e_star:.3f}",
        template="plotly_white",
        font=dict(family="Inter, Segoe UI, system-ui, sans-serif"),
        height=900,
        margin=dict(l=60, r=60, t=90, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.98, title=None),
    )

    # Data payload for JS (spectra & bands of sampled points)
    payload = {"bands": bands.tolist(), "spectra_sampled": Xs.tolist(), "B": int(B)}

    html = pio.to_html(
        fig,
        full_html=True,
        include_plotlyjs="cdn",
        div_id=div_id,
        post_script=f"""
(function() {{
  const el = document.getElementById("{div_id}");
  const payload = {json.dumps(payload)};
  let t = 0;                 // animation counter
  let playing = true;        // global play state
  let bandIdx = 0;           // current band
  let theta = 0;             // camera angle
  let e0 = 1.0;              // baseline E★

  // Extract initial E★ from title text
  (function initE() {{
    const title = el.layout.title?.text || '';
    const m = title.match(/E★=([0-9\\.]+)/);
    e0 = m ? parseFloat(m[1]) : 1.0;
  }})();

  function setTitleE(val) {{
    const title = el.layout.title?.text || 'Prompt Atlas — Hyperspectral Performance';
    const cleaned = title.replace(/E★=[0-9\\.]+/, '').replace(/\\s+•\\s+$/, '');
    const sep = cleaned.includes('•') ? '' : ' •';
    Plotly.relayout(el, {{'title.text': cleaned + sep + ' • E★=' + val.toFixed(3)}});
  }}

  function animateBand(i) {{
    const frameName = 'band-' + i;
    Plotly.animate(el, [frameName], {{frame: {{duration: 0, redraw: true}}, transition: {{duration: 0}}, mode: 'immediate'}});
  }}

  function rotateCamera(stepDeg) {{
    const step = stepDeg * Math.PI / 180.0;
    theta += step;
    const r = 2.0;
    const eye = {{x: r * Math.cos(theta), y: r * Math.sin(theta), z: 1.2}};
    Plotly.relayout(el, {{'scene.camera.eye': eye}});
  }}

  // Click: PCA → spectrum (trace index 2)
  el.on('plotly_click', function(ev) {{
    const p = ev?.points?.[0];
    if (!p || p.data?.type !== 'scatter3d') return;
    const idx = p.customdata;
    if (idx == null) return;
    const spectrum = payload.spectra_sampled[idx];
    Plotly.restyle(el, {{ x: [payload.bands], y: [spectrum] }}, [2]);
  }});

  // Keyboard controls
  window.addEventListener('keydown', (e) => {{
    if (e.code === 'Space') {{
      playing = !playing;
      e.preventDefault();
    }} else if (e.key === 'r' || e.key === 'R') {{
      bandIdx = 0; theta = 0; t = 0; setTitleE(e0);
      Plotly.relayout(el, {{'scene.camera.eye': {{x:1.6, y:1.6, z:1.2}}}});
      animateBand(bandIdx);
    }}
  }});

  // Main loop
  function tick() {{
    if (playing) {{
      bandIdx = (bandIdx + 1) % payload.B;
      animateBand(bandIdx);
      rotateCamera(0.5);
      const e = e0 + 0.15 * Math.cos(t * 0.05);
      setTitleE(e);
      t++;
    }}
    window.requestAnimationFrame(tick);
  }}
  tick();
}})();
""",
    )
    return html


def main():
    cube, labels, bands, base_specs = make_hyperspectral_cube(H=128, W=128, B=64, K=6, seed=7)
    html = build_dashboard(cube, labels, bands, base_specs, div_id="atlas_fig")
    out = "prompt_atlas_performance.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(
        json.dumps(
            {
                "ok": True,
                "output_html": os.path.abspath(out),
                "shape": list(cube.shape),
                "bands_nm_head": [float(x) for x in bands[:6]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
