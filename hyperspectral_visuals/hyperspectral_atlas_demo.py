#!/usr/bin/env python3
"""
Prompt Atlas — Hyperspectral Multidimensional Interactive Demo
Generates an interactive HTML explorer showcasing:
- Hyperspectral band slider (128x128x64 synthetic cube)
- False-color composites
- 3D PCA of pixel spectra
- Cluster-mean spectra + parallel coordinates
- E★ (InfoNCE-like coherence proxy) stamp

Output: prompt_atlas_hyperspectral_explorer.html
"""

import os
import math
import json
import random
import numpy as np

# Optional libs
try:
    from sklearn.manifold import TSNE  # noqa: F401
    HAVE_SK = True
except Exception:
    HAVE_SK = False

try:
    import umap  # noqa: F401
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------
# 1) Synthetic hyperspectral cube
# -------------------------
def make_hyperspectral_cube(H=128, W=128, B=64, K=5, seed=42):
    """
    Create a synthetic hyperspectral cube with K endmembers + smooth spatial fields.
    Returns:
        cube: [H, W, B] float32
        labels: [H, W] int cluster id
        bands: [B] wavelengths (arb units)
    """
    rng = np.random.default_rng(seed)
    bands = np.linspace(400, 1000, B)  # nm, arbitrary

    # K endmembers with smooth spectra
    means = rng.uniform(0.2, 0.9, size=(K, 1))
    widths = rng.uniform(80, 200, size=(K, 1))
    centers = rng.uniform(430, 970, size=(K, 1))
    base_specs = []
    for k in range(K):
        # gaussian-ish bump + sinusoidal ripple
        g = means[k] * np.exp(-0.5 * ((bands - centers[k]) / widths[k]) ** 2)
        ripple = 0.08 * np.sin(2 * np.pi * (bands - 400) / rng.uniform(140, 260))
        spec = np.clip(g + ripple + rng.normal(0, 0.01, size=bands.shape), 0, 1)
        base_specs.append(spec)
    base_specs = np.stack(base_specs, axis=0)  # [K,B]

    # Smooth spatial fields to mix endmembers
    yy, xx = np.mgrid[0:H, 0:W]
    fields = []
    for k in range(K):
        cx, cy = rng.uniform(0.2, 0.8) * W, rng.uniform(0.2, 0.8) * H
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        f = np.exp(-0.5 * (r / rng.uniform(W * 0.15, W * 0.35)) ** 2)
        fields.append(f)
    fields = np.stack(fields, axis=2)  # [H,W,K]
    fields = fields + rng.uniform(0, 0.1, size=fields.shape)
    weights = fields / (1e-6 + fields.sum(axis=2, keepdims=True))

    # Mix spectra per pixel
    cube = np.tensordot(weights, base_specs, axes=(2, 0))  # [H,W,B]
    cube = np.clip(cube + rng.normal(0, 0.01, size=cube.shape), 0, 1).astype(np.float32)

    # Cluster labels (argmax of weights = dominant endmember)
    labels = np.argmax(weights, axis=2).astype(np.int32)
    return cube, labels, bands, base_specs


# -------------------------
# 2) PCA (numpy SVD)
# -------------------------
def pca_np(X, n_components=3):
    """
    X: [N,D] row-wise samples
    Returns: comps [D,nc], scores [N,nc], var_exp [nc]
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:n_components].T                    # [D,nc]
    scores = Xc @ comps                            # [N,nc]
    var = (S ** 2) / (X.shape[0] - 1)
    var_exp = var[:n_components] / var.sum()
    return comps, scores, var_exp


# -------------------------
# 3) Simple k-means (numpy), to avoid sklearn dependency
# -------------------------
def kmeans_np(X, k=6, iters=25, seed=123):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    idx = rng.choice(N, size=k, replace=False)
    centroids = X[idx].copy()
    for _ in range(iters):
        d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        assign = np.argmin(d2, axis=1)
        for j in range(k):
            pts = X[assign == j]
            if len(pts) > 0:
                centroids[j] = pts.mean(axis=0)
    return assign, centroids


# -------------------------
# 4) E★ (coherence proxy): invert an InfoNCE-like score on random projections
# -------------------------
def e_star_proxy(X, tau=0.1, seed=7, max_samples=4000):
    """
    Cheap proxy: normalize features, take dot products as logits, cross-entropy vs identity.
    """
    rng = np.random.default_rng(seed)
    if X.shape[0] > max_samples:
        sel = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[sel]
    # normalize
    Xn = X / (1e-9 + np.linalg.norm(X, axis=1, keepdims=True))
    logits = (Xn @ Xn.T) / tau
    # stabilize
    logits = logits - logits.max(axis=1, keepdims=True)
    # softmax
    exp = np.exp(logits)
    P = exp / exp.sum(axis=1, keepdims=True)
    # cross-entropy to identity matching
    y = np.arange(X.shape[0])
    ce = -np.log(P[np.arange(X.shape[0]), y] + 1e-12).mean()
    # invert and squash
    e_star = float(max(0.0, 2.0 - ce))
    return e_star


# -------------------------
# 5) Build interactive Plotly dashboard
# -------------------------
def build_dashboard(cube, labels, bands, base_specs, sample_n=12000, seed=0, out_html="prompt_atlas_hyperspectral_explorer.html"):
    H, W, B = cube.shape
    rng = np.random.default_rng(seed)

    # Band slider frames (downscale for speed if needed)
    band_images = [(cube[:, :, b] * 255).astype(np.uint8) for b in range(B)]

    # False-color presets (R,G,B band indices)
    presets = {
        "Natural (R=650,G=560,B=470)": (np.argmin(np.abs(bands - 650)),
                                        np.argmin(np.abs(bands - 560)),
                                        np.argmin(np.abs(bands - 470))),
        "NIR (R=860,G=650,B=560)":     (np.argmin(np.abs(bands - 860)),
                                        np.argmin(np.abs(bands - 650)),
                                        np.argmin(np.abs(bands - 560))),
        "SWIR-ish (R=950,G=860,B=650)":(np.argmin(np.abs(bands - 950)),
                                        np.argmin(np.abs(bands - 860)),
                                        np.argmin(np.abs(bands - 650))),
    }

    def make_rgb(idx_triplet):
        r, g, b = idx_triplet
        # normalize each channel to [0,1] with percentile stretch
        def stretch(x):
            lo, hi = np.percentile(x, 2), np.percentile(x, 98)
            x = np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)
            return x
        R = stretch(cube[:, :, r])
        G = stretch(cube[:, :, g])
        Bc = stretch(cube[:, :, b])
        rgb = np.stack([R, G, Bc], axis=2)
        return (rgb * 255).astype(np.uint8)

    rgb_images = {name: make_rgb(idx) for name, idx in presets.items()}

    # Flatten for embeddings
    X = cube.reshape(-1, B)  # [N,B]
    L = labels.reshape(-1)

    # Sample for scatter
    N = X.shape[0]
    if N > sample_n:
        sel = rng.choice(N, size=sample_n, replace=False)
    else:
        sel = np.arange(N)
    Xs = X[sel]
    Ls = L[sel]

    # PCA → 3D
    _, scores, var_exp = pca_np(Xs, n_components=3)
    pc1, pc2, pc3 = scores[:, 0], scores[:, 1], scores[:, 2]
    e_star = e_star_proxy(scores)

    # K-means on PCA scores (simple clusters for colors)
    assign, _ = kmeans_np(scores, k=min(8, max(3, len(np.unique(Ls)))), iters=20, seed=seed)
    palette = [
        "#3b82f6", "#10b981", "#ef4444", "#f59e0b", "#8b5cf6", "#06b6d4", "#64748b", "#22c55e"
    ]
    colors = [palette[i % len(palette)] for i in assign]

    # Cluster-mean spectra (on the sampled set)
    means_by_cluster = {}
    for cid in np.unique(assign):
        means_by_cluster[int(cid)] = Xs[assign == cid].mean(axis=0)

    # ------------------ FIGURE ------------------
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "image"}, {"type": "scene"}],
               [{"type": "xy"}, {"type": "domain"}]],
        subplot_titles=(
            "Band Viewer (slider)",
            "3D PCA of Spectra (sampled)",
            "Cluster-Mean Spectra",
            "Parallel Coordinates (random picks)"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )

    # (A) Band viewer with slider (start at band 0)
    fig.add_trace(go.Image(z=band_images[0]), row=1, col=1)

    # Build frames for each band
    frames = [go.Frame(data=[go.Image(z=band_images[b])], name=f"band-{b}") for b in range(len(bands))]
    fig.frames = frames  # <-- put frames on the figure, not in layout

    # Slider for bands
    slider_steps = [{
        "args": [[f"band-{b}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
        "label": f"{int(bands[b])}nm",
        "method": "animate"
    } for b in range(len(bands))]

    band_slider = {
        "active": 0,
        "y": 0.93,
        "x": 0.08,
        "len": 0.8,
        "steps": slider_steps
    }

    # Play/Pause buttons (animation controls)
    anim_buttons = {
        "type": "buttons",
        "showactive": False,
        "x": 0.12, "y": 1.12,
        "xanchor": "left",
        "buttons": [
            {"label": "Play", "method": "animate", "args": [None, {"fromcurrent": True, "frame": {"duration": 60}}]},
            {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}}]}
        ]
    }

    # False-color dropdown (updates the first image trace)
    fc_buttons = []
    for name, rgb in rgb_images.items():
        fc_buttons.append({
            "label": name,
            "method": "update",
            "args": [
                {"z": [rgb]},   # update trace 0's image array
                {"title": f"Prompt Atlas — Hyperspectral Explorer • {name}"}  # optional
            ]
        })
    falsecolor_menu = {
        "type": "dropdown",
        "direction": "down",
        "showactive": True,
        "x": 0.75, "y": 1.12,
        "buttons": fc_buttons
    }

    # Apply menus + slider to layout (NO frames here)
    fig.update_layout(
        updatemenus=[anim_buttons, falsecolor_menu],
        sliders=[band_slider]
    )

    # (B) 3D PCA scatter
    fig.add_trace(
        go.Scatter3d(
            x=pc1, y=pc2, z=pc3,
            mode="markers",
            marker=dict(size=2.2, color=colors, opacity=0.7),
            hovertext=[f"cluster {int(c)}" for c in assign],
            hoverinfo="text"
        ),
        row=1, col=2
    )
    fig.update_scenes(
        dict(
            xaxis_title=f"PC1 ({var_exp[0]*100:.1f}%)",
            yaxis_title=f"PC2 ({var_exp[1]*100:.1f}%)",
            zaxis_title=f"PC3 ({var_exp[2]*100:.1f}%)",
        ), row=1, col=2
    )

    # (C) Cluster-mean spectra
    for cid, mean_spec in means_by_cluster.items():
        fig.add_trace(
            go.Scatter(
                x=bands, y=mean_spec, mode="lines", name=f"cluster {cid}",
                line=dict(width=2)
            ),
            row=2, col=1
        )
    fig.update_xaxes(title_text="Wavelength (nm)", row=2, col=1)
    fig.update_yaxes(title_text="Reflectance (a.u.)", row=2, col=1)

    # (D) Parallel coordinates of random spectra
    pick_idx = np.random.choice(X.shape[0], size=min(200, X.shape[0]), replace=False)
    Xp = X[pick_idx]
    # Normalize for parallel coords display
    Xpn = (Xp - Xp.min(axis=0, keepdims=True)) / (1e-9 + (Xp.max(axis=0, keepdims=True) - Xp.min(axis=0, keepdims=True)))
    dims = [dict(label=f"{int(wl)}nm", values=Xpn[:, i]) for i, wl in enumerate(bands)]
    par = go.Parcoords(
        line=dict(color=np.linspace(0, 1, Xpn.shape[0]), colorscale="Viridis"),
        dimensions=dims
    )
    fig.add_trace(par, row=2, col=2)

    # Title + E★ stamp
    fig.update_layout(
        title=f"Prompt Atlas — Hyperspectral Explorer • E★={e_star:.3f}",
        height=1000,
        template="plotly_white",
        margin=dict(l=60, r=60, t=90, b=60)
    )

    # Save to HTML
    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)
    return out_html, float(e_star)


def main():
    cube, labels, bands, base_specs = make_hyperspectral_cube(H=128, W=128, B=64, K=6, seed=7)
    out, e_star = build_dashboard(cube, labels, bands, base_specs)
    print(json.dumps({
        "ok": True,
        "output_html": os.path.abspath(out),
        "shape": list(cube.shape),
        "bands_nm": [float(b) for b in bands[:5]] + ["..."],
        "E_star": round(e_star, 4)
    }, indent=2))

if __name__ == "__main__":
    main()
