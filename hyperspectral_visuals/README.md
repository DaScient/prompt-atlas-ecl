# Hyperspectral Visuals — The Prompt Atlas Performance Suite

The Hyperspectral Visuals module forms the visual-analytic core of the Prompt Atlas Engine, integrating high-dimensional spectral synthesis, manifold learning, and interactive visualization into a unified computational narrative.
Each script within this folder generates a reproducible, publication-grade figure or performance artifact, designed for both exploratory analysis and demonstrative communication of emergent spectral phenomena.



## 1 · Scientific Context

Hyperspectral data encode reflectance as a continuous function of wavelength, yielding hundreds of correlated measurements per pixel.
This subdirectory implements computational performances—interactive visualizations that render these spectra as dynamic systems rather than static plots.
They serve to expose latent geometric and informational structure in synthetic or empirical data cubes, emphasizing entropy–coherence interplay (E★) as an interpretable proxy for representational stability within high-dimensional manifolds.



## 2 · Core Modules

Script	Description
hyperspectral_atlas_performance.py	Generates a fully self-contained HTML visualization illustrating dynamic band traversal, 3D PCA rotation, and E★ modulation. The figure auto-animates, enabling cinematic yet quantitative exploration of hyperspectral embeddings.
(Future additions)	hyperspectral_atlas_live.py — real-time stream visualizer.   atlas_manifold_theatre.py — interactive manifold navigator.   spectral_embeddings_vr.py — stereo-immersive rendering for VR environments.




## 3 · Analytic Components
	1.	Synthetic Cube Generator — Produces a 64-band cube with Gaussian and sinusoidal spectral bases; includes realistic intra-scene variation.
	2.	Dimensional Reduction — Applies PCA via singular-value decomposition to expose principal axes of spectral variance.
	3.	Cluster Mapping — Executes lightweight k-means segmentation for visual stratification of latent clusters.
	4.	Entropy Proxy (E★) — Quantifies global self-similarity; visualized as a breathing scalar in the title.
	5.	Visualization Layer — Composes multi-panel dashboards (spectral image, 3D PCA, parallel coordinates, and spectrum viewer) in Plotly with JavaScript-driven interactivity.



## 4 · Execution & Usage

cd hyperspectral_visuals
pip install numpy plotly
python hyperspectral_atlas_performance.py

The script outputs:

prompt_atlas_performance.html

Opening this file in any modern browser launches a fully offline, interactive experiment.



## 5 · Interactive Controls

Function	Key / Interaction	Description
Play / Pause	Spacebar	Toggle the hyperspectral sequence animation
Reset	R	Reinitialize band index and camera orientation
Inspect Spectrum	Mouse click on a PCA point	Reveal full reflectance profile for that observation
Color Modes	Dropdown menu	Switch between Natural, NIR, and SWIR composites
Auto-Orbit	Continuous	Maintains a slow rotation of the 3D PCA manifold




## 6 · Scientific Significance

This visualization suite demonstrates how high-dimensional spectral data can be re-expressed as performative models of coherence.
By coupling manifold learning with interactive visualization, it offers a compact window into:
	•	The topology of spectral embeddings,
	•	The entropy–information balance of learned representations, and
	•	The potential for real-time, human-interpretable AI diagnostics.

Each performance thus functions as both scientific instrument and aesthetic experiment—a methodological bridge between quantitative modeling and visual cognition.



## 7 · Reproducibility & Licensing

All scripts are deterministic with fixed random seeds and require only NumPy ≥ 1.24 and Plotly ≥ 5.15.
Outputs are self-contained HTML files; no external dependencies or network calls are performed.

License: Creative Commons Attribution–NonCommercial 4.0 International
© 2025 DaScient | ArchCore | Prompt Atlas Collective



## 8 · Citation

If this work contributes to your research or presentation, please cite:

Don D.M. Tadaya (2025). Prompt Atlas — Hyperspectral Visuals: Interactive Performance Framework for Entropic Manifold Analysis.
DaScient / ArchCore Research Laboratory.



Would you like me to generate a companion banner image (e.g., a minimalist spectral gradient with the Prompt Atlas wordmark) for the top of this README to make it publication-ready on GitHub?
