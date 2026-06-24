# SpecLens-edu — interpreting a small CNN with SAEs (CIFAR-100)

A slim, education-only slice of [SpecLens](https://github.com/) for a ~50-minute Colab tutorial.
It keeps the **core machinery** (SAE training via the shared-activation-buffer pipeline, feature
indexing, FRI attribution) on one tiny model, and drops the research code and other model packs.

## Run it
1. Open **`cifar_speclens_tutorial.ipynb`** in Google Colab (Runtime → GPU / T4).
2. In the first cell, set `GH_USER` to the GitHub account hosting this branch + the artifact Release.
3. Run all cells (~10 min of compute; the rest is reading).

The notebook clones this slim branch for code and `wget`s a small precomputed-artifact bundle
(`cifar_tutorial_artifacts.tar.gz`, ~50 MB) from the repo's GitHub **Release**, then runs five sections:

1. **CNN + SAEs** — load the 71%-accuracy CNN; look at what a layer-4 SAE feature detects.
   (SAEs are precomputed; *optionally* retrain all layers live with `train_sae_config.py` — the
   shared activation buffer trains every layer's SAE in one pass.)
2. **Mechanistic tree** — top-down, FRI-attributed feature→feature tree (precomputed interactive HTML).
3. **Debugging** — name the culprit feature behind a misclassification (exact linear attribution).
4. **Why-confused** — shared vs discriminative features for a confused class pair, and in which layer.
5. **Spurious shortcut** — a planted patch shortcut, found by the SAE and fixed by cleaning the data.

## What's included
- `src/core/` — SAE training/store, indexing, FRI attribution, runtime capture (full core).
- `src/packs/cifar_cnn` (+ `resnet` for the reused adapter) — the tiny CIFAR CNN pack.
- `scripts/` — `train_sae_config.py`, `sae_index_main.py`, `cifar_fri_feature.py`, and the tutorial
  analysis scripts.
- `configs/` — `cifar_cnn_sae.yaml`, `cifar_cnn_index.yaml`, `cifar_cnn_sae_colab.yaml`.

Honest theme: interpretability is great for **diagnosis** and **fixing real bugs**; it is not a magic
accuracy button on already-clean data.
