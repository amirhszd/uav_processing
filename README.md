# uav_processing

## Overview

This codebase implements a **UAV (drone) hyperspectral image processing pipeline**. Its overall objective is to take raw hyperspectral imagery (HSI) captured by a UAV, co-register it with multispectral imagery (MSI) for spatial alignment, calibrate the radiance data into physically meaningful surface reflectance values, and finally stitch individual flight-line image strips into a single seamless mosaic — producing analysis-ready hyperspectral maps of a scene (e.g., agricultural fields).

---

## Three-Step Pipeline

**Step 1 — `1_coregister_hsi_msi.py`: Co-registration**
This script spatially aligns the hyperspectral image (HSI) strips with a reference multispectral image (MSI). It extracts RGB composites from the HSI data cube, detects matching keypoints between the two images using feature-based methods (via OpenCV), and applies a geometric warp to bring them into the same coordinate space.

**Step 2 — `2_calibrate_to_reflectance_globalelm.py` / `2_calibrate_to_reflectance_localelm.py`: Radiometric Calibration**
These scripts convert the co-registered radiance imagery into surface reflectance. They use empirical line method (ELM) calibration based on known reflectance panels (light gray, medium gray, dark gray, and black) captured in the scene. The "global" variant fits a single calibration model across all flight lines, while the "local" variant fits it per-strip.

**Step 3 — `3_mosaic_raster.py`: Mosaicking**
This script merges all individually calibrated, co-registered HSI strips into a single continuous raster mosaic. It uses GDAL to build a virtual raster (VRT) from all ENVI-format inputs and translates it into a single ENVI binary output, preserving hyperspectral metadata (wavelengths, FWHM, band names, etc.).

---

## Installation

The environment can be set up using **Conda** with the provided YAML files:

- **`environment_anyos.yml`** — cross-platform environment (Windows/macOS/Linux)
- **`environment_linux.yml`** — Linux-specific environment

To install, run:

```bash
conda env create -f environment_anyos.yml
conda activate uav_processing311
```

The environment (named `uav_processing311`) includes all required dependencies: GDAL, Rasterio, NumPy, OpenCV, scikit-image, spectral, and many others — all pinned to exact versions for reproducibility.
