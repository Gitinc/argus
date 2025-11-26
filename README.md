# ARGUS

**ARGUS** — inspired by *Argus Panoptes* (Ancient Greek: Ἄργος Πανόπτης, “All-seeing Argos”), the many-eyed giant of Greek mythology — is a robust, general, and computationally efficient framework for cell tracking in time-resolved microscopy data.

The name **ARGUS** stands for:

**A** **R**obust and **G**eneral comp**U**tationally efficient framework for cell tracking in time-re**S**olved microscopy

---

## Overview

ARGUS provides a complete pipeline for *detecting*, *tracking*, *analyzing*, and *visualizing* cells in microscopy image sequences.  
It is designed to be:

- **Robust** — works across diverse imaging conditions and cell morphologies  
- **Generalizable** — minimal parameter tuning required across datasets  
- **Computationally efficient** — optimized for long time-lapse sequences and high-throughput workflows  
- **Modular** — each component (preprocessing, detection, tracking, analysis, visualization) can be used independently  

ARGUS integrates classical image processing, optical flow methods, and modern tracking algorithms.  
A set of Jupyter notebooks is included to help users explore, reproduce, and evaluate results interactively.

---

## Features

### Preprocessing
- Multiple optional denoising methods (e.g., wavelet, bm3d, monogenic filtering)
- Global brightness/thrift correction tools  

### Detection
- Local maxima–based detection  
- Centroid estimation for larger cell structures  
- Monogenic-filter–based detection for texture-rich datasets  

### Tracking
- Optical-flow-assisted linking  
- Distance-based frame-to-frame association

### Visualization
- Overlays of tracks on raw or filtered images  
- MSD (Mean Square Displacement) analysis for all trajectories  
- Normalized trajectory visualizations  
- Path filtering based on MSD or custom metrics  
- Automatic video generation of the full tracking sequence  

---

## Repository Structure

```
ARGUS/
│
├─ **notebooks/** # Interactive Jupyter notebooks for demos and analysis
├─ **src/** # Core implementation of detection and tracking algorithms
├─ **results/** # Output folders for generated videos
└─ **README.md** # Project documentation (this file)
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-repo/ARGUS.git
cd ARGUS
```
### 2. Install Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Use the Notebooks
```bash
jupyter notebook
```
Useful starting points:
    - Detect and track cells (see `compute_cell_paths.ipynb`)
    - Visualize results (see `visualize_cell_tracks.ipynb`)

## Contributing

Contributions, bug reports, and feature requests are welcome!
Please open an issue or submit a pull request.

## Citation

A formal citation entry will be added soon.
In the meantime, feel free to reference this repository.

## License
MIT License

Copyright (c) 2025 The ARGUS Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.    
