# Enhancement of Machine Learning Models for Urban Heat Island Assessment

**Author:** Alexey Chechin  
**Institution:** University of Twente  
**Supervisors:** Dr. Ir. Monica Pena Acosta, Dr. Ir. Farid Vahdatikhaki

---

## Overview

This repository contains all code and data for the MSc thesis *"Enhancement of Machine Learning Models for Urban Heat Island Assessment"*. The work develops a spatial machine learning pipeline that improves Urban Heat Island (UHI) prediction by explicitly incorporating spatial dependencies between street segments — a component missing from prior data-driven approaches.

The core idea is that a street segment's heat intensity is not only influenced by its own physical properties (building density, vegetation, etc.) but also by the conditions of surrounding streets. The pipeline captures this by constructing a street-network graph, identifying neighbours for each segment, and aggregating their features before model training.

### Pipeline steps

1. Construct a street-network graph from coordinate data
2. Pre-compute distance and shortest-path matrices between all segment pairs
3. Select neighbours for each segment via a radius or node threshold
4. Aggregate neighbour features into the training dataset using one of four strategies
5. Train and optimise ML models (Random Forest, Neural Network, Linear Regression) via a Genetic Algorithm

The pipeline is trained on Apeldoorn (Netherlands) satellite data and evaluated on Rotterdam, Montreal, and Apeldoorn bike-collected datasets.

---

## Methodology in Brief

### Neighbour selection

Two methods define which street segments count as neighbours of a given segment:

| Method | Criterion | Matrix used |
|---|---|---|
| **Radius-based** | Haversine (straight-line) distance ≤ threshold *r* (metres) | `distance_matrix.npy` |
| **Node-based** | Number of intersections on shortest network path ≤ threshold *g* | `shortest_path_matrix.npy` |

The radius-based method captures geographic proximity (e.g., streets within 500 m). The node-based method captures network connectivity (e.g., streets reachable within 2 intersections), which better reflects how heat propagates along connected streets rather than across empty space.

### Feature aggregation strategies

Once neighbours are identified, their features are aggregated into the input dataset. Four strategies were tested and are selected via the `EXPERIMENT` variable in each code cell:

| `EXPERIMENT` | Name | Description | Output dimensionality |
|---|---|---|---|
| `1` | **Separate Means** | Unweighted mean of each neighbour feature appended as extra columns alongside the original features | 2× original (original + neighbour averages) |
| `2` | **United Means** | Each feature replaced by the average of its own value and the neighbour mean — blends local and contextual information into a single vector | Same as original |
| `3` | **PCA Means** | PCA applied per feature across the segment and its neighbours, compressing the neighbourhood into a single transformed value per feature | Same as original |
| `4` | **Weighted Means** | Like Separate Means, but neighbours contribute with weights based on distance and network proximity — closer and more connected streets receive higher weight. The blend between distance and shortest-path weighting is controlled by parameters α and β, which are also optimised by the GA | 2× original |

**Why four strategies?** Experiments 1–4 explore the trade-off between information preservation (Separate/Weighted Means keep both local and neighbour features) and dimensionality reduction (United/PCA Means compress them). Weighted Means additionally tests whether proximity-weighted aggregation outperforms simple averaging.

### Model training and optimisation

Each experiment is optimised using a **Genetic Algorithm (GA)** that searches for the best ML hyperparameters. For every spatial radius (or node threshold), a full GA run is performed independently:

- **Random Forest** hyperparameters: number of trees, max features, max depth, min samples per split, bootstrap
- **Neural Network** hyperparameters: activation function, batch size, epochs, layers, neurons per layer, optimiser, kernel initialisation
- **Objective**: maximise R² on the held-out test set (80/20 split)

---

## What Was Tested Where

Not all combinations were run on all datasets. The table below summarises the scope of the study:

| Dataset | Neighbour selection | Aggregation strategies | Models | Reason for scope |
|---|---|---|---|---|
| **Apeldoorn Satellite** | Radius + Node-based | EXP 1, 2, 3, 4 (all) | RF + NN + Linear Regression | Full training dataset; all methods explored |
| **Rotterdam** | Radius only | EXP 1, 4 | RF only | Street network is spatially fragmented — see note below |
| **Montreal** | Radius only | EXP 1, 4 | RF only | Same reason as Rotterdam |
| **Apeldoorn Bike** | Radius only | EXP 1, 4 | RF only | Closed 8 km loop; same fragmentation constraint |

**Why no node-based selection for Rotterdam, Montreal, and Bike data?**  
The street networks for these datasets are highly fragmented into many disconnected patches. Node-based selection requires a connected graph (to compute shortest paths). Extracting the largest connected component — as was done for Apeldoorn — would discard most of the segments and make the dataset too small to be meaningful. Therefore, only radius-based selection (which works without connectivity) was applied. Shortest-path matrices were computed for completeness but are not used in the analysis.

**Why only Experiments 1 and 4 for transfer tests?**  
After testing all four aggregation strategies on the Apeldoorn satellite data, Separate Means (EXP 1) and Weighted Means (EXP 4) emerged as the two most competitive methods. United Means and PCA Means did not offer consistent improvements, so the transfer tests were scoped to the two best-performing strategies.

---

## Repository Structure

```
.
├── Scripts Library.ipynb       # Main notebook — all code
├── requirements.txt            # Python dependencies
├── Datasets/
│   ├── Cl_mainRoads.xlsx           # Apeldoorn satellite dataset (training)
│   ├── Rotterdam_coordinates.xlsx  # Rotterdam dataset (transfer test)
│   ├── Montreal_coordinates.xlsx   # Montreal dataset (transfer test)
│   ├── BikeData_Coordinates.xlsx   # Apeldoorn bike dataset (transfer test)
│   └── Rest/
│       ├── Apeldoorn_final.xlsx
│       ├── Rotterdam_final.xlsx
│       ├── Montreal_final.xlsx
│       ├── Enschede_final.xlsx
│       └── NewYork_final.xlsx
└── precomputed_matrices/
    ├── edge_attributes.pkl              # Apeldoorn graph: edges + features
    ├── edge_attributes_rotterdam.pkl    # Rotterdam graph
    ├── edge_attributes_Montreal.pkl     # Montreal graph
    ├── edge_attributes_Bike.pkl         # Bike graph
    ├── timebased_attributes_Bike.pkl    # Bike temporal measurements
    ├── distance_matrix.npy              # Apeldoorn Haversine distance matrix  †
    ├── shortest_path_matrix.npy         # Apeldoorn network shortest-path matrix  †
    ├── distance_matrix_rotterdam.npy    # Rotterdam Haversine distance matrix  †
    ├── distance_matrix_Montreal.npy     # Montreal Haversine distance matrix  †
    ├── distance_matrix_Bike.npy         # Bike Haversine distance matrix
    ├── shortest_path_matrix_rotterdam.npy  # Rotterdam shortest-path (main component only; not used in analysis)
    └── shortest_path_matrix_Montreal.npy   # Montreal shortest-path (main component only; not used in analysis)
```

> **† Large files (Git LFS):** The four matrices marked † are tracked via Git LFS (combined ~1.2 GB). Install Git LFS before cloning (`git lfs install`). Alternatively, re-generate them from the Excel datasets by running **Cells 1–4** of the notebook. Note: the Apeldoorn all-pairs shortest-path computation is computationally intensive (~several hours on a standard laptop for 7,166 segments).

---

## Notebook Structure and Navigation

The notebook `Scripts Library.ipynb` is organised into seven sections. Each code cell is self-contained: set the parameters in the `USER CONFIGURATION` block at the top, then run the cell independently.

| Section | Cells | Description |
|---|---|---|
| **1. Graph & Matrix Construction** | 1–4 | Build street graphs and compute distance/shortest-path matrices from Excel data. Run these first if the precomputed matrices are not available. |
| **2. GA Random Forest — Apeldoorn Satellite** | 6–7 | Radius-based (Cell 6) and node-based (Cell 7) neighbour selection with GA-tuned Random Forest. All four aggregation strategies (EXP 1–4). |
| **3. GA Neural Network — Apeldoorn Satellite** | 9 | Radius-based neighbour selection with GA-tuned Neural Network. All four aggregation strategies. |
| **4. Linear Regression** | 11–12 | Spatially-lagged (SLX) linear regression (Cell 11) and non-spatial baseline (Cell 12). No EXPERIMENT variable — these cells run as-is. |
| **5. Transferability Tests** | 14–21 | Pipeline applied to Rotterdam (Cells 14–15), Montreal (Cells 16–17), and Apeldoorn Bike — Surface (Cells 18–19) and Canopy (Cells 20–21) temperature targets. Each dataset has one cell for EXP 1 and one for EXP 4. |
| **6. Feature Importance** | 23–24 | Feature importance sensitivity analysis across radii for EXP 4 (Cell 23) and EXP 1 (Cell 24). Uses hardcoded R² values from the GA runs. |
| **7. Visualisations** | 26–34 | All result plots from the thesis. Hardcoded from recorded GA runs — can be reproduced without re-running the GA. |

### Setting the experiment in a cell

Every cell in Sections 2, 3, and 5 has a `USER CONFIGURATION` block near the top. The key parameter is:

```python
EXPERIMENT = 1   # Set to 1, 2, 3, or 4
```

For the Bike cells (Section 5), the aggregation type and experiment mode are controlled separately:

```python
EXPERIMENT_MODE = "surface"   # or "canopy"
AGGREGATION_TYPES = [1, 4]    # runs both strategies in one cell
```

---

## Data Description

### Apeldoorn Satellite Dataset (`Cl_mainRoads.xlsx`, sheet `Cl_mainRoads_Proccessed`)
- **Source:** Landsat 7 satellite imagery; originally collected by Pena Acosta (2024), used with permission
- **Coverage:** 7,166 street segments, city of Apeldoorn, Netherlands
- **Target variable:** `Delta_T` — Land Surface Temperature difference (inner city vs. rural reference)
- **Features:** Building density, vegetation density, height/width ratio, street width, predominant facade material, predominant land use, mean population, mean/max building height, water density, bike cover, vehicle cover

### Transfer Test Datasets

| Dataset | City | Segments | Target | Notes |
|---|---|---|---|---|
| `Rotterdam_coordinates.xlsx` | Rotterdam, NL | ~2,700 | `Delta_T` | Satellite; highly fragmented network |
| `Montreal_coordinates.xlsx` | Montreal, CA | ~2,400 | `Delta_T` | Satellite; highly fragmented network |
| `BikeData_Coordinates.xlsx` | Apeldoorn, NL | 105 | `Delta_surface` / `Delta_canopy` | Bike-collected; closed 8 km loop; includes temporal features (humidity, reference temperatures) measured across multiple dates and times of day |

---

## Setup and Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Obtain precomputed matrices

**Option A — Git LFS (recommended):** Ensure Git LFS is installed before cloning:
```bash
git lfs install
git clone <repo-url>
```

**Option B — Regenerate:** Run Cells 1–4 sequentially. Each cell reads an Excel dataset, builds the street graph, and saves the corresponding `.npy` and `.pkl` files to `./precomputed_matrices/`. The Apeldoorn shortest-path matrix (Cell 1) is the most compute-intensive step.

### 3. Run the ML pipeline

Open `Scripts Library.ipynb`, go to the section of interest, set `EXPERIMENT` and any radius/threshold parameters in the `USER CONFIGURATION` block, and run the cell. All cells in Sections 2–6 are independent of each other (they each load the precomputed matrices directly).

### 4. Reproduce visualisations

Cells 26–34 produce all figures from the thesis. They contain hardcoded R² and feature importance values from the recorded GA runs and do not require the precomputed matrices. Run them directly.

---

## Reproducibility Notes

- All random seeds are fixed: `random_state=42` throughout scikit-learn calls, plus manual seeding of Python's `random` module where applicable.
- **GA settings used in the paper** — Random Forest: population 100, 50 generations; Neural Network: population 20, 50 generations; crossover 0.8, mutation 0.5, early stopping after 5 non-improving generations. The code cells use reduced defaults (population 20, 10 generations) to allow quick re-runs; restore the above values in the `USER CONFIGURATION` block to exactly replicate the thesis experiments.
- Train/test split: 80/20, fixed with `random_state=42`.
- The visualisation cells (Section 7) contain hardcoded result values from the recorded GA runs, ensuring exact reproducibility of all figures without re-running the full GA.

---

## Citation

If you use this code or data, please cite:

> Chechin, A. (2025). *Enhancement of Machine Learning Models for Urban Heat Island Assessment*. MSc Thesis, University of Twente.
