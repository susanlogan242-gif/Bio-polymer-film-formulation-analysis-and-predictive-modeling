# Bio-polymer film formulation analysis and predictive modeling

## Overview
Bio-based film formulation analysis and predictive modeling for chitin nanofiber / montmorillonite clay / sorbitol / gelatin composite films. The goal is to identify optimal formulations for a transparent, mechanically robust packaging film and build predictive models for key material properties.

## Application Context
Developing a bio-based film to replace traditional synthetic "window" laminates in dry-food pouches (coffee, tea, granola). The film must:
- Showcase the product on-shelf with high optical clarity
- Withstand packaging and distribution without tearing or puncturing
- Maintain integrity during accidental exposure to elevated temperatures

## Project Structure
```
Materiom/
├── data/                              # Raw and cleaned datasets
├── models/                            # Trained model files (.pkl)
├── results/                           # Prediction outputs and plots
├── notebooks/
│   ├── 01_data_cleaning.ipynb         # Data loading, inspection and cleaning
│   ├── 02_formulation_analysis.ipynb  # Task 1: Formulation recommendation
│   ├── 03_predictive_modeling.ipynb   # Task 2: Predictive modeling
│   ├── utils.py                       # Shared utility functions
│   ├── recommendations.md             # Task 1 detailed writeup
│   └── writeup.md                     # Full technical writeup
├── predict.py                         # Prediction entrypoint for holdout data
├── references.md                      # Literature references and physical constants
├── requirements.txt                   # Python dependencies
└── README.md
```

## Task 1: Formulation Analysis

### Approach

The formulation analysis follows a systematic multi-criteria evaluation:

1. **Data Cleaning** — The dataset (143 raw samples) is noisy with mixed units (MPa/GPa), percentage symbols in numeric fields, missing values marked as `?`, and 13 duplicate rows. These are standardised and cleaned to 130 unique samples.

2. **Requirements Mapping** — Translated the application requirements (food pouch window film) into quantitative property targets based on industry standards for flexible packaging:
   - Optical transmission (visible): ≥80% for clear product visibility
   - Tensile strength: 30-60 MPa for puncture/tear resistance during handling
   - Post-fire residual ratio: maximised as a proxy for thermal stability near sealing equipment

3. **Physics-Based Outlier Detection** — Beyond standard IQR box plots, physical limits (values impossible for bio-composite films) were used to remove measurement errors, while composition-property inconsistencies were flagged with severity tiers rather than deleted (legitimate exceptions exist in nanocomposite systems).

4. **Multi-Criteria Scoring** — Candidates are normalised to a 0-1 scale and scored using weighted criteria:
   - 35% optical transmission (primary function as a "window")
   - 35% tensile strength (must survive packaging and distribution)
   - 20% fire test ratio (thermal stability)
   - 10% flexibility (moderate elongation preferred)

5. **Composition Analysis** — Top-scoring formulations are analysed for compositional patterns, interpreted through material science understanding of each ingredient's role.

### Key Requirements
| Requirement | Property | Desired |
|---|---|---|
| Optical clarity | Visible light transmission | Higher is better |
| Mechanical strength | Tensile strength | Higher is better |
| Thermal stability | Fire test (material remaining) | Higher is better |

### Recommendation

The top 3 formulations represent distinct trade-off strategies:

| Rank | Chitin | Clay | Gelatin | Sorbitol | Strength | Optical | Fire | Score |
|------|--------|------|---------|----------|----------|---------|------|-------|
| 1 | 37.6% | 43.4% | 6.2% | 12.8% | 108.5 MPa | 71.9% | 0.76 | 0.77 |
| 2 | 38.6% | 12.3% | 35.8% | 13.3% | 96.3 MPa | 77.9% | 0.74 | 0.76 |
| 3 | 48.6% | 5.2% | 28.6% | 17.6% | 96.6 MPa | 82.7% | 0.43 | 0.72 |

**Rank 2 (~39% chitin, ~12% clay, ~36% gelatin, ~13% sorbitol) is recommended** as the primary candidate because it offers balanced performance (within 2% of Rank 1's score with 6% better optical clarity), better processability (the ~36% gelatin matrix enables conventional film-casting), and robustness to batch variation. Rank 1's extreme clay loading (43%) with minimal gelatin (6%) risks a brittle, filler-dominated composite despite its superior strength.

### Material Science Context
The four ingredients serve distinct roles in the composite film:

| Ingredient | Role | Effect on Properties |
|---|---|---|
| **Gelatin** | Matrix polymer | Film-forming base, good transparency, moderate strength |
| **Chitin nanofibers** | Reinforcing agent | Increases tensile strength and stiffness, may reduce transparency at high loadings |
| **Montmorillonite clay** | Nano-filler | Improves barrier properties and fire resistance, can reduce transparency |
| **Sorbitol** | Plasticizer | Improves flexibility and processability, may reduce strength |

## Task 2: Predictive Modeling

### Targets
- Tensile strength (MPa)
- Optical transmission - visible band (%)

### Feature Engineering

A multicollinearity analysis revealed that weighted-average physical constants (refractive index, Tg, density) collapse to near-perfect proxies of individual ingredients (r > 0.9) in a 4-component mixture. With only 130 samples, redundant features dilute signal and increase overfitting risk.

**Final feature set (8 features):**
- 4 raw ingredients: sorbitol, montmorillonite clay, gelatin, chitin nanofibers
- 4 non-redundant engineered features:
  - `chitin_gelatin_ratio` — reinforcement-to-matrix balance
  - `total_reinforcement` — combined filler loading (chitin + clay)
  - `chitin_x_clay` — reinforcement synergy interaction
  - `gelatin_x_sorbitol` — plasticisation effect interaction

### Model Comparison

Five models were compared: Ridge Regression (linear baseline), Random Forest, Gradient Boosting, SVR (RBF kernel), and KNN. The best model is selected per target since tensile strength and optical transmission are governed by different physical mechanisms.

Optical target skewness (-1.82) was addressed by testing log (reflected), square, and Box-Cox transformations against the untransformed baseline.

### Results

| Target | Model | MAE | RMSE | R² |
|---|---|---|---|---|
| Optical Trans. Vis | Random Forest | 2.49 | 3.60 | 0.87 |
| Tensile Strength | Best after tuning (see notebook 03) | — | — | — |

*Re-run notebook 03 to regenerate exact figures for both targets.*

### Value vs Baseline
- **Optical Transmission:** ~69% MAE improvement over naive baseline (predicting training set mean)
- The optical model outperforms the tensile model because optical transmission is directly governed by composition (filler scattering follows Rayleigh/Mie theory), whereas tensile strength depends on microstructural factors (filler dispersion, interfacial adhesion, void content) not captured by composition alone

### Future Enhancements
With 2 additional days:
- **Processing features:** Adding casting temperature, drying time, and mixing conditions to capture microstructural variation
- **Full cross-validation:** k-fold CV for all model evaluations (partially implemented via GridSearchCV)
- **Ensemble stacking:** Combine predictions from multiple models as a blended estimator

## How to Run

### Setup
```bash
pip install -r requirements.txt
```

### Run Notebooks
```bash
cd notebooks
jupyter notebook
```
Run notebooks in order: `01` → `02` → `03`

### Predict on New Data
To evaluate on a holdout dataset:
```bash
python predict.py --input data/holdout_dataset.csv --output results/predictions.csv
```

The script loads trained models from `models/`, engineers the same features used during training, and outputs predictions for tensile strength and optical transmission.

## AI Tools Usage
This project used **Claude Code** (Anthropic's CLI tool) as an AI coding assistant for:
- Project scaffolding and notebook structure
- Code debugging and troubleshooting
- Material science domain knowledge discussions
- Physics-based anomaly detection criteria (literature-informed physical limits)
- Multicollinearity analysis and feature selection rationale
- Literature research on physical constants (refractive indices, Tg, densities)

**Perplexity AI** was also used for literature research on material properties and physical constants.

## References
See [references.md](references.md) for the full list of literature references and physical constants used.

## Author
Amirah
