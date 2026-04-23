# Bio-Polymer Film Formulation Analysis & Predictive Modelling

**Python · scikit-learn · Pandas · NumPy · Matplotlib**

Bio-based film formulation analysis and ML-driven predictive modelling for chitin nanofiber / montmorillonite clay / sorbitol / gelatin composite films. The goal is to identify optimal formulations for a transparent, mechanically robust packaging film and build predictive models for key material properties.

---

## Application Context

Developing a bio-based film to replace traditional synthetic "window" laminates in dry-food pouches (coffee, tea, granola). The film must:
- Showcase the product on-shelf with high optical clarity
- Withstand packaging and distribution without tearing or puncturing
- Maintain integrity during accidental exposure to elevated temperatures

---

## Project Structure

```
├── data/
│   ├── materiom_dataset.csv       # Raw data (143 samples)
│   └── materiom_clean.csv         # Cleaned data (130 samples)
├── models/                        # Serialised model files (.pkl)
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_formulation_analysis.ipynb
│   └── 03_predictive_modeling.ipynb
├── results/                       # Plots and figures
├── predict.py                     # Inference script
├── references.md
└── requirements.txt
```

---

## Task 1 — Data Cleaning & Exploration

### Data Quality
- **143 raw samples**, 13 columns (4 ingredients, 9 measured properties)
- **13 duplicates removed** → 130 unique samples
- Mixed data types: unit suffixes (GPa, MPa, %) and placeholder characters parsed and cleaned
- Unit inconsistency: tensile strength/modulus mixed GPa and MPa — standardised to MPa
- Missing values (1–3 per column) imputed with column median

### Outlier Detection: Statistical + Physics-Based
Standard IQR box plots enhanced with **physical limits** (values impossible for bio-composite films) and **packaging target ranges** to distinguish extreme-but-valid formulations from measurement errors.

Two-tier anomaly detection:
1. **Hard physical limits (removed):** e.g. strength <1 MPa = gel not film; modulus >6 GPa = beyond biopolymers; transmission = 100% = impossible
2. **Composition-property inconsistencies (flagged, not removed):** Severity tiers 0–3 assigned rather than deleted — legitimate exceptions exist (well-dispersed nanofibers can maintain optical clarity at high loading; chitin N-acetyl groups provide fire resistance even without clay)

### Optical Properties
Three independent optical bands retained as separate features:
| Band | Mean | Std | Physical driver |
|---|---|---|---|
| Visible (400–700 nm) | 76% | — | Product visibility |
| UV (200–400 nm) | 79% | 6% | Gelatin aromatic amino acid absorption ~280 nm |
| IR (700+ nm) | 47% | 16% | Chitin nanofiber and clay scattering |

---

## Task 2 — Formulation Analysis

### Scoring Weights
| Property | Weight | Rationale |
|---|---|---|
| Optical clarity | 35% | Visible light transmission — product display |
| Tensile strength | 35% | Mechanical integrity during packaging/distribution |
| Fire/thermal resistance | 20% | Integrity near heat-sealing equipment |
| Flexibility | 10% | Moderate elongation prevents cracking |

### Top 3 Formulations
| Rank | Chitin | Clay | Gelatin | Sorbitol | Strength | Optical | Fire | Score |
|---|---|---|---|---|---|---|---|---|
| 1 | 37.6% | 43.4% | 6.2% | 12.8% | 108.5 MPa | 71.9% | 0.76 | 0.77 |
| 2 | 38.6% | 12.3% | 35.8% | 13.3% | 96.3 MPa | 77.9% | 0.74 | 0.76 |
| 3 | 48.6% | 5.2% | 28.6% | 17.6% | 96.6 MPa | 82.7% | 0.43 | 0.72 |

**Recommended: Rank 2** (~39% chitin, ~12% clay, ~36% gelatin, ~13% sorbitol)
- Within 2% of Rank 1 score with 6% better optical clarity
- Higher gelatin content (36%) enables conventional film-casting — better processability
- More robust to batch variation than Rank 1 (extreme 43% clay loading risks brittle composite)

---

## Task 3 — Predictive Modelling

### Feature Engineering
Weighted-average physical constants (refractive index, Tg, density) collapse to near-perfect proxies of individual ingredients (r > 0.9) in a 4-component mixture — excluded to avoid redundancy. With only 130 samples, redundant features dilute signal and increase overfitting risk.

**Final 8 features:**
| Feature | Type |
|---|---|
| sorbitol, clay, gelatin, chitin | Raw ingredients |
|  | Reinforcement-to-matrix balance |
|  | Combined filler loading (chitin + clay) |
|  | Reinforcement synergy interaction |
|  | Plasticisation effect interaction |

>  has VIF = ∞ (exact linear combination) but is retained — Random Forest and Gradient Boosting split on ranked thresholds and are unaffected by multicollinearity. Removing a physically meaningful feature to satisfy a linear-model assumption when no linear model is selected discards real signal without benefit.

### No Data Leakage
Tensile modulus, stress-strain alpha/beta were excluded as model inputs — they are **measured output properties** not known before manufacturing. Including them would inflate R² by teaching inter-property correlations rather than composition-to-property relationships.

### Model Comparison
Five models evaluated: Ridge Regression (linear baseline), Random Forest, Gradient Boosting, SVR (RBF kernel), KNN. Best model selected **per target** — tensile and optical are governed by different physical mechanisms.

### Final Results

| Target | Model | MAE | RMSE | R² |
|---|---|---|---|---|
| Tensile Strength | Gradient Boosting (tuned) | 13.11 | 17.40 | **0.51** |
| Optical Transmission (Vis) | Random Forest | 2.38 | 3.36 | **0.89** |

### vs. Naive Baseline (predicting training mean)
| Target | MAE improvement | RMSE improvement |
|---|---|---|
| Tensile Strength | 37.8% | 32.8% |
| Optical Transmission | 70.1% | 66.7% |

### Why the Gap Between R² 0.89 and 0.51

The optical model substantially outperforms the tensile model because **optical transmission is directly governed by composition** — filler scattering follows Rayleigh/Mie theory, so chitin nanofiber and clay loading directly set transmission.

Tensile strength depends on **microstructural factors** — filler dispersion quality, interfacial adhesion, void content, crystallinity — none captured by composition percentages alone.

Feature importance confirms this: chitin nanofibers account for **56.5% of optical model importance**, while tensile importance is spread across all ingredients (sorbitol 32.4%, gelatin 19.5%, clay 12.7%), reflecting the complex multi-factor nature of mechanical failure.

> This is a **fundamental limitation of composition-only features**, not a model deficiency.

### Transformation Experiment
Optical transmission is left-skewed (skewness = −1.82). Three transforms tested (log reflected, square, Box-Cox) vs. untransformed baseline — **none adopted**. No transform exceeded both adoption thresholds (>5% MAE improvement AND >0.02 R² improvement), confirming tree-based models are inherently robust to target skewness.

### Cross-Validation
5-fold CV run on best model per target (scaler fitted inside each fold to prevent normalisation leakage), giving mean ± std across 5 splits as a more reliable generalisation estimate than a single 80/20 split on 88 samples.

---

## Future Improvements

- **Processing features:** Casting temperature, drying time, mixing conditions could significantly improve tensile prediction by capturing microstructural variation
- **Ensemble stacking:** Combine predictions from multiple models as a blended estimator
- **Larger dataset:** More samples would reduce sensitivity of R² to individual test-set composition

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3 |
| ML | scikit-learn (Random Forest, Gradient Boosting, Ridge, SVR, KNN) |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Notebooks | Jupyter |
| Serialisation | joblib / pickle |
