# Materiom Assessment - Writeup

## Data Cleaning & Exploration (Notebook 01)

### Data Quality Issues Found
- **143 raw samples** with 13 columns (4 ingredients, 9 measured properties)
- **13 duplicate rows** removed → 130 unique samples
- **Mixed data types:** Values contained unit suffixes (GPa, MPa, %) and placeholder characters ('?') requiring parsing
- **Unit inconsistency:** Tensile strength/modulus mixed GPa and MPa — standardised to MPa
- **Missing values:** 1-3 per column across 7 properties, imputed with column median

### Outlier Detection: Statistical + Physics-Based

Standard IQR-based box plots were enhanced with **physical limits** (red lines — values impossible for bio-composite films) and **packaging target ranges** (green bands) to distinguish genuine extreme formulations from measurement errors.

**Physics-based anomaly detection** used a two-tier approach:
1. **Hard physical limits (REMOVED):** Values outside bio-composite capability (e.g. strength <1 MPa = gel not film, modulus >6 GPa = beyond biopolymers, transmission = 100% = impossible)
2. **Composition-property inconsistencies (FLAGGED, not removed):** Unusual combinations assigned severity tiers (0=clean, 1=minor, 2=moderate, 3=severe) rather than deleted, because legitimate exceptions exist:
   - Well-dispersed nanofibers can maintain optical clarity at high loading (Rayleigh scattering regime)
   - Chitin's N-acetyl groups form nitrogen-rich char, providing fire resistance even without clay
   - Phase separation (stiff chitin network + soft plasticized gelatin) produces unexpected mechanical combinations
   - Poor filler dispersion/aggregation can reduce modulus despite high nominal loading

### Wavelength-Dependent Optical Properties

The three optical bands are independent measurements (not duplicates):
- **Visible (400-700 nm):** Mean 76%, critical for product visibility (r_vis-UV = 0.75, r_vis-IR = 0.90)
- **UV (200-400 nm):** Mean 79%, narrowly distributed (std: 6%) — dominated by gelatin's aromatic amino acid absorption (~280 nm)
- **IR (700+ nm):** Mean 47%, widest spread (std: 16%) — composition-sensitive scattering by chitin nanofibers and clay platelets

All three retained as independent features capturing distinct physical phenomena.

### Key Property Trade-offs

Correlations between the 5 target properties (strength, modulus, strain, optical clarity, fire resistance) were analysed to identify inherent material trade-offs constraining formulation optimisation. These inform the scoring weights in Task 1.

---

## Task 1: Formulation Analysis (Notebook 02)

### Approach

Formulations were scored using a weighted multi-criteria system based on packaging film requirements:
- **Optical clarity** (35% weight) — visible light transmission, higher = better for product display
- **Tensile strength** (35% weight) — mechanical integrity during packaging and distribution
- **Fire/thermal resistance** (20% weight) — post-fire residual ratio, integrity near sealing equipment
- **Flexibility** (10% weight) — tensile strain, moderate elongation preferred

Each property was normalised to 0-1 scale. Ingredient effects on key properties were visualised via scatter plots (4 ingredients x 3 key properties).

### Rationale

Optical clarity and tensile strength share equal top priority (35% each) because the film must be transparent enough for product visibility while strong enough to survive packaging and distribution. Fire/thermal resistance (20%) accounts for integrity near heat-sealing equipment. Flexibility receives the lowest weight (10%) — moderate elongation prevents cracking but excessive flexibility undermines structural integrity.

### Recommendation

The top 3 formulations represent distinct trade-off strategies:

| Rank | Chitin | Clay | Gelatin | Sorbitol | Strength | Optical | Fire | Score |
|------|--------|------|---------|----------|----------|---------|------|-------|
| 1 | 37.6% | 43.4% | 6.2% | 12.8% | 108.5 MPa | 71.9% | 0.76 | 0.77 |
| 2 | 38.6% | 12.3% | 35.8% | 13.3% | 96.3 MPa | 77.9% | 0.74 | 0.76 |
| 3 | 48.6% | 5.2% | 28.6% | 17.6% | 96.6 MPa | 82.7% | 0.43 | 0.72 |

**Rank 2 (~39% chitin, ~12% clay, ~36% gelatin, ~13% sorbitol) is recommended** as the primary candidate because it offers balanced performance (within 2% of Rank 1's score with 6% better optical clarity), better processability (the ~36% gelatin matrix enables conventional film-casting), and robustness to batch variation. Rank 1's extreme clay loading (43%) with minimal gelatin (6%) risks a brittle, filler-dominated composite despite its superior strength.

---

## Task 2: Predictive Modeling (Notebook 03)

### Feature Engineering

A multicollinearity analysis revealed that weighted-average physical constants (refractive index, glass transition temperature, density) collapse to near-perfect proxies of individual ingredients (r > 0.9) in a 4-component mixture — they add no new information. With only 130 samples, redundant features dilute signal and increase overfitting risk.

**Final feature set (8 features):**
- 4 raw ingredients: sorbitol, montmorillonite clay, gelatin, chitin nanofibers
- 4 non-redundant engineered features capturing non-linear relationships:
  - `chitin_gelatin_ratio` — reinforcement-to-matrix balance
  - `total_reinforcement` — combined filler loading (chitin + clay)
  - `chitin_x_clay` — reinforcement synergy interaction
  - `gelatin_x_sorbitol` — plasticisation effect interaction

**Dropped (redundant):** estimated_density (r=0.997 with clay), avg_RI (r=-0.964 with gelatin), RI_variance, estimated_Tg (r=-0.903 with sorbitol), filler_x_plasticizer, plasticizer_polymer_fraction, polymer_plasticizer_ratio, total_polymer

### Target Skewness

Notebook 01 identified that optical transmission (visible) has a skewness of -1.82 (left-skewed), while tensile strength is approximately symmetric (skew = 0.50). Three transformations were tested on the optical target — log (reflected), square, and Box-Cox — and compared against the untransformed baseline using Random Forest on the same train/test split. All metrics were evaluated on the original scale after inverse-transforming predictions.

**Outcome:** The transformation experiment is self-evaluating in the notebook. If any transform achieves >5% MAE improvement AND >0.02 R² improvement over baseline, it is automatically adopted for the model comparison. Otherwise, it is documented as a known limitation — tree-based models (Random Forest, Gradient Boosting) are inherently robust to target skewness because they split on thresholds rather than assuming normally distributed residuals.

### Model Selection

Five models were compared: Ridge Regression (linear baseline), Random Forest, Gradient Boosting, SVR (RBF kernel), and KNN. Rather than selecting one model for both targets, the **best model is selected per target** — this is important because tensile strength and optical transmission are governed by different physical mechanisms and may suit different model architectures.

The top 2 tensile candidates were further tuned via GridSearchCV with 5-fold cross-validation to maximise tensile strength prediction. Evaluation metrics: MSE, RMSE, MAE, and R².

### Results

*(Values below are from the most recent notebook run — re-run to regenerate exact figures.)*

| Target | Model | MAE | RMSE | R² |
|---|---|---|---|---|
| Tensile Strength | Best after tuning (see notebook) | — | — | — |
| Optical Trans. Vis | Random Forest | 2.49 | 3.60 | 0.87 |

### Value vs Baseline

Compared against a naive baseline (predicting the training set mean for all test samples):
- **Optical Trans. Vis:** ~69% MAE improvement over baseline

The optical model performs substantially better than the tensile model because optical transmission is directly governed by composition (filler scattering follows Rayleigh/Mie theory). Tensile strength depends on **microstructural factors** — filler dispersion quality, interfacial adhesion, void content, crystallinity — none of which are captured by composition percentages alone. This is a fundamental limitation of composition-only features, not a model deficiency.

### Future Enhancements (with 2 more days)

- **Processing features:** Adding casting temperature, drying time, and mixing conditions could significantly improve tensile prediction by capturing microstructural variation
- **Cross-validation:** Full k-fold CV for all model evaluations (partially implemented via GridSearchCV tuning)
- **Ensemble stacking:** Combine predictions from multiple models as a blended estimator

---

## AI Tools Usage
This project used Claude Code (Anthropic's CLI) for:
- Project setup and notebook scaffolding
- Code debugging and explanation
- Material science domain knowledge discussion
- Physics-based anomaly detection criteria (literature-informed physical limits)
- Multicollinearity analysis and feature selection rationale

Example prompts used:
- "flag (not remove) physics-based anomalies like "high sorbitol but high strength" since they might be unique cases"
- "fix unreadable boxplots, overlapping titles, switch to Plotly, make plots insightful not decorative"
- "The random forest model does not appear to be the best model for predicting tensile strength, please expand the types of models tested and train the model that best fits each target propety."
