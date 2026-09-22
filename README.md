# AAZHI / OceanGuard — AI Microplastic Risk Prioritization

An AI-powered decision-support tool that predicts ocean microplastic risk
(High/Low) at any coordinate, using a Random Forest classifier trained on
NOAA's real Marine Microplastics Database. Built to help prioritize where
limited ocean monitoring and cleanup resources go, especially in regions
with almost no direct measurement (e.g. the entire Indian Ocean coastline).

**[Live Dashboard →](https://yourusername.github.io/oceanguard/)** *(update after enabling GitHub Pages)*

## Key Results

| Metric | Value |
|---|---|
| Accuracy | 91.0% (single split) / 91.2% ± 0.5% (5-fold CV) |
| Recall (High-risk) | 84.1% / 85.2% ± 1.2% |
| Precision (High-risk) | 80.6% / 80.8% ± 1.0% |
| ROC-AUC | 0.957 / 0.958 ± 0.003 |

Full quantitative breakdown (confusion matrix, per-fold scores, feature
importance) is in [`models/full_quant_metrics.json`](models/full_quant_metrics.json).

## Key Finding

**Zero verified samples exist within 2,500km of India's coastline** (Arabian
Sea + Bay of Bengal), confirmed directly from the data. Globally, the Indian
Ocean accounts for just 17 of 13,861 records (0.12%) — and even those cluster
off South Africa, not India. This is the tool's highest-priority
recommendation for new sensor deployment.

## Repo Structure

```
oceanguard/
├── data/
│   ├── raw/                    # original NOAA export
│   └── processed/               # cleaned dataset (13,861 records)
├── src/
│   ├── clean_data.py             # raw → cleaned
│   └── train_model.py            # cleaned → trained model + dashboard data
├── models/                       # trained model + evaluation metrics
├── docs/                         # the live dashboard (GitHub Pages root)
│   ├── index.html
│   └── data/                     # grid.json, samples.json, metrics.json
└── assets/                       # charts, images
```

## Reproducing This Project

```bash
git clone https://github.com/yourusername/oceanguard.git
cd oceanguard
pip install -r requirements.txt

python src/clean_data.py     # data/raw → data/processed/noaa_cleaned.csv
python src/train_model.py    # trains model, writes docs/data/*.json
```

Then open `docs/index.html` via a local server (it fetches its data files,
so it needs `http://`, not `file://`):

```bash
cd docs && python -m http.server 8000
# visit http://localhost:8000
```

## Data Pipeline Notes

The raw NOAA export mixes several incompatible measurement conventions.
`clean_data.py` handles this explicitly:

1. **Unit filtering**: keeps only `pieces/m3` (dropped ~6,000 records using
   time-based or mass-based units not comparable to volume concentration).
2. **Sampling-method filtering**: excludes "Stainless steel spoon," "PVC
   cylinder," and "Aluminum bucket" — these produced concentration values
   1,000-10,000x higher than net-based methods, strongly suggesting a
   different (sediment/bulk) measurement regime mislabeled under the same
   unit.
3. **Threshold**: the elevated-risk cutoff (0.08 pieces/m³) is the 75th
   percentile of the cleaned concentration values — data-derived, not
   arbitrary.

## Model

Random Forest (300 trees, max_depth=12, class_weight="balanced") on:
`Latitude, Longitude, Year, Month, Ocean, Sampling Method, distance-to-nearest-gyre`.

The dashboard's risk grid is masked to only display predictions within
**800km of a real verified sample** — the map intentionally leaves regions
blank rather than implying confidence where there's no supporting data.

## Limitations

- This is a prioritization layer, not a replacement for direct sensor-based
  detection.
- Coverage is heavily skewed toward the Atlantic (68.8%) and Pacific
  (29.8%); predictions in under-sampled regions carry more uncertainty even
  within the 800km mask.
- The "Sampling Method" adjustment shown in the dashboard's method selector
  is a post-hoc statistical multiplier (relative high-risk rate by method),
  not a re-run of the trained model with that feature changed.

## Data Source

NOAA National Centers for Environmental Information (NCEI), Marine
Microplastics Database.
