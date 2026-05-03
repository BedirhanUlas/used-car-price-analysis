# Used Car Price Prediction

End-to-end regression analysis predicting used car market prices from 426,880 vehicle listings. Applies the CRISP-DM framework — from data cleaning through feature engineering to model evaluation — to identify the key drivers of resale value.

## Business Problem

Used car dealers and consumers lack reliable price benchmarks, leading to mispriced inventory and poor purchasing decisions. A data-driven pricing model enables:
- **Dealers:** Optimal pricing of incoming inventory to maximize margin
- **Consumers:** Fair-value assessment before negotiating
- **Marketplaces:** Automated price suggestions and anomaly detection for fraudulent listings

## Dataset

| Attribute | Value |
|---|---|
| Source | Kaggle — Craigslist used vehicle listings (USA) |
| Records | 426,880 listings |
| Features | 18 (year, manufacturer, model, condition, odometer, etc.) |
| Target | Price (USD) |
| Price range | $1 — $3,736,929 |

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (mean price) | $7,041 | — | 0.00 |
| Linear Regression | **$3,898** | **$3,603** | **0.61** |
| Ridge Regression (α=1.0) | $3,902 | $3,607 | 0.61 |
| Lasso Regression (α=1.0) | $3,920 | $3,618 | 0.60 |

Linear Regression achieved a **44.6% reduction in MAE** vs. the naive baseline.

## Key Price Drivers

Based on regression coefficients and feature analysis:

| Feature | Impact on Price | Direction |
|---|---|---|
| Year (vehicle age) | Very High | ↑ Newer = more expensive |
| Odometer | High | ↓ More miles = lower price |
| Condition | High | ↑ Excellent condition premium |
| Manufacturer | Moderate | Varies by brand |
| Drive type (4WD vs. FWD) | Moderate | ↑ 4WD commands premium |
| Fuel type | Low–Moderate | Varies |

**Main finding:** Vehicle year and odometer reading explain the majority of price variance, consistent with depreciation curves.

## CRISP-DM Methodology

```
Business Understanding → Data Understanding → Data Preparation
     → Modeling → Evaluation → (Deployment recommendations)
```

1. **Data Understanding** — 426K listings, significant missing values in condition/cylinders/size
2. **Data Preparation** — Dropped columns with >40% missing, removed price outliers (<$500, >$100K), encoded categoricals
3. **Feature Engineering** — Extracted vehicle age from year, log-transformed price distribution
4. **Modeling** — Linear, Ridge, Lasso regression with cross-validation
5. **Evaluation** — MAE, RMSE, R² on held-out test set; residual analysis

## Project Structure

```
used-car-price-analysis/
├── prompt_II.ipynb      # Full CRISP-DM analysis notebook
├── data/
│   └── vehicles.csv     # Craigslist listings dataset
└── LICENSE              # MIT
```

## Quick Start

```bash
git clone https://github.com/BedirhanUlas/used-car-price-analysis.git
cd used-car-price-analysis
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook prompt_II.ipynb
```

## Tech Stack

`Python` · `scikit-learn` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn`

## Future Improvements

- Apply ensemble methods (Random Forest, XGBoost, LightGBM) — expected R² > 0.85
- Add log-price transformation to handle right-skewed distribution
- NLP on vehicle description text for additional signal
- Deploy as a pricing API with real-time inference

## License

MIT
