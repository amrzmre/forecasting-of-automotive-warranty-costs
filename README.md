# forecasting-of-automotive-warranty-costs

# AI-Driven Forecasting of Automotive Warranty Costs

**Predicting monthly alternator warranty claim costs using machine learning and time series analysis**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Business Problem](#business-problem)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Models Evaluated](#models-evaluated)
- [Installation](#installation)
- [Usage](#usage)
- [Data Requirements](#data-requirements)
- [Results & Validation](#results--validation)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

This project develops a production-ready forecasting system for monthly automotive warranty costs of alternator P Model. Using historical warranty claims data (FY2019–FY2023), built and validated four machine learning models to predict FY2024 costs with 93% annual accuracy.

**Key Features:**
- ✅ Multi-model ensemble approach (Linear Regression, ARIMA, Random Forest, XGBoost)
- ✅ Rigorous 3-stage validation (in-sample → out-of-sample → production)
- ✅ Handles COVID-19 data gaps and seasonal patterns
- ✅ Production-ready Python codebase with comprehensive documentation
- ✅ Executive-friendly visualizations and accuracy metrics
  
---

## 💼 Business Problem

**Challenge:**  
Alternator's warranty budget planning relied on informal rules of thumb and expert judgment, leading to:
- Over-provisioning (excess reserves tied up unnecessarily)
- Under-provisioning (mid-year budget shortfalls)
- Limited audit trail for budget decisions

**Objective:**  
Build a data-driven forecasting system that:
1. Predicts monthly warranty costs with ≥80% accuracy
2. Provides uncertainty bounds for contingency planning
3. Automatically updates as new data becomes available
4. Explains model decisions to non-technical stakeholders

**Impact:**
- **93% annual forecast accuracy** (ARIMA model)
- **RM 2,310 average annual forecast error** (7% of actual)
- **Reduced budget uncertainty** from ±50% to ±30%
- **Quarterly retraining capability** as new data accumulates

---

## 🏆 Key Results

### FY2024 Forecast Validation

| Model | Monthly Accuracy | Annual Accuracy | Annual Error | Grade | Recommendation |
|-------|------------------|-----------------|--------------|-------|----------------|
| **ARIMA** | **52.8%** | **92.9%** | **−7.1%** | **Good** | **Primary model** |
| Random Forest | 33.7% | 70.6% | −29.4% | Fair | Backup |
| XGBoost | 22.6% | 65.0% | −35.0% | Fair | Backup |
| Linear Regression | 0.0% | 0.0% | −100.6% | Poor | Upper bound only |

**Actual FY2024 Results:**
- Total claims: 351 accepted claims
- Total cost: **RM 32,714**
- ARIMA forecast: **RM 35,024** (7.1% over-prediction)
- Planning range: RM 30,000–60,000 (±30% uncertainty)

### Why ARIMA Won

All models learned from FY2019–FY2023 that costs rise each year. When FY2024 unexpectedly declined, the **conservative model** (ARIMA: "expect stability") was closest to reality, while the **aggressive model** (Linear Regression: "growth will continue") over-predicted by 101%.

**Key Insight:** With limited historical data (46 months), conservative forecasting outperforms sophisticated trend extrapolation when patterns change.

---

## 📁 Project Structure

```
warranty-cost-forecasting/
│
├── data/
│   ├── raw/                          # Original warranty claims exports
│   ├── processed/                    # Cleaned and aggregated datasets
│   └── validation/                   # FY2024 actual data for validation
│
├── notebooks/
│   ├── phase1_data_preparation/
│   │   ├── block01_data_loading.ipynb
│   │   ├── block02_data_quality.ipynb
│   │   └── block03_monthly_aggregation.ipynb
│   │
│   ├── phase2_exploratory_analysis/
│   │   ├── block04_temporal_patterns.ipynb
│   │   ├── block05_distribution_analysis.ipynb
│   │   └── block06_business_metrics.ipynb
│   │
│   ├── phase3_model_development/
│   │   ├── block07_feature_engineering.ipynb
│   │   ├── block08_linear_regression.ipynb
│   │   ├── block09_arima.ipynb
│   │   ├── block10_random_forest.ipynb
│   │   └── block11_xgboost.ipynb
│   │
│   ├── phase4_validation/
│   │   └── block12_iterative_validation.ipynb
│   │
│   └── phase5_production/
│       ├── block14_production_forecasting.ipynb
│       └── block15_forecast_validation.ipynb
│
├── src/
│   ├── data_processing/              # Data cleaning and aggregation modules
│   ├── feature_engineering/          # Feature generation utilities
│   ├── models/                       # Model training and prediction classes
│   ├── validation/                   # Validation metrics and visualizations
│   └── utils/                        # Helper functions and constants
│
├── outputs/
│   ├── models/                       # Trained model artifacts (.pkl)
│   ├── forecasts/                    # Production forecast outputs (.csv)
│   ├── visualizations/               # Charts and plots (.png)
│   └── reports/                      # Executive summaries (.pdf)
│
├── docs/
│   ├── project_conclusion.md         # Comprehensive project summary
│   ├── workflow_explanation.md       # Technical and C-level documentation
│   └── model_selection_rationale.md  # Why ARIMA was chosen
│
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package installation script
├── config.yaml                       # Project configuration
├── LICENSE                           # MIT License
└── README.md                         # This file
```

---

## 🔬 Methodology

### 3-Stage Validation Framework

Our workflow follows industry best practices for production ML systems:

#### **Phase 3: Model Development** (Blocks 8–11)
- Train each model on full historical data (FY2019, FY2021–2023)
- Evaluate in-sample metrics (R², RMSE, MAPE)
- Tune hyperparameters to minimize overfitting
- **Purpose:** Establish baseline performance and identify structural issues

#### **Phase 4: Iterative Validation** (Block 12)
- Progressive out-of-sample testing across 3 iterations:
  - Iteration 1: Train(FY19) → Test(FY21)
  - Iteration 2: Train(FY19+21) → Test(FY22)
  - Iteration 3: Train(FY19+21+22) → Test(FY23)
- **Purpose:** Test if models generalize to unseen fiscal years

#### **Phase 5: Production Deployment** (Blocks 14–15)
- Train all models on complete history (FY19+21+22+23)
- Generate FY2024 forecast (Block 14)
- Validate against actual FY2024 data (Block 15)
- **Purpose:** Produce actionable forecast and measure real-world accuracy

**Why 3 stages?**  
Each stage catches different failure modes:
- Stage 1 catches models that can't even fit historical data
- Stage 2 catches models that overfit and can't generalize
- Stage 3 catches models that fail when reality changes

---

## 🤖 Models Evaluated

### 1. **Linear Regression**
- **Approach:** OLS with time trend + fiscal month dummies
- **Strengths:** Interpretable, fast, stable
- **Weaknesses:** Cannot adapt to non-linear patterns
- **Result:** 100.6% over-prediction (catastrophic failure)
- **Use case:** Worst-case upper bound for stress testing

### 2. **ARIMA (AutoRegressive Integrated Moving Average)**
- **Specification:** ARIMA(0,1,1)
- **Approach:** Differencing to remove trend + moving average correction
- **Strengths:** Conservative, doesn't make bold predictions
- **Weaknesses:** Constant forecast (no month-to-month variation)
- **Result:** 7.1% over-prediction, 93% annual accuracy ✅
- **Use case:** **PRIMARY MODEL** for FY2025 planning

### 3. **Random Forest**
- **Configuration:** 20 trees, max_depth=5 (conservative to prevent overfitting)
- **Approach:** Ensemble of decision trees averaging predictions
- **Strengths:** Handles non-linear patterns, robust to outliers
- **Weaknesses:** Cannot extrapolate beyond training range
- **Result:** 29.4% over-prediction, 71% annual accuracy
- **Use case:** Backup model with manual adjustment

### 4. **XGBoost (Gradient Boosting)**
- **Configuration:** 10 rounds, depth=2, strong L1/L2 regularization
- **Approach:** Sequential trees correcting each other's errors
- **Strengths:** State-of-art accuracy on structured data
- **Weaknesses:** Requires careful tuning, limited data (46 months) hurts performance
- **Result:** 35.0% over-prediction, 65% annual accuracy
- **Use case:** Monitor for improvement as more data accumulates

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- 4GB RAM minimum (8GB recommended for full dataset)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/warranty-cost-forecasting.git
cd warranty-cost-forecasting
```

2. **Create virtual environment:**
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n warranty-forecast python=3.8
conda activate warranty-forecast
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify installation:**
```bash
python -c "import pandas, numpy, sklearn, xgboost, statsmodels; print('All dependencies installed successfully')"
```

### Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
statsmodels>=0.13.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
openpyxl>=3.0.0  # For Excel file handling
```

---

## 🚀 Usage

### Quick Start

**1. Prepare your data:**
```python
# Place warranty claims CSV in data/raw/
# Required columns: Claimed Month, Accept/Reject status, RM, Model, Part Category
```

**2. Run the full pipeline:**
```bash
# Process all notebooks sequentially
jupyter nbconvert --execute notebooks/phase1_data_preparation/*.ipynb
jupyter nbconvert --execute notebooks/phase2_exploratory_analysis/*.ipynb
jupyter nbconvert --execute notebooks/phase3_model_development/*.ipynb
jupyter nbconvert --execute notebooks/phase4_validation/*.ipynb
jupyter nbconvert --execute notebooks/phase5_production/*.ipynb
```

**3. Or run individual phases:**
```bash
# Just generate FY2024 forecast
jupyter notebook notebooks/phase5_production/block14_production_forecasting.ipynb
```

### Command-Line Interface

```bash
# Train all models and generate forecast
python src/models/train.py --config config.yaml --output outputs/forecasts/

# Validate against actual data
python src/validation/validate.py --forecast outputs/forecasts/fy24_forecast.csv \
                                   --actual data/validation/fy24_actual.csv

# Generate executive report
python src/reports/generate_report.py --output outputs/reports/executive_summary.pdf
```

### Python API

```python
from src.models import ARIMAForecaster, RandomForestForecaster
from src.data_processing import load_and_clean_data

# Load data
df = load_and_clean_data("data/raw/warranty_claims.csv")

# Train ARIMA model
arima = ARIMAForecaster(order=(0, 1, 1))
arima.fit(df)

# Generate 12-month forecast
forecast = arima.predict(steps=12)
print(f"FY2025 forecast: RM {forecast.sum():,.0f}")
```

---

## 📊 Data Requirements

### Input Data Format

The system expects a CSV file with the following columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `Claimed Month` | Integer | YYYYMM format | 202404 |
| `Accept/Reject status` | String | Claim decision | "ACC", "REJ", "OVER WARRANTY" |
| `RM` | Float | Claim cost in Ringgit Malaysia | 127.50 |
| `Model` | String | Vehicle model | "MYVI", "AXIA", "BEZZA" |
| `Part Category` | String | Component type | "Alternator", "Starter" |
| `Failure Condition` | String | Issue description (optional) | "No charging" |

**Minimum data requirements:**
- At least 36 months of historical data (3 complete fiscal years)
- At least 10 claims per month on average
- Accepted claims (status = "ACC") only for cost forecasting

### Sample Data

```csv
Claimed Month,Accept/Reject status,RM,Model,Part Category,Failure Condition
202404,ACC,127.50,MYVI,Alternator,No charging
202404,ACC,89.20,AXIA,Alternator,Intermittent charging
202404,REJ,0.00,BEZZA,Alternator,Customer misuse
202405,ACC,234.10,MYVI,Alternator,Complete failure
```

### Data Privacy

**This repository does NOT contain actual Perodua warranty data.**  
All example outputs use synthetic data for demonstration purposes. To use this system with your own data:
1. Place your CSV in `data/raw/`
2. Update `config.yaml` with your column names
3. Run the pipeline

---

## 📈 Results & Validation

### FY2024 Validation Results

**Actual vs Forecast (Annual Totals):**
```
Actual:             RM 32,714
ARIMA Forecast:     RM 35,024  (7.1% error)  ✅ CLOSEST
Random Forest:      RM 42,344  (29.4% error)
XGBoost:            RM 44,161  (35.0% error)
Linear Regression:  RM 65,612  (100.6% error)
```

**Monthly Breakdown:**

| Month | Actual | ARIMA | Random Forest | XGBoost | Linear Reg |
|-------|--------|-------|---------------|---------|------------|
| Apr 2024 | 3,437 | 2,919 | 3,529 | 5,051 | 7,544 |
| May 2024 | 5,266 | 2,919 | 3,529 | 2,957 | 4,964 |
| Jun 2024 | 2,634 | 2,919 | 3,529 | 3,379 | 4,238 |
| Jul 2024 | 3,177 | 2,919 | 3,529 | 3,495 | 5,035 |
| Aug 2024 | 4,624 | 2,919 | 3,529 | 3,238 | 2,704 |
| Sep 2024 | 1,805 | 2,919 | 3,529 | 4,319 | 5,463 |
| Oct 2024 | 2,804 | 2,919 | 3,529 | 3,379 | 4,432 |
| Nov 2024 | 2,497 | 2,919 | 3,529 | 3,719 | 11,511 |
| Dec 2024 | 1,487 | 2,919 | 3,529 | 4,067 | 4,719 |
| Jan 2025 | 2,318 | 2,919 | 3,529 | 3,491 | 5,069 |
| Feb 2025 | 1,504 | 2,919 | 3,529 | 3,688 | 5,001 |
| Mar 2025 | 1,159 | 2,919 | 3,529 | 3,379 | 4,932 |

### Accuracy Metrics Explained

**For C-Level Audiences:**
- **Monthly Accuracy:** What % of the time the model predicted each month correctly
- **Annual Accuracy:** How close the full-year total was to actual budget
- **Directional Accuracy:** Did the model correctly predict "costs going up" vs "costs going down"?

**Technical Definition:**
- Monthly Accuracy = max(0, 100 − MAPE)
- Annual Accuracy = max(0, 100 − |Annual Error %|)
- Directional Accuracy = % of consecutive months with correct direction prediction

---

## 🔮 Future Improvements

### Short-Term (Next Quarter)
- [ ] Implement sequential one-step-ahead ARIMA forecasting (remove constant prediction)
- [ ] Add Holt-Winters Exponential Smoothing as ARIMA alternative
- [ ] Build automated quarterly retraining pipeline
- [ ] Create Streamlit dashboard for non-technical users

### Medium-Term (Next 6 Months)
- [ ] Accumulate FY2025 data and retrain all models
- [ ] Evaluate Prophet (Facebook's time series library)
- [ ] Add external features (vehicle sales volume, supplier changes)
- [ ] Implement confidence intervals on forecasts

### Long-Term (Next Year)
- [ ] Multi-part forecasting (separate models per component)
- [ ] Multi-model forecasting (separate models per vehicle model)
- [ ] Deep learning models (LSTM, Transformer) once 60+ months available
- [ ] Real-time forecast updates as claims arrive

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Reporting Issues
- Use GitHub Issues to report bugs
- Include: Python version, error message, steps to reproduce
- Label: `bug`, `enhancement`, `documentation`

### Pull Requests
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update README.md if adding new functionality

---

## 📊 Project Statistics

- **Lines of Code:** ~5,000 (Python notebooks + modules)
- **Models Trained:** 4 production models × 3 validation iterations = 12 training runs
- **Data Points:** 46 months × ~30 claims/month = ~1,380 raw observations
- **Training Time:** ~2 minutes per model (total: 8 minutes)
- **Forecast Horizon:** 12 months ahead
- **Project Duration:** 3 months (development + validation)

---

**Last Updated:** February 2026  
**Version:** 1.0.0  
