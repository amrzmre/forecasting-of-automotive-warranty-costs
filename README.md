# Forecasting of Automotive Warranty Costs

## Project overview

This project forecasts monthly automotive warranty-claim costs to support budgeting and planning. It aggregates accepted claims into a monthly cost series, compares statistical and machine-learning forecasts, and evaluates the FY2024 forecast against the corresponding actual claims.

## Dataset and preprocessing

The notebook uses locally supplied, claim-level warranty CSV exports; it gives no public dataset citation or source URL. The historical training data covers June 2019–March 2024 (46 monthly observations across FY2019 and FY2021–FY2023); FY2020 has no observations. FY2024 validation covers April 2024–March 2025.

The data is filtered to accepted (`ACC`) claims. Rows with missing `RM` amounts are removed, and duplicate `Claim ID` rows are reduced to the first occurrence when that field is present. `Claimed Month` is mapped to an April–March fiscal year. Claims are aggregated by month, including count, total, mean, and standard deviation. Features include fiscal-month and time-trend indicators, a COVID-period flag, cost and count lags, rolling statistics, growth rates, and claim-mix features.

**Target-column note:** the notebook's introductory comments identify `DNMY RM` as the primary target, but the implemented monthly aggregation and model target use `RM`, with monthly `Total_RM` as the forecast target. Confirm which amount is intended before reusing the analysis.

The CSV files are not included in this repository. The notebook's FY2024 local-validation path is `fy24_warranty_claims.csv`; the historical file is selected through an upload prompt in the notebook.

## Methodology

The notebook develops Linear Regression, ARIMA, Random Forest, and XGBoost models. Production forecasting uses ARIMA(0,1,1) and three feature-based models. Forecast features cover 12 fiscal months; the notebook notes that lag features are held static across the forecast horizon. An enhanced linear model is audited but excluded from planning after producing negative forecasts.

The code also contains progressive, time-ordered validation across historical fiscal years. Its output is not saved in the current notebook, so no historical walk-forward metrics are reported here. The available final evaluation is the FY2024 holdout comparison, using RMSE, MAE, MAPE, annual error, and directional accuracy. The notebook cautions against using its in-sample scores for model selection.

## Key results

The FY2024 validation output contains 351 accepted claims and an actual annual total of **RM 32,713.77** (average monthly total RM 2,726.15). Amounts below are reported as printed in the notebook; metrics are rounded there.

| Model | FY2024 forecast (RM) | Annual error* (RM) | Annual error* | RMSE (RM) | MAE (RM) | MAPE | Directional accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| ARIMA(0,1,1) | 35,024 | -2,310 | -7.1% | 1,216.99 | 997.44 | 47.2% | 0.0% |
| Random Forest | 42,344 | -9,630 | -29.4% | 1,445.01 | 1,274.74 | 66.3% | 0.0% |
| XGBoost | 44,161 | -11,448 | -35.0% | 1,741.35 | 1,569.96 | 77.4% | 18.2% |
| Linear Regression | 65,612 | -32,898 | -100.6% | 3,746.40 | 3,111.98 | 150.1% | 54.5% |

\*The notebook defines annual error as actual minus forecast; a negative value means the forecast was higher than actual.

ARIMA was the closest model on annual total and had the lowest monthly MAPE. Its forecast was flat at approximately RM 2,919 per month, however, and its monthly MAPE was 47.2%. It is better supported as an annual planning baseline than as a month-by-month predictor. Actual monthly totals ranged from RM 1,159.13 to RM 5,266.22.

## Visualizations

The notebook displays monthly claim-cost, claim-count, and mean-cost time series, along with forecast-versus-actual and model-comparison plots. These are inline notebook outputs; no separate plot files are present.

## Technologies

- **Python:** notebook metadata identifies a `Python 3` kernel but does not record an exact version.
- **Imported libraries:** pandas, NumPy, Matplotlib, Seaborn, SciPy, statsmodels, scikit-learn, and XGBoost.
- **Notebook environment:** the first data-loading cell imports `google.colab.files`; see the local-execution note below.

No dependency file is currently present. A minimal, unpinned `requirements.txt` for the imported third-party libraries could contain:

```text
ipykernel
matplotlib
numpy
pandas
scikit-learn
scipy
seaborn
statsmodels
xgboost
```

`ipykernel` is included for VS Code kernel selection; the notebook itself does not import it.

## Setup and execution

Clone the repository, or download and extract it from GitHub:

```text
git clone https://github.com/amrzmre/custom-elt-data-engineering-project.git
cd custom-elt-data-engineering-project
```

Create and activate a virtual environment, then install the notebook dependencies:

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ipykernel matplotlib numpy pandas scikit-learn scipy seaborn statsmodels xgboost
```

**Git Bash**

```bash
py -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install ipykernel matplotlib numpy pandas scikit-learn scipy seaborn statsmodels xgboost
```

Open [the notebook](forecasting_of_automotive_warranty_costs.ipynb) in VS Code, choose the `.venv` Python interpreter/kernel, and run its cells in order after making the input CSVs available.

**Local execution caveat:** the first data-loading cell unconditionally imports `google.colab.files` and calls its upload interface. That interface is specific to Google Colab and prevents the notebook from running end-to-end in a standard local VS Code kernel as currently written. The FY2024 validation cell has a local-path option, but this does not remove the earlier Colab dependency. The notebook must be adapted to local file paths before local execution; this README-only change does not modify it.

## Repository structure

```text
.
├── README.md
└── forecasting_of_automotive_warranty_costs.ipynb
```

The `requirements.txt` shown above is a recommended optional file; it has not been added. The required claim CSVs are not present in the repository.

## Limitations and conclusion

The training series is small (46 observations) and has no FY2020 data. The model therefore has limited history for learning seasonality and generalizing to changing claim volumes or costs. Forecast lag features remain static over the 12-month horizon. The FY2024 comparison supports ARIMA as the closest annual estimate, but its monthly error and flat forecast limit its usefulness for monthly decisions.

Before relying on the results, review the `DNMY RM` versus `RM` target discrepancy and the interpretation of directional accuracy: the notebook reports 0% for the flat ARIMA and Random Forest forecasts, which makes that metric difficult to interpret as directional skill. Also note that the intended historical walk-forward validation has no saved results in the current notebook.
