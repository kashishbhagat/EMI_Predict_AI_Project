# EMIPredict AI

An end-to-end machine learning platform that predicts **EMI eligibility**
(`Eligible` / `High_Risk` / `Not_Eligible`) and **maximum affordable monthly
EMI** for loan applicants across five lending scenarios (Personal Loan,
Vehicle, Education, Home Appliances, and E-commerce Shopping EMI).

Built on 404,792 cleaned applicant records, with full experiment tracking
via MLflow and an interactive Streamlit application for real-time scoring.

## 🏆 Results

| Task | Best Model | Key Test-Set Metric |
|---|---|---|
| Eligibility classification | XGBoost Classifier | 98.05% accuracy, F1 (macro) 0.915, ROC-AUC 0.996 |
| Max monthly EMI regression | XGBoost Regressor | RMSE ₹619, R² 0.993 |

Both were selected out of **4 candidate models each** (exceeding the
minimum of 3), all trained and logged with MLflow:

- Classification: Logistic Regression, Random Forest, XGBoost, Decision Tree
- Regression: Linear Regression, HistGradientBoosting, XGBoost, Decision Tree

## 📁 Project Structure

```
emipredict/
├── data/
│   ├── emi_prediction_dataset.csv   # raw input data
│   ├── emi_cleaned.csv              # after preprocessing
│   └── emi_features.csv             # after feature engineering
├── src/
│   ├── preprocessing.py             # data cleaning pipeline
│   ├── feature_engineering.py       # derived features + encoding
│   ├── eda.py                       # exploratory analysis & figures
│   └── train_models.py              # model training + MLflow tracking
├── models/
│   ├── best_classifier.pkl
│   ├── best_regressor.pkl
│   ├── scaler.pkl
│   ├── clf_label_encoder.pkl
│   └── model_metadata.json          # metrics for every trained model
├── mlruns/
│   └── mlflow.db                    # MLflow SQLite tracking store
├── reports/
│   ├── figures/                     # EDA chart images
│   └── eda_insights.txt
├── app/
│   ├── Home.py                      # Streamlit entry point
│   ├── utils.py                     # shared model/feature logic
│   └── pages/
│       ├── 1_Data_Exploration.py
│       ├── 2_Predict.py
│       ├── 3_Model_Performance.py
│       └── 4_Data_Explorer.py
└── requirements.txt
```

## 🚀 Running the project

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Re)run the pipeline (optional — outputs are already generated)
```bash
cd src
python preprocessing.py        # data/emi_cleaned.csv
python feature_engineering.py  # data/emi_features.csv
python eda.py                  # reports/figures/*.png
python train_models.py         # models/*.pkl + MLflow logs
```

### 3. Launch the Streamlit app
```bash
streamlit run app/Home.py
```
Then open the printed local URL (default `http://localhost:8501`).

### 4. Browse MLflow experiment tracking (optional)
```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```
Open `http://localhost:5000` to see every training run's params, metrics,
and logged model artifacts, plus the registered `EMIPredict-Classifier`
and `EMIPredict-Regressor` models in the Model Registry.

## 🧹 Data Quality Issues Handled

- `age`, `monthly_salary`, `bank_balance` contained malformed numeric
  strings from an export bug (duplicated decimal points, e.g. `58.0.0`) — repaired via regex-safe parsing.
- `gender` had inconsistent casing (`Male`/`male`/`M`/`MALE`) — standardized.
- `education`, `monthly_rent`, `credit_score`, `bank_balance`,
  `emergency_fund` had ~0.6% missing values — imputed using domain-aware
  logic (e.g. rent=0 for homeowners) or median imputation.
- `credit_score` contained out-of-range values above the valid 300–850
  scale — clipped.
- Continuous financial fields were capped at the 99.5th percentile to
  limit the influence of extreme outliers.
- 8 exact duplicate rows were removed.

## 🧮 Feature Engineering

27 features were engineered on top of the 17 raw numeric/categorical
inputs, including:
- **Affordability ratios**: debt-to-income, expense-to-income,
  obligation-to-income, disposable income (absolute & ratio)
- **Risk scores**: credit risk tier, employment stability score,
  composite affordability risk score
- **Interaction terms**: income × credit, age × experience, tenure × amount
- **Label-encoded categoricals**: gender, marital status, education,
  employment type, company type, house type, existing loans, EMI scenario

## 📊 App Pages

1. **Home** — project overview and key metrics
2. **Data Exploration** — interactive charts (eligibility distribution,
   financial ratios, demographics, correlations)
3. **Predict** — real-time scoring form for a new applicant
4. **Model Performance** — comparison table/charts across all trained
   models plus MLflow tracking details
5. **Data Explorer** — filterable/exportable view of the applicant dataset

## 🌐 Deployment

To deploy on **Streamlit Community Cloud**: push this repository to
GitHub, then point Streamlit Cloud at `app/Home.py` as the main file.
Ensure `data/`, `models/`, and `mlruns/` are included (or regenerate them
via the pipeline scripts as a build step, since the raw CSV is ~72MB —
consider Git LFS if pushing the data file to GitHub).
