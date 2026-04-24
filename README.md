# 🏦 CreditWise — Loan Approval Predictor

A machine learning project that predicts whether a loan application will be approved or denied, with a clean Streamlit web app so anyone can try it without touching any code.

---

## What this project does

Banks and lenders look at a bunch of factors before approving a loan — your credit score, how much debt you already have, your savings, and so on. This project trains a Logistic Regression model on 1,000 real-ish loan applications and then wraps it in a simple web interface where you fill in 6 key fields and get an instant decision.

It also tells you *why* your application might be rejected — not just a yes/no, but actual feedback like "your credit score is too low" or "your debt-to-income ratio is above the safe threshold."

---

## Project structure

```
CreditWise_LoadApprove/
├── loan_approval_data.csv   # The dataset (1000 rows, 20 columns)
├── loan_approve.ipynb       # Jupyter notebook — EDA, preprocessing, model training
├── app.py                   # Streamlit web app
└── requirements.txt         # Python dependencies
```

---

## The dataset

The CSV has 1,000 loan applications with 20 columns covering:

- **Applicant info** — age, gender, marital status, dependents, education level
- **Financial info** — income, co-applicant income, savings, collateral value, existing loans, DTI ratio
- **Loan details** — amount, term, purpose
- **Context** — employment status, employer category, property area
- **Target** — `Loan_Approved` (Yes / No)

Every column has exactly 50 missing values (5% of the data), which we handle with mean imputation for numbers and most-frequent imputation for categories.

---

## How the model works

The notebook walks through the full pipeline:

1. **Load & explore** — check the data shape, missing values, class balance
2. **Visualize** — boxplots and histograms to understand which features actually matter
3. **Clean** — impute missing values, drop `Applicant_Income` (it didn't separate approved vs denied well)
4. **Encode** — OneHotEncoder for 6 nominal categorical columns, LabelEncoder for education level and the target
5. **Scale** — StandardScaler so Logistic Regression doesn't get thrown off by different value ranges
6. **Train** — 80/20 train/test split, Logistic Regression with `max_iter=1000`
7. **Evaluate** — accuracy, precision, recall, F1, confusion matrix

The model hits around **85% accuracy** on the held-out test set.

---

## The web app

The Streamlit app (`app.py`) lets you check loan eligibility without opening the notebook. It trains the model once on startup (cached so it doesn't retrain on every interaction) and then gives you a form with 6 fields:

| Field | What it is |
|---|---|
| 🎯 Credit Score | Your credit score (300–900) |
| 💰 Loan Amount | How much you want to borrow |
| 🏦 Savings | Your liquid assets |
| 📊 DTI Ratio | Debt-to-income ratio (lower is better) |
| 📅 Loan Term | Repayment period in months |
| 💼 Employment Status | Salaried / Self-employed / Contract / Unemployed |

After clicking **Check Eligibility**, you get:
- ✅ Approved or ❌ Denied with a confidence percentage
- A progress bar showing approval probability
- A summary of your key metrics
- A breakdown of any issues found (e.g. "your credit score is below 650")

---

## Running it locally

**1. Clone or download the project**

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Make sure `loan_approval_data.csv` is in the same folder as `app.py`**

**4. Launch the app**
```bash
streamlit run app.py
```

That's it. The app will open in your browser at `http://localhost:8501`.

---

## Requirements

```
streamlit
pandas
numpy
scikit-learn
```

Python 3.8+ should work fine.

---

## What I learned / notes

- Dropping `Applicant_Income` actually helped — the boxplots showed it barely differed between approved and denied applicants, so it was just noise
- The 5% uniform missingness across all columns suggests the dataset was synthetically generated, which is fine for a learning project
- Logistic Regression is a solid baseline here — interpretable, fast, and 85% accuracy is respectable for this kind of tabular classification task
- The requirement checks in the app are rule-based (not model-based), but they align well with what real lenders actually look at

---

## Future ideas

- Try Random Forest or XGBoost and compare accuracy
- Add SHAP values to explain individual predictions
- Let users input all 17 features instead of just 6
- Deploy to Streamlit Cloud so it's accessible without running locally
