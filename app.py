import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="CreditWise – Loan Approval", page_icon="🏦", layout="centered")

# ── Constants ─────────────────────────────────────────────────────────────────
CAT_OHE_COLS   = ["Employment_Status", "Marital_Status", "Loan_Purpose",
                   "Property_Area", "Gender", "Employer_Category"]
CAT_LE_COLS    = ["Education_Level"]
TARGET_COL     = "Loan_Approved"
DROP_COLS      = ["Applicant_ID", "Applicant_Income"]   # dropped in notebook

EMPLOYMENT_OPTIONS   = ["Salaried", "Self-employed", "Contract", "Unemployed"]
MARITAL_OPTIONS      = ["Married", "Single"]
LOAN_PURPOSE_OPTIONS = ["Personal", "Car", "Home", "Business", "Education"]
PROPERTY_OPTIONS     = ["Urban", "Semiurban", "Rural"]
EDUCATION_OPTIONS    = ["Graduate", "Not Graduate"]
GENDER_OPTIONS       = ["Male", "Female"]
EMPLOYER_OPTIONS     = ["Private", "Government", "MNC", "Business", "Unemployed"]
LOAN_TERM_OPTIONS    = [12, 24, 36, 48, 60, 72, 84]

# ── Model training (cached so it only runs once) ──────────────────────────────
@st.cache_resource
def train_model():
    df = pd.read_csv("loan_approval_data.csv")

    # ── Impute ────────────────────────────────────────────────────────────────
    num_cols = df.select_dtypes(include=["float64"]).columns
    cat_cols = df.select_dtypes(include="object").columns

    num_imp = SimpleImputer(strategy="mean")
    df[num_cols] = num_imp.fit_transform(df[num_cols])

    cat_imp = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = cat_imp.fit_transform(df[cat_cols])

    # ── Drop unused columns ───────────────────────────────────────────────────
    df = df.drop(columns=DROP_COLS, errors="ignore")

    # ── OHE ───────────────────────────────────────────────────────────────────
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = ohe.fit_transform(df[CAT_OHE_COLS])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(CAT_OHE_COLS), index=df.index)
    df = pd.concat([df.drop(columns=CAT_OHE_COLS), encoded_df], axis=1)

    # ── Label-encode Education_Level & target ─────────────────────────────────
    le_edu = LabelEncoder()
    le_tgt = LabelEncoder()
    df["Education_Level"] = le_edu.fit_transform(df["Education_Level"])
    df[TARGET_COL]        = le_tgt.fit_transform(df[TARGET_COL])

    # ── Split & scale ─────────────────────────────────────────────────────────
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    acc = model.score(X_test_s, y_test)

    return model, scaler, ohe, le_edu, le_tgt, X.columns.tolist(), acc


model, scaler, ohe, le_edu, le_tgt, feature_cols, accuracy = train_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .metric-card {
        background: #f8f9fb;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.5rem;
        border: 1px solid #e3e6ea;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏦 CreditWise")
st.markdown("#### Instant Loan Approval Predictor")
st.caption(f"Logistic Regression · Test accuracy **{accuracy:.1%}**")
st.divider()

st.markdown("### 📋 Key Details")
st.markdown("Fill in the 6 fields below to get an instant decision.")
st.write("")

col1, col2 = st.columns(2, gap="large")

with col1:
    credit_score      = st.number_input("🎯 Credit Score",      min_value=300, max_value=900, value=680, step=1,
                                         help="Your credit score (300–900)")
    loan_amount       = st.number_input("💰 Loan Amount ($)",   min_value=1000, value=15000, step=500,
                                         help="Total loan amount requested")
    savings           = st.number_input("🏦 Savings ($)",       min_value=0, value=10000, step=500,
                                         help="Total savings / liquid assets")

with col2:
    dti_ratio         = st.slider("📊 DTI Ratio",              min_value=0.0, max_value=1.0, value=0.35, step=0.01,
                                   help="Debt-to-Income ratio (lower is better)")
    loan_term         = st.selectbox("📅 Loan Term (months)",  LOAN_TERM_OPTIONS, index=3,
                                      help="Repayment period in months")
    employment_status = st.selectbox("💼 Employment Status",   EMPLOYMENT_OPTIONS,
                                      help="Your current employment type")

st.divider()

# ── Hidden defaults for remaining model features ──────────────────────────────
coapplicant_income = 3000
age                = 35
dependents         = 1
existing_loans     = 1
collateral_value   = 20000
education_level    = "Graduate"
marital_status     = "Married"
loan_purpose       = "Personal"
property_area      = "Urban"
gender             = "Male"
employer_category  = "Private"

if st.button("🔍 Check Eligibility", use_container_width=True, type="primary"):

    # ── Build raw input row ───────────────────────────────────────────────────
    raw = pd.DataFrame([{
        "Coapplicant_Income": coapplicant_income,
        "Age":                age,
        "Dependents":         dependents,
        "Credit_Score":       credit_score,
        "Existing_Loans":     existing_loans,
        "DTI_Ratio":          dti_ratio,
        "Savings":            savings,
        "Collateral_Value":   collateral_value,
        "Loan_Amount":        loan_amount,
        "Loan_Term":          loan_term,
        "Education_Level":    education_level,
        "Employment_Status":  employment_status,
        "Marital_Status":     marital_status,
        "Loan_Purpose":       loan_purpose,
        "Property_Area":      property_area,
        "Gender":             gender,
        "Employer_Category":  employer_category,
    }])

    # ── OHE ───────────────────────────────────────────────────────────────────
    ohe_arr    = ohe.transform(raw[CAT_OHE_COLS])
    ohe_df     = pd.DataFrame(ohe_arr, columns=ohe.get_feature_names_out(CAT_OHE_COLS))
    raw_num    = raw.drop(columns=CAT_OHE_COLS)

    # ── Label-encode Education_Level ──────────────────────────────────────────
    raw_num = raw_num.copy()
    raw_num["Education_Level"] = le_edu.transform(raw_num["Education_Level"])

    # ── Combine & align columns ───────────────────────────────────────────────
    input_df = pd.concat([raw_num.reset_index(drop=True), ohe_df], axis=1)
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)

    # ── Scale & predict ───────────────────────────────────────────────────────
    input_scaled = scaler.transform(input_df)
    pred         = model.predict(input_scaled)[0]
    proba        = model.predict_proba(input_scaled)[0]

    # le_tgt: 0 = No, 1 = Yes  (alphabetical order)
    approved = le_tgt.inverse_transform([pred])[0]
    prob_yes = proba[list(le_tgt.classes_).index("Yes")]

    st.divider()
    if approved == "Yes":
        st.success(f"✅ **Loan Approved** — confidence {prob_yes:.1%}")
    else:
        st.error(f"❌ **Loan Denied** — approval probability {prob_yes:.1%}")

    st.progress(float(prob_yes), text=f"Approval probability: {prob_yes:.1%}")

    # ── Summary metrics ───────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("🎯 Credit Score", credit_score)
    c2.metric("📊 DTI Ratio",    f"{dti_ratio:.0%}")
    c3.metric("💰 Loan Amount",  f"${loan_amount:,}")

    # ── Requirement checks ────────────────────────────────────────────────────
    issues = []

    if credit_score < 650:
        issues.append(("🎯 Credit Score is too low",
                        f"Your score is **{credit_score}**. Lenders typically require **650+**. "
                        "Work on paying bills on time and reducing existing debt."))

    if dti_ratio > 0.43:
        issues.append(("📊 Debt-to-Income ratio is too high",
                        f"Your DTI is **{dti_ratio:.0%}**. The safe threshold is **≤ 43%**. "
                        "Try paying down existing loans before applying."))

    if savings < 5000:
        issues.append(("🏦 Savings are insufficient",
                        f"Your savings are **${savings:,}**. Having at least **$5,000** in liquid "
                        "assets improves approval chances significantly."))

    if loan_amount > savings * 5:
        issues.append(("💰 Loan amount is high relative to savings",
                        f"You are requesting **${loan_amount:,}** but only have **${savings:,}** saved. "
                        "Consider a smaller loan or building more savings first."))

    if employment_status == "Unemployed":
        issues.append(("💼 Employment status is Unemployed",
                        "Lenders require a stable income source. Secure employment before applying."))

    if loan_term <= 12 and loan_amount > 20000:
        issues.append(("📅 Loan term is very short for this amount",
                        f"Repaying **${loan_amount:,}** in **{loan_term} months** results in very high "
                        "monthly payments. Consider a longer term."))

    if issues:
        st.divider()
        st.markdown("### ⚠️ Issues Found")
        for title, detail in issues:
            with st.expander(title, expanded=True):
                st.markdown(detail)
    elif approved == "Yes":
        st.divider()
        st.markdown("### ✅ All Requirements Met")
        st.markdown("Your profile looks strong across all key criteria. Good luck! 🎉")
