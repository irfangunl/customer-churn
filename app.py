import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Telco Customer Churn Dashboard", page_icon="📉", layout="wide")


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)
    return df


def build_model(df: pd.DataFrame):
    model_df = df.drop(columns=["customerID", "Churn", "ChurnFlag"])
    y = df["ChurnFlag"]

    categorical_cols = model_df.select_dtypes(include=["object", "str"]).columns.tolist()
    numeric_cols = model_df.select_dtypes(exclude=["object", "str"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        model_df,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "report": classification_report(y_test, pred, output_dict=True),
        "cm": confusion_matrix(y_test, pred),
        "X_test": X_test,
        "y_test": y_test,
        "pred": pred,
        "proba": proba,
    }

    return clf, metrics


def risk_segments(df: pd.DataFrame, score_col: str = "ChurnScore") -> pd.DataFrame:
    seg = (
        df.groupby(["Contract", "InternetService", "PaymentMethod"], as_index=False)
        .agg(
            CustomerCount=("customerID", "count"),
            AvgRisk=(score_col, "mean"),
            ChurnRate=("ChurnFlag", "mean"),
            AvgMonthlyCharges=("MonthlyCharges", "mean"),
        )
        .sort_values(["AvgRisk", "CustomerCount"], ascending=[False, False])
    )
    seg["AvgRiskPct"] = seg["AvgRisk"] * 100
    seg["ChurnRatePct"] = seg["ChurnRate"] * 100
    return seg


def generate_actions(row: pd.Series) -> list[str]:
    actions = []
    if row["Contract"] == "Month-to-month":
        actions.append("12 ay kontrat gecis indirimi")
    if row["MonthlyCharges"] > 80:
        actions.append("Fatura dengeleme / paket optimizasyonu")
    if row.get("TechSupport", "No") == "No":
        actions.append("Ilk 3 ay ucretsiz teknik destek")
    if row.get("OnlineSecurity", "No") == "No":
        actions.append("Guvenlik paketi promosyonu")
    if row["tenure"] <= 12:
        actions.append("Ilk yil hosgeldin sadakat kampanyasi")
    if not actions:
        actions.append("Memnuniyet aramasi ve capraz satis teklifi")
    return actions


def main():
    st.title("Telco Customer Churn - Analiz ve Tahmin Uygulamasi")
    st.caption("Sorun tespiti + veri analizi + tahminleme + aksiyon planlama")

    df = load_data("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    with st.sidebar:
        st.header("Filtreler")
        contract_filter = st.multiselect(
            "Contract",
            options=sorted(df["Contract"].dropna().unique().tolist()),
            default=sorted(df["Contract"].dropna().unique().tolist()),
        )
        internet_filter = st.multiselect(
            "InternetService",
            options=sorted(df["InternetService"].dropna().unique().tolist()),
            default=sorted(df["InternetService"].dropna().unique().tolist()),
        )
        payment_filter = st.multiselect(
            "PaymentMethod",
            options=sorted(df["PaymentMethod"].dropna().unique().tolist()),
            default=sorted(df["PaymentMethod"].dropna().unique().tolist()),
        )

    filtered = df[
        df["Contract"].isin(contract_filter)
        & df["InternetService"].isin(internet_filter)
        & df["PaymentMethod"].isin(payment_filter)
    ].copy()

    model, m = build_model(df)
    scored = df.copy()
    model_input = scored.drop(columns=["customerID", "Churn", "ChurnFlag"])
    scored["ChurnScore"] = model.predict_proba(model_input)[:, 1]

    problem_tab, analysis_tab, solution_tab = st.tabs([
        "1) Problem Tanimi",
        "2) Analiz ve Model",
        "3) Cozum ve Simulasyon",
    ])

    with problem_tab:
        st.subheader("Is Problemi")
        st.write(
            "Amaç: Müşteri kaybını azaltmak. Churn, gelir kaybına ve yeni müşteri edinme maliyetlerinin artmasına neden oluyor."
        )

        p1, p2, p3 = st.columns(3)
        base_churn = df["ChurnFlag"].mean() * 100
        p1.metric("Toplam Musteri", f"{len(df):,}")
        p2.metric("Mevcut Churn Orani", f"%{base_churn:.2f}")
        p3.metric("Aylik Ortalama Gelir", f"{df['MonthlyCharges'].mean():.2f}")

        top_contract = (
            df.groupby("Contract", as_index=False)["ChurnFlag"].mean().sort_values("ChurnFlag", ascending=False)
        )
        top_contract["ChurnPct"] = top_contract["ChurnFlag"] * 100
        fig_problem = px.bar(
            top_contract,
            x="Contract",
            y="ChurnPct",
            title="Kok Neden Sinyali: Sozlesme Tipine Gore Churn",
            text_auto=".2f",
        )
        st.plotly_chart(fig_problem, width="stretch")
        st.info(
            "Gozlem: Month-to-month musterilerde churn daha yuksek. Bu segment, oncelikli aksiyon alani olarak secilebilir."
        )

    with analysis_tab:
        c1, c2, c3, c4 = st.columns(4)
        churn_rate = filtered["ChurnFlag"].mean() if len(filtered) else 0
        c1.metric("Kayit Sayisi", f"{len(filtered):,}")
        c2.metric("Churn Orani", f"%{churn_rate * 100:.2f}")
        c3.metric("Ortalama Tenure", f"{filtered['tenure'].mean():.1f}" if len(filtered) else "0")
        c4.metric(
            "Ortalama Aylik Ucret",
            f"{filtered['MonthlyCharges'].mean():.2f}" if len(filtered) else "0",
        )

        st.subheader("Veri Gorunumu")
        st.dataframe(filtered.head(50), width="stretch")

        g1, g2 = st.columns(2)
        with g1:
            pie_data = filtered["Churn"].value_counts().rename_axis("Churn").reset_index(name="Count")
            fig_pie = px.pie(pie_data, names="Churn", values="Count", title="Churn Dagilimi")
            st.plotly_chart(fig_pie, width="stretch")

        with g2:
            contract_churn = (
                filtered.groupby("Contract", as_index=False)["ChurnFlag"].mean().sort_values("ChurnFlag", ascending=False)
            )
            contract_churn["ChurnPct"] = contract_churn["ChurnFlag"] * 100
            fig_contract = px.bar(
                contract_churn,
                x="Contract",
                y="ChurnPct",
                title="Sozlesmeye Gore Churn Orani (%)",
                text_auto=".2f",
            )
            st.plotly_chart(fig_contract, width="stretch")

        g3, g4 = st.columns(2)
        with g3:
            fig_tenure = px.histogram(
                filtered,
                x="tenure",
                color="Churn",
                barmode="overlay",
                nbins=30,
                title="Tenure Dagilimi (Churn Bazli)",
            )
            st.plotly_chart(fig_tenure, width="stretch")

        with g4:
            fig_charges = px.box(
                filtered,
                x="Churn",
                y="MonthlyCharges",
                color="Churn",
                title="Monthly Charges - Churn Karsilastirmasi",
            )
            st.plotly_chart(fig_charges, width="stretch")

        st.subheader("Model Sonuclari (Random Forest)")

        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{m['accuracy']:.4f}")
        m2.metric("ROC-AUC", f"{m['roc_auc']:.4f}")
        m3.metric("Test Kayit Sayisi", f"{len(m['y_test'])}")

        cm = m["cm"]
        cm_df = pd.DataFrame(cm, index=["Actual No", "Actual Yes"], columns=["Pred No", "Pred Yes"])
        fig_cm = px.imshow(cm_df, text_auto=True, title="Confusion Matrix")
        st.plotly_chart(fig_cm, width="stretch")

        report_df = pd.DataFrame(m["report"]).transpose().round(3)
        st.write("Classification Report")
        st.dataframe(report_df, width="stretch")

    with solution_tab:
        st.subheader("Aksiyonlanabilir Cozum")
        st.write(
            "Model ile yuksek riskli musteriler belirlenir, ilgili segmente uygun retention kampanyasi atanir ve beklenen finansal etki simule edilir."
        )

        seg = risk_segments(scored)
        fig_seg = px.bar(
            seg.head(10),
            x="Contract",
            y="AvgRiskPct",
            color="InternetService",
            hover_data=["PaymentMethod", "CustomerCount", "ChurnRatePct"],
            title="En Riskli 10 Segment (Model Skoruna Gore)",
        )
        st.plotly_chart(fig_seg, width="stretch")

        high_risk_threshold = st.slider("Yuksek risk esigi", min_value=0.40, max_value=0.90, value=0.60, step=0.01)
        campaign_cost = st.number_input("Musteri basi kampanya maliyeti", min_value=0.0, value=15.0, step=1.0)
        expected_save_rate = st.slider("Kampanya basari orani", min_value=0.05, max_value=0.80, value=0.30, step=0.05)

        high_risk = scored[scored["ChurnScore"] >= high_risk_threshold].copy()
        high_risk = high_risk.sort_values("ChurnScore", ascending=False)
        high_risk["ActionPlan"] = high_risk.apply(lambda r: " | ".join(generate_actions(r)), axis=1)

        targeted_count = len(high_risk)
        expected_saved = int(targeted_count * expected_save_rate)
        avg_rev = high_risk["MonthlyCharges"].mean() if targeted_count else 0
        gross_saved_revenue = expected_saved * avg_rev
        campaign_total_cost = targeted_count * campaign_cost
        net_impact = gross_saved_revenue - campaign_total_cost

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Hedeflenen Musteri", f"{targeted_count:,}")
        s2.metric("Beklenen Kurtarilan", f"{expected_saved:,}")
        s3.metric("Tahmini Gelir Koruma", f"{gross_saved_revenue:,.2f}")
        s4.metric("Net Etki", f"{net_impact:,.2f}")

        st.write("Oncelikli Musteriler ve Onerilen Aksiyon")
        show_cols = [
            "customerID",
            "ChurnScore",
            "Contract",
            "tenure",
            "MonthlyCharges",
            "ActionPlan",
        ]
        st.dataframe(high_risk[show_cols].head(50), width="stretch")

    st.subheader("Tek Musteri Icin Churn Olasiligi")
    default_row = df.drop(columns=["customerID", "Churn", "ChurnFlag"]).iloc[0].to_dict()

    with st.form("predict_form"):
        input_data = {}
        for col, val in default_row.items():
            if isinstance(val, (int, float)) and col != "SeniorCitizen":
                input_data[col] = st.number_input(col, value=float(val))
            elif col == "SeniorCitizen":
                input_data[col] = st.selectbox(col, options=[0, 1], index=int(val))
            else:
                options = sorted(df[col].dropna().astype(str).unique().tolist())
                if str(val) in options:
                    idx = options.index(str(val))
                else:
                    idx = 0
                input_data[col] = st.selectbox(col, options=options, index=idx)

        submitted = st.form_submit_button("Tahmin Et")

    if submitted:
        input_df = pd.DataFrame([input_data])
        input_df["TotalCharges"] = pd.to_numeric(input_df["TotalCharges"], errors="coerce")
        churn_prob = model.predict_proba(input_df)[:, 1][0]
        churn_pred = int(churn_prob >= 0.5)
        st.success(f"Tahmini Churn Olasiligi: %{churn_prob * 100:.2f}")
        st.info(f"Model Sinifi: {'Yes' if churn_pred == 1 else 'No'}")


if __name__ == "__main__":
    main()
