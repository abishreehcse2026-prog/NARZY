
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

st.set_page_config(
    page_title="NARZY | Risk Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("NARZY")
st.subheader("AI-Powered Financial Risk and Fraud Analytics")
st.write("An early-warning prototype for corporate financial distress.")

uploaded_file = st.file_uploader(
    "Upload bankruptcy dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset uploaded successfully!")

        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")

        target_options = list(df.columns)
        default_target = (
            target_options.index("class")
            if "class" in target_options
            else len(target_options) - 1
        )

        target = st.selectbox(
            "Select the bankruptcy label column",
            target_options,
            index=default_target
        )

        if st.button("Train Risk Model"):
            data = df.copy()
            data[target] = pd.to_numeric(data[target], errors="coerce")
            data = data.dropna(subset=[target])

            X = data.drop(columns=[target])
            X = X.apply(pd.to_numeric, errors="coerce")
            X = X.replace([np.inf, -np.inf], np.nan)

            valid = data[target].isin([0, 1])
            X = X.loc[valid]
            y = data.loc[valid, target].astype(int)

            if y.nunique() != 2 or y.value_counts().min() < 2:
                st.error(
                    "The selected label must contain both 0 and 1, "
                    "with at least two examples of each."
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=0.2,
                    random_state=42,
                    stratify=y
                )

                model = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("forest", RandomForestClassifier(
                        n_estimators=200,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1
                    ))
                ])

                with st.spinner("Training model..."):
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)

                st.subheader("Model Evaluation")
                st.caption(
                    "Results are from a stratified random holdout. "
                    "This is an initial baseline, not time-aware validation."
                )

                c1, c2, c3, c4 = st.columns(4)
                c1.metric(
                    "Accuracy",
                    f"{accuracy_score(y_test, predictions):.3f}"
                )
                c2.metric(
                    "Precision",
                    f"{precision_score(y_test, predictions, zero_division=0):.3f}"
                )
                c3.metric(
                    "Recall",
                    f"{recall_score(y_test, predictions, zero_division=0):.3f}"
                )
                c4.metric(
                    "F1-score",
                    f"{f1_score(y_test, predictions, zero_division=0):.3f}"
                )

                st.write("Label distribution (0 = non-bankrupt, 1 = bankrupt):")
                st.write(y.value_counts().sort_index())

                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, predictions, labels=[0, 1])
                st.dataframe(
                    pd.DataFrame(
                        cm,
                        index=["Actual 0", "Actual 1"],
                        columns=["Predicted 0", "Predicted 1"]
                    )
                )

                st.subheader("Most Important Features")
                importances = pd.Series(
                    model.named_steps["forest"].feature_importances_,
                    index=X.columns
                ).sort_values(ascending=False).head(10)

                st.bar_chart(importances)
                
                st.subheader("Company Risk Ranking")

                risk_scores = model.predict_proba(X_test)[:, 1]

                risk_table = pd.DataFrame({
                      "Record ID": X_test.index.astype(str),
                      "Bankruptcy Risk (%)": (risk_scores * 100).round(2),
                      "Risk Status": np.where(
                          risk_scores >= 0.5,
                          "High Risk",
                          "Lower Risk"
                      )
                  })
                
                  st.subheader("Why Is This Record Flagged?")

                  selected_record = st.selectbox(
                      "Choose a record to investigate",
                      risk_table["Record ID"].tolist()
                  )

                  selected_index = int(selected_record)
                  selected_features = X.loc[selected_index]

                  feature_importance = pd.Series(
                      model.named_steps["forest"].feature_importances_,
                      index=X.columns
                  ).sort_values(ascending=False)

                  st.write(
                      "The following are the model's most important "
                      "features for this dataset. They are not proof "
                      "of the reason for an individual prediction."
                  )

                  explanation = pd.DataFrame({
                      "Feature": feature_importance.head(3).index,
                      "Record Value": [
                          selected_features.get(feature, np.nan)
                          for feature in feature_importance.head(3).index
                      ],
                      "Overall Model Importance": [
                          round(value, 4)
                          for value in feature_importance.head(3).values
                      ]
                  })

                  st.dataframe(
                      explanation,
                      use_container_width=True,
                      hide_index=True
                  )


                risk_table = risk_table.sort_values(
                      by="Bankruptcy Risk (%)",
                      ascending=False
                  )
                
                st.subheader("Risk Alert Summary")

                high_risk = int((risk_scores >= 0.5).sum())
                lower_risk = int((risk_scores < 0.5).sum())

                col1, col2 = st.columns(2)

                col1.metric("High-Risk Records", high_risk)
                col2.metric("Lower-Risk Records", lower_risk)


                risk_table.insert(
                      0,
                      "Rank",
                      range(1, len(risk_table) + 1)
                  )

                st.write(
                      "Records ranked by predicted bankruptcy risk. "
                      "These are holdout test records, not verified "
                      "company-level predictions."
                  )

                st.dataframe(
                      risk_table,
                      use_container_width=True,
                      hide_index=True
                  )


                st.warning(
                    "This baseline uses financial records, not identified "
                    "company names. Predictions are research signals, "
                    "not proof of fraud or certain default."
                )

    except Exception as e:
        st.error(f"Could not process this dataset: {e}")

st.divider()
st.caption("NARZY | Financial risk screening prototype")




