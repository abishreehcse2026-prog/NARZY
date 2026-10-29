
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score

st.set_page_config(page_title="NARZY", layout="wide")

st.title("NARZY")
st.subheader("AI-Powered Corporate Risk Early-Warning System")

st.warning(
    "Prototype demonstration. Synthetic data is not real company data "
    "and must not be used for financial decisions."
)

st.header("Upload Financial Dataset")
file = st.file_uploader("Upload a CSV file", type=["csv"])

if file is not None:
    data = pd.read_csv(file)
    st.write("Dataset preview")
    st.dataframe(data.head())

    target = st.selectbox(
        "Select the distress target column",
        data.columns
    )

    if st.button("Train Risk Model"):
        features = data.select_dtypes(include="number").drop(
            columns=[target], errors="ignore"
        )
        usable = pd.concat([features, data[target]], axis=1).dropna()

        if features.empty or usable[target].nunique() != 2:
            st.error("Choose a dataset with numeric features and a binary target.")
        else:
            X = usable[features.columns]
            y = usable[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42,
                stratify=y
            )

            model = RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=42
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            st.metric(
                "Accuracy",
                f"{accuracy_score(y_test, predictions):.2%}"
            )
            st.metric(
                "Recall",
                f"{recall_score(y_test, predictions, zero_division=0):.2%}"
            )

            st.subheader("Feature Importance")
            importance = pd.Series(
                model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
            st.bar_chart(importance)

            st.success("Prototype model training completed.")
else:
    st.info("Upload a suitable CSV dataset to begin.")

