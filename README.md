# NARZY — Financial Risk & Fraud Analytics

**NARZY** is a prototype designed to help analysts identify companies that may be at risk of financial distress using machine learning.

## Problem Statement
Financial distress can be difficult to identify early. NARZY explores how financial data can help analysts prioritize companies for further investigation.

## Current Features
- Upload a bankruptcy dataset in CSV format.
- Train a Random Forest classification model.
- Evaluate accuracy, precision, recall, F1-score, and confusion matrix.
- View important model features.
- Rank test records by predicted bankruptcy risk.
- Adjust a preliminary alert threshold.
- View a preliminary risk alert summary.

## Technology Stack
- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn

## Dataset
The initial prototype uses the UCI Polish Companies Bankruptcy dataset, converted to CSV format.

## How to Run Locally
1. Install Python.
2. Clone this repository.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Start the application:

   `streamlit run app.py`

## Limitations
- Dataset record IDs are not verified company names.
- A stratified random holdout is used in the current baseline; time-aware validation is not yet implemented.
- Feature importance does not establish the cause of an individual prediction.
- The alert threshold is preliminary and requires validation.
- The prototype is for research and prioritization, not proof of fraud or certain bankruptcy.

## Future Improvements
- Time-aware model validation.
- Individual prediction explanations using SHAP.
- Indian company case studies and financial ratios.
- Additional financial manipulation indicators.
- A documented alert review and escalation policy.

**Project status:** Working prototype under development.

