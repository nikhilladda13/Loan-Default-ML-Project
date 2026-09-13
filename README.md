
# Loan Default Prediction Using Machine Learning

## 1. Project Overview

This project develops a machine learning based loan default risk prediction system. The system uses applicant characteristics to classify an application as a comparatively good or bad credit risk and provides a business recommendation through an interactive Streamlit application.
## Project Links

GitHub Repository:https://github.com/nikhilladda13/Loan-Default-ML-Project.git
Live Streamlit App:https://loan-default-ml-project-i7l5bde7etpzpu7rvxogy8.streamlit.app/

## 2. Business Problem

Banks and lending institutions need to assess credit risk before approving loans. Incorrectly approving a high-risk borrower can lead to financial losses, while rejecting a potentially good borrower can reduce business opportunities.

The objective is to build a classification model that can support credit-risk assessment by estimating the probability that an applicant will be classified as a bad credit risk.

## 3. Dataset

The project uses the **Statlog (German Credit Data)** from the UCI Machine Learning Repository.

- Dataset ID: 144
- Instances: 1,000
- Features: 20
- Task: Binary classification
- Target: Good / Bad credit risk
- Source: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

The dataset contains categorical and integer attributes describing credit applications. The original UCI resource also provides a cost matrix, which is relevant because misclassifying a bad credit customer as good can be more costly than the reverse.

## 4. Methodology

1. Import dataset using `ucimlrepo`
2. Inspect data types and target distribution
3. Handle missing values through preprocessing pipelines
4. Encode categorical variables using one-hot encoding
5. Scale numerical variables
6. Split data into training and testing sets using stratification
7. Train Logistic Regression and Random Forest
8. Evaluate using Accuracy, Precision, Recall, F1 and ROC-AUC
9. Select the best model using ROC-AUC with recall as a secondary business consideration
10. Save the selected pipeline as `models/loan_default_model.pkl`
11. Deploy predictions through Streamlit

## 5. How to Run

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The first command installs dependencies. The second downloads the UCI data, trains the models and creates the model file. The third starts the Streamlit application.

## 6. Business Interpretation

The application groups predicted risk into lower, medium and high risk bands. The output is accompanied by suggested actions such as enhanced verification, affordability checks, conservative limits or normal underwriting.

These recommendations are decision-support guidance only. A real lending institution should combine model output with regulatory requirements, internal policies, affordability checks, human review and explainability/fairness controls.

## 7. Model Performance

The exact performance metrics are generated automatically by `train_model.py` and stored in `models/model_metrics.json`. This avoids inventing performance figures and ensures the README/report reflects the actual train/test split used.

## 8. Project Structure

```text
Loan_Default_ML_Project/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── report.pdf
├── data/
│   └── german_credit_clean.csv
├── models/
│   ├── loan_default_model.pkl
│   └── model_metrics.json
├── notebooks/
│   └── Loan_Default_Analysis.ipynb
└── pages/
    ├── 1_Business_Problem.py.txt
    ├── pages2_Data_Insights.py.txt
    └── pages3_Prediction.py.txt
