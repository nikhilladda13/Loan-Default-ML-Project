
# Submission & Deployment Checklist

## Before submission
- [ ] Run `pip install -r requirements.txt`
- [ ] Run `python train_model.py`
- [ ] Confirm `models/loan_default_model.pkl` exists
- [ ] Confirm `models/model_metrics.json` exists
- [ ] Open and run `notebooks/Loan_Default_Analysis.ipynb`
- [ ] Review the generated metrics and update the report if desired
- [ ] Run `streamlit run app.py`
- [ ] Test several applicant profiles
- [ ] Create GitHub repository and upload the complete project
- [ ] Deploy `app.py` on Streamlit Community Cloud
- [ ] Add GitHub and Streamlit links to the final submission

## Viva points
1. Why loan default prediction?
2. Why classification?
3. Why Logistic Regression and Random Forest?
4. Why are recall and ROC-AUC important?
5. What is a false negative in credit risk?
6. Why should the model not be the sole basis for loan approval?
7. What are the limitations of the dataset?
8. How could the model be improved?
