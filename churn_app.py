from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"C:\Users\rutha\OneDrive\Documents\Customer Churn Project\WA_Fn-UseC_-Telco-Customer-Churn (1).csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce') # Convert TotalCharges from string to numeric.
df = df.dropna(subset=['TotalCharges']) # Drop rows where TotalCharges is missing.
df = df.drop(columns=['customerID']) #Drop customerid as it is not needed.
df_model = df.copy()
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
le = LabelEncoder()
for col in binary_cols:
    df_model[col] = le.fit_transform(df_model[col])
df_model = pd.get_dummies(df_model, drop_first=True)
X = df_model.drop('Churn', axis=1)  
y = df_model['Churn']   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
model = RandomForestClassifier()
model.fit(X_train, y_train) 
joblib.dump(model, "churn_model.pkl")
model = joblib.load("churn_model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")
model_columns = joblib.load("model_columns.pkl")
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print(df.groupby('Churn')['tenure'].mean())
print(df['Contract'].value_counts(normalize=True))
print(df.groupby('Contract')['Churn'].value_counts(normalize=True))
print(df.groupby('Churn')['MonthlyCharges'].mean())
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")
st.title("Customer Churn Prediction App")
st.write("Fill the customer info below to check if they are likely to churn:")
gender = st.selectbox("Gender", ["Male", "Female"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
tenure = st.slider("Tenure (months)", 0, 72)
if st.button("Predict"):
    input_dict = {
        'gender': gender,
        'Contract': contract,
        'PaymentMethod': payment_method,
        'InternetService': internet_service,
        'MonthlyCharges': monthly_charges,
        'tenure': tenure,
    }
    input_df = pd.DataFrame([input_dict])
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    for col in input_df.select_dtypes(include='object').columns:
        input_df[col] = LabelEncoder().fit(df[col]).transform(input_df[col])
    input_df = input_df[model_columns]
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    st.subheader("Prediction:")
    if prediction == 1:
        st.error(f"The customer is likely to churn ({probability*100:.1f}%)")
    else:
        st.success(f"The customer is likely to stay ({100 - probability*100:.1f}%)")

    st.caption("Prediction made using a Random Forest model.")
