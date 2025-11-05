import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import pickle
import io

st.set_page_config(page_title="Insurance Charges Predictor", layout="centered")

st.title("💰 Insurance Charges Prediction App")
st.write("This app uses Linear Regression to predict insurance charges based on Age, BMI, and Smoking Status.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your insurance dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head())

    # Basic information
    st.subheader("🔍 Data Overview")
    st.write(df.describe())

    # Visualizations
    st.subheader("📈 BMI Distribution")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(df['bmi'], kde=True, ax=ax)
    st.pyplot(fig)

    # Outlier removal
    Q1 = df['bmi'].quantile(0.25)
    Q3 = df['bmi'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['bmi'] >= lower_bound) & (df['bmi'] <= upper_bound)]

    # One hot encoding
    df = pd.get_dummies(df, drop_first=True)

    # Feature and target
    if 'charges' not in df.columns:
        st.error("⚠️ The dataset must include a 'charges' column for prediction.")
    else:
        X = df[["age", "bmi", "smoker_yes"]]
        y = df["charges"]

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

        # Model training
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        st.success(f"✅ Model trained successfully!")
        st.write(f"**R² Score:** {r2:.3f}")
        st.write(f"**Mean Squared Error:** {mse:.2f}")

        # Save model
        with open("best_model.pkl", "wb") as f:
            pickle.dump(model, f)

        st.download_button("📥 Download Model", data=open("best_model.pkl", "rb"), file_name="best_model.pkl")

        st.divider()
        st.subheader("🔮 Make Predictions")

        # User Inputs
        age = st.slider("Select Age", 18, 100, 30)
        bmi = st.number_input("Enter BMI", 10.0, 50.0, 25.0)
        smoker = st.selectbox("Smoker?", ["No", "Yes"])

        smoker_yes = 1 if smoker == "Yes" else 0
        input_data = np.array([[age, bmi, smoker_yes]])

        if st.button("Predict Insurance Charges"):
            prediction = model.predict(input_data)[0]
            st.success(f"💵 Estimated Insurance Charges: **${prediction:,.2f}**")

else:
    st.info("/content/insurance.csv")
