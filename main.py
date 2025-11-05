import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import pickle

st.set_page_config(page_title="Insurance Linear Regression", layout="wide")

st.title("💡 Insurance Charges Prediction using Linear Regression")
st.write("An interactive Streamlit app for EDA, outlier cleaning, and model training.")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your insurance.csv file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Basic Info
    st.subheader("📊 Dataset Summary")
    st.write(df.describe())

    # Visualization
    st.subheader("BMI Distribution (Before Cleaning)")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df['bmi'], kde=True, ax=ax)
    st.pyplot(fig)

    # Boxplot
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.boxplot(x=df['bmi'], ax=ax2)
    st.pyplot(fig2)

    # Outlier Removal
    Q1, Q3 = df['bmi'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df_cleaned = df[(df['bmi'] >= lower) & (df['bmi'] <= upper)].drop_duplicates()

    st.success(f"Outliers removed! BMI range: {lower:.2f} - {upper:.2f}")
    st.write("Cleaned dataset shape:", df_cleaned.shape)

    # Distribution after cleaning
    st.subheader("BMI Distribution (After Cleaning)")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    sns.histplot(df_cleaned['bmi'], kde=True, ax=ax3)
    st.pyplot(fig3)

    # One Hot Encoding
    df_encoded = pd.get_dummies(df_cleaned, drop_first=True)

    # Correlation Heatmap
    st.subheader("📈 Correlation Heatmap")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm", ax=ax4)
    st.pyplot(fig4)

    # Model Training
    st.subheader("⚙️ Train the Linear Regression Model")
    x = df_encoded[["age", "bmi", "smoker_yes"]]
    y = df_encoded["charges"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_train_pred = model.predict(x_train)

    st.write("**Model Evaluation:**")
    st.write(f"✅ R² (Test): {r2_score(y_test, y_pred):.4f}")
    st.write(f"✅ R² (Train): {r2_score(y_train, y_train_pred):.4f}")
    st.write(f"📉 MSE (Test): {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"📉 MSE (Train): {mean_squared_error(y_train, y_train_pred):.2f}")

    # Save Model
    if st.button("💾 Save Trained Model"):
        with open("best_model.pkl", "wb") as f:
            pickle.dump(model, f)
        st.success("Model saved as `best_model.pkl`!")

else:
    st.info("👆 Please upload the `insurance.csv` file to get started.")
