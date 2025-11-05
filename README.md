💡 Insurance Charges Prediction using Linear Regression

This project is an interactive Streamlit web app that predicts medical insurance charges based on user data such as age, BMI, and smoking habits.
It also provides data visualization, outlier detection, and model training features.

🚀 Features

📂 Upload your own insurance.csv dataset

📊 Explore the data (summary statistics & visualizations)

🧹 Remove outliers automatically using the IQR method

🔥 Train a Linear Regression model on selected features

📈 View R² and MSE performance metrics

💾 Save the trained model as a .pkl file

🧮 Optional prediction module (coming soon!)

🧠 Tech Stack

Python 3.10+

Streamlit – for interactive web app

Pandas / NumPy – for data processing

Seaborn / Matplotlib – for visualization

Scikit-learn – for machine learning

Pickle – for model saving

📦 Installation
1️⃣ Clone or download this repository
git clone https://github.com/yourusername/insurance-linear-regression.git
cd insurance-linear-regression

2️⃣ Install the dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit app
streamlit run app.py


Then open the local URL shown in your terminal (usually http://localhost:8501).

📊 Dataset

The app expects a CSV file named insurance.csv containing the following columns:

Column	Description
age	Age of the individual
bmi	Body Mass Index
smoker	Whether the person smokes (yes/no)
charges	Insurance cost (target variable)

📎 Example:

age,bmi,smoker,charges
25,27.9,no,16884.92
33,33.7,yes,37742.58
45,28.5,no,8026.66

🧩 File Structure
.
├── app.py               # Streamlit main application
├── requirements.txt     # Dependencies
├── best_model.pkl       # Saved trained model (after training)
└── README.md            # Project documentation

📈 Model Overview

Model used: Linear Regression

Input features: age, bmi, smoker_yes

Target variable: charges

Evaluation metrics: R² Score and Mean Squared Error (MSE)

💾 Saving the Model

Once trained, the app will automatically save the model as:

best_model.pkl


You can later load it to make predictions or deploy it in a Flask/Streamlit API.

🌐 Deployment (Optional)

You can deploy this app easily using:

Streamlit Cloud

Hugging Face Spaces

Render

👨‍💻 Author

Vishal Ramteke
🎓 Project: Insurance Linear Regression ML App
📅 Created on: November 2025
