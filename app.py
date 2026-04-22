import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Title
st.title("💳 Credit Card Fraud Detection App")

# Load dataset
@st.cache_data
def load_data():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    data = pd.read_csv(url)
    return data

data = load_data()

# Prepare data
X = data.drop("Class", axis=1)
y = data["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_scaled, y)

# UI Input
st.sidebar.header("Enter Transaction Details")

input_data = []
for i in range(1, 6):  # taking first 5 features for simplicity
    val = st.sidebar.number_input(f"Feature V{i}", value=0.0)
    input_data.append(val)

# Pad remaining features with 0
while len(input_data) < X.shape[1]:
    input_data.append(0.0)

input_array = np.array(input_data).reshape(1, -1)
input_scaled = scaler.transform(input_array)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_scaled)

    if prediction[0] == 0:
        st.success("✅ Normal Transaction")
    else:
        st.error("🚨 Fraudulent Transaction")

# Show dataset preview
if st.checkbox("Show Dataset"):
    st.write(data.head())