import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load data
data = pd.read_csv('creditcard.csv')

# Separate legitimate and fraud transactions
legit = data[data['Class'] == 0]
fraud = data[data['Class'] == 1]

# Undersample legitimate transactions to balance classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split data into training and testing sets
X = data.drop(columns="Class", axis=1)
Y = data["Class"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)  # Added max_iter to prevent convergence warning
model.fit(X_train, Y_train)

# Evaluate model
train_acc = accuracy_score(model.predict(X_train), Y_train)
test_acc = accuracy_score(model.predict(X_test), Y_test)

# Web app
st.title("Credit Card Fraud Detection Model")

# Get input features from user
input_df = st.text_input('Enter all required feature values separated by commas')

submit = st.button("Submit")

if submit:
    try:
        # Split input string into a list of floats
        input_df_splited = [float(x) for x in input_df.split(",")]
        
        # Check if the number of input features matches the number of features in the model
        if len(input_df_splited) != X.shape[1]:
            st.error("Invalid number of features. Please enter {} feature values.".format(X.shape[1]))
        else:
            # Make prediction
            features = np.array([input_df_splited], dtype=np.float32)  # Changed to np.array with shape (1, -1)
            predictions = model.predict(features)

            # Display result
            if predictions[0] == 0:
                st.write("Transaction is Legitimate")
            else:
                st.write("Transaction is Fraud")
    except ValueError:
        st.error("Invalid input. Please enter feature values separated by commas.")