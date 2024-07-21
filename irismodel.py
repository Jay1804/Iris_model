# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_iris

# Load the trained model
def load_model():
    try:
        with open('iris_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is not None:
    # Load the iris dataset for feature names
    iris = load_iris()

    st.title("Iris Flower Prediction App")

    st.write("""
    This app predicts the **Iris flower** type based on user inputs.
    """)

    # Define input fields
    st.sidebar.header("User Input Features")
    def user_input_features():
        try:
            sepal_length = st.sidebar.slider('Sepal length', float(iris.data[:, 0].min()), float(iris.data[:, 0].max()), float(iris.data[:, 0].mean()))
            sepal_width = st.sidebar.slider('Sepal width', float(iris.data[:, 1].min()), float(iris.data[:, 1].max()), float(iris.data[:, 1].mean()))
            petal_length = st.sidebar.slider('Petal length', float(iris.data[:, 2].min()), float(iris.data[:, 2].max()), float(iris.data[:, 2].mean()))
            petal_width = st.sidebar.slider('Petal width', float(iris.data[:, 3].min()), float(iris.data[:, 3].max()), float(iris.data[:, 3].mean()))
            data = {'sepal_length': sepal_length,
                    'sepal_width': sepal_width,
                    'petal_length': petal_length,
                    'petal_width': petal_width}
            features = pd.DataFrame(data, index=[0])
            return features
        except Exception as e:
            st.error(f"Error in input feature processing: {e}")
            return None

    input_df = user_input_features()

    if input_df is not None:
        # Display user inputs
        st.subheader("User Input Features")
        st.write(input_df)

        # Prediction
        try:
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)

            st.subheader("Prediction")
            st.write(iris.target_names[prediction])

            st.subheader("Prediction Probability")
            st.write(prediction_proba)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.error("Model not loaded.")
