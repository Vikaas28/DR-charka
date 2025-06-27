import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# --- IMPORTANT: st.set_page_config() MUST BE THE FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="Diabetes Prediction AI Assistant", layout="wide", initial_sidebar_state="expanded")

# --- 1. Load Pre-trained Model, Scaler, Medians, and Evaluation Metrics ---
@st.cache_resource # Cache the model loading to improve performance
def load_assets():
    try:
        model = joblib.load('diabetes_prediction_xgb_model.pkl')
        scaler = joblib.load('diabetes_scaler.pkl')
        medians = joblib.load('medians_for_imputation.pkl')
        metrics = joblib.load('model_evaluation_metrics.pkl')
        return model, scaler, medians, metrics
    except FileNotFoundError:
        st.error("Error: One or more model files not found! Please ensure 'diabetes_prediction_xgb_model.pkl', 'diabetes_scaler.pkl', 'medians_for_imputation.pkl', and 'model_evaluation_metrics.pkl' are in the same directory as this 'app.py' file.")
        raise FileNotFoundError("Required model files are missing.") # Raise an exception to stop execution gracefully
    except Exception as e:
        st.error(f"An unexpected error occurred while loading model assets: {e}")
        raise # Re-raise the exception to stop the app

# Call load_assets and handle the potential exception
try:
    model, scaler, medians_for_imputation, eval_metrics = load_assets()
except (FileNotFoundError, Exception):
    st.stop() # Ensure Streamlit app stops if loading fails


st.title("🩺 Diabetes Prediction AI Assistant")
st.markdown("---")

# Sidebar for navigation
st.sidebar.header("Navigation")
page_selection = st.sidebar.radio("Go to", ["Dashboard & Model Performance", "Make a New Prediction"])

# --- Helper Function for Input Preprocessing ---
def preprocess_input(input_data_df, scaler, medians_for_imputation):
    # Ensure column order matches training data
    feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_data_df = input_data_df[feature_cols] # Reorder columns if necessary

    # Handle 0 values with loaded medians
    cols_to_impute_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_impute_zero:
        if input_data_df[col].iloc[0] == 0:
            input_data_df[col] = medians_for_imputation[col]

    # Scale the input data
    scaled_data = scaler.transform(input_data_df)
    return scaled_data

# --- Dashboard Section ---
if page_selection == "Dashboard & Model Performance":
    st.header("Model Performance Overview")
    st.markdown("This section displays the performance metrics and visualizations of the trained model.")

    st.subheader("Key Evaluation Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Accuracy", f"{eval_metrics['accuracy']:.4f}")
    with col2:
        st.metric("Precision", f"{eval_metrics['precision']:.4f}")
    with col3:
        st.metric("Recall", f"{eval_metrics['recall']:.4f}")
    with col4:
        st.metric("F1-Score", f"{eval_metrics['f1_score']:.4f}")
    with col5:
        st.metric("AUC-ROC", f"{eval_metrics['roc_auc']:.4f}")

    st.markdown("---")

    col_charts_1, col_charts_2 = st.columns(2)

    with col_charts_1:
        st.subheader("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(eval_metrics['conf_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Non-Diabetic', 'Predicted Diabetic'],
                    yticklabels=['Actual Non-Diabetic', 'Actual Diabetic'], ax=ax_cm)
        ax_cm.set_xlabel('Predicted Label')
        ax_cm.set_ylabel('True Label')
        ax_cm.set_title('Confusion Matrix')
        st.pyplot(fig_cm)

    with col_charts_2:
        st.subheader("Receiver Operating Characteristic (ROC) Curve")
        fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
        ax_roc.plot(eval_metrics['fpr'], eval_metrics['tpr'], color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {eval_metrics["roc_auc"]:.2f})')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curve')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

    st.markdown("---")

    # Feature Importance (if model supports it)
    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance")
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                         'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax_fi)
        ax_fi.set_title('Feature Importance (XGBoost)')
        ax_fi.set_xlabel('Relative Importance')
        ax_fi.set_ylabel('Feature')
        st.pyplot(fig_fi)
    else:
        st.info("Feature importance plot is not available for the selected model type.")

# --- Prediction Section ---
elif page_selection == "Make a New Prediction":
    st.header("Predict Diabetes Status")
    st.markdown("Enter patient's medical details below to get a diabetes prediction.")

    # Input fields for patient data
    st.subheader("Patient Medical Data")
    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, value=1, step=1)
        glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=200, value=120, step=1)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=122, value=70, step=1)

    with col2:
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=99, value=25, step=1)
        insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=846, value=80, step=1)
        bmi = st.number_input("BMI", min_value=0.0, max_value=67.1, value=25.0, step=0.1)

    with col3:
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.078, max_value=2.42, value=0.4, step=0.001, format="%.3f")
        age = st.number_input("Age (years)", min_value=21, max_value=81, value=30, step=1)

    st.markdown("---")

    if st.button("Predict Diabetes"):
        # Create a DataFrame from input data
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [dpf],
            'Age': [age]
        })

        # Preprocess the input data
        processed_input = preprocess_input(input_data, scaler, medians_for_imputation)

        # Make prediction
        prediction = model.predict(processed_input)[0]
        prediction_proba = model.predict_proba(processed_input)[0, 1] # Probability of being diabetic

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"**Prediction: Diabetic** (Probability: {prediction_proba:.2f})")
            st.write("Based on the provided data, the model indicates a high likelihood of diabetes. Further medical consultation and tests are strongly recommended.")
        else:
            st.success(f"**Prediction: Non-Diabetic** (Probability: {prediction_proba:.2f})")
            st.write("Based on the provided data, the model indicates a low likelihood of diabetes. Continue to monitor health as advised by your doctor.")

        st.info("Disclaimer: This prediction is generated by a machine learning model and should be used as a supplementary tool. Always rely on professional medical advice and clinical judgment for diagnosis and treatment.")