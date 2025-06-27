import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os
from datetime import datetime, date
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- IMPORTANT: st.set_page_config() MUST BE THE FIRST STREAMLIT COMMAND ---
st.set_page_config(
    page_title="Breast Cancer Prediction AI Assistant", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="💖"
)

# --- Initialize session state for patient database ---
if 'patient_database' not in st.session_state:
    st.session_state.patient_database = pd.DataFrame(columns=[
        'patient_id', 'name', 'age', 'date_of_prediction', 'prediction', 
        'probability', 'risk_level', 'follow_up_date', 'notes', 'contact_info'
    ])

if 'next_patient_id' not in st.session_state:
    st.session_state.next_patient_id = 1

# --- 1. Load Pre-trained Model and Preprocessing Objects ---
@st.cache_resource
def load_breast_cancer_assets():
    model_dir = 'breast_cancer_model_assets'
    model_path = os.path.join(model_dir, 'breast_cancer_prediction_xgb_model.pkl')
    scaler_path = os.path.join(model_dir, 'breast_cancer_scaler.pkl')
    imputer_path = os.path.join(model_dir, 'breast_cancer_imputer.pkl')
    metrics_path = os.path.join(model_dir, 'breast_cancer_model_evaluation_metrics.pkl')

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        imputer = joblib.load(imputer_path)
        metrics = joblib.load(metrics_path)
        return model, scaler, imputer, metrics
    except FileNotFoundError as e:
        st.error(f"Error: Missing model asset file! Please ensure the '{os.path.basename(e.filename)}' file is in the '{model_dir}' directory.")
        st.info("Make sure you've run the breast cancer model training script to save all necessary `.pkl` files.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading breast cancer model assets: {e}")
        st.stop()

# Call load_assets
best_xgb_model_bc, breast_cancer_scaler, breast_cancer_imputer, bc_eval_metrics = load_breast_cancer_assets()

# Get feature names and medians from loaded metrics
FEATURE_NAMES = bc_eval_metrics.get('feature_names', [])
FEATURE_MEDIANS = bc_eval_metrics.get('feature_medians', {})

if not FEATURE_NAMES:
    st.error("Error: Feature names not found in evaluation metrics. Cannot create input fields.")
    st.stop()

# --- Helper Functions ---
def preprocess_bc_input(input_data_df, scaler, imputer, feature_cols):
    processed_df = input_data_df.copy()
    processed_df[:] = imputer.transform(processed_df)
    
    try:
        processed_df = processed_df[feature_cols]
    except KeyError as e:
        st.error(f"Missing or mismatched feature: {e}. Ensure all input fields are correctly named and present.")
        st.stop()
    
    scaled_data = scaler.transform(processed_df)
    return scaled_data

def determine_risk_level(probability):
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.7:
        return "Moderate Risk"
    else:
        return "High Risk"

def add_patient_to_database(name, age, prediction, probability, notes="", contact_info=""):
    risk_level = determine_risk_level(probability)
    
    new_patient = {
        'patient_id': st.session_state.next_patient_id,
        'name': name,
        'age': age,
        'date_of_prediction': datetime.now().strftime('%Y-%m-%d'),
        'prediction': 'Malignant' if prediction == 1 else 'Benign',
        'probability': probability,
        'risk_level': risk_level,
        'follow_up_date': '',
        'notes': notes,
        'contact_info': contact_info
    }
    
    st.session_state.patient_database = pd.concat([
        st.session_state.patient_database, 
        pd.DataFrame([new_patient])
    ], ignore_index=True)
    
    st.session_state.next_patient_id += 1
    return new_patient['patient_id']

def export_patient_data():
    if not st.session_state.patient_database.empty:
        csv = st.session_state.patient_database.to_csv(index=False)
        return csv
    return None

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["Dashboard", "New Prediction", "Patient Database", "Risk Management", "Analytics"]
)

# --- Main Application ---
st.title("💖 Breast Cancer Prediction AI Assistant")
st.markdown("---")

if page == "Dashboard":
    st.header("Model Performance Overview")
    st.markdown("This section displays the performance metrics and visualizations of the trained model.")

    # Key metrics
    st.subheader("Key Evaluation Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Accuracy", f"{bc_eval_metrics['accuracy']:.4f}")
    with col2:
        st.metric("Precision", f"{bc_eval_metrics['precision']:.4f}")
    with col3:
        st.metric("Recall", f"{bc_eval_metrics['recall']:.4f}")
    with col4:
        st.metric("F1-Score", f"{bc_eval_metrics['f1_score']:.4f}")
    with col5:
        st.metric("AUC-ROC", f"{bc_eval_metrics['roc_auc']:.4f}")

    # Charts
    col_charts_1, col_charts_2 = st.columns(2)

    with col_charts_1:
        st.subheader("Confusion Matrix")
        conf_matrix_np = np.array(bc_eval_metrics['conf_matrix'])
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(conf_matrix_np, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Benign (0)', 'Predicted Malignant (1)'],
                    yticklabels=['Actual Benign (0)', 'Actual Malignant (1)'], ax=ax_cm)
        ax_cm.set_xlabel('Predicted Label')
        ax_cm.set_ylabel('True Label')
        ax_cm.set_title('Confusion Matrix')
        st.pyplot(fig_cm)

    with col_charts_2:
        st.subheader("ROC Curve")
        fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
        fpr_np = np.array(bc_eval_metrics['fpr'])
        tpr_np = np.array(bc_eval_metrics['tpr'])
        ax_roc.plot(fpr_np, tpr_np, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {bc_eval_metrics["roc_auc"]:.2f})')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curve')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

    # Feature importance
    if hasattr(best_xgb_model_bc, 'feature_importances_') and FEATURE_NAMES:
        st.subheader("Feature Importance")
        importances = best_xgb_model_bc.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': FEATURE_NAMES, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        fig_fi, ax_fi = plt.subplots(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax_fi)
        ax_fi.set_title('Feature Importance (XGBoost)')
        ax_fi.set_xlabel('Relative Importance')
        ax_fi.set_ylabel('Feature')
        st.pyplot(fig_fi)

elif page == "New Prediction":
    st.header("New Patient Prediction")
    
    # Patient information
    st.subheader("Patient Information")
    col1, col2 = st.columns(2)
    with col1:
        patient_name = st.text_input("Patient Name")
        patient_age = st.number_input("Age", min_value=1, max_value=120, value=50)
    with col2:
        contact_info = st.text_input("Contact Information (Optional)")
        notes = st.text_area("Additional Notes (Optional)")
    
    st.markdown("---")
    st.subheader("Tumor Characteristics")
    
    input_values = {}
    num_cols_in_expander = 3

    # Group features by their type
    feature_groups = {
        "Mean Values (First 10 Features)": FEATURE_NAMES[0:10],
        "Standard Error Values (Next 10 Features)": FEATURE_NAMES[10:20],
        "Worst/Largest Values (Last 10 Features)": FEATURE_NAMES[20:30]
    }

    # Create expanders for each group of features
    for group_name, features_in_group in feature_groups.items():
        with st.expander(f"Enter {group_name}", expanded=False):
            cols = st.columns(num_cols_in_expander)
            for i, feature_name in enumerate(features_in_group):
                with cols[i % num_cols_in_expander]:
                    default_val = FEATURE_MEDIANS.get(feature_name, 0.0)
                    input_values[feature_name] = st.number_input(
                        f"{feature_name.replace('_', ' ').title()}",
                        value=float(f"{default_val:.5g}"),
                        format="%.5g",
                        key=f"input_{feature_name}"
                    )

    st.markdown("---")

    if st.button("Predict Breast Cancer", type="primary"):
        if not patient_name:
            st.error("Please enter patient name before making prediction.")
        else:
            # Create DataFrame from input data
            input_data = pd.DataFrame([input_values])[FEATURE_NAMES]
            
            # Preprocess the input data
            processed_input = preprocess_bc_input(
                input_data,
                breast_cancer_scaler,
                breast_cancer_imputer,
                FEATURE_NAMES
            )
            
            # Make prediction
            prediction = best_xgb_model_bc.predict(processed_input)[0]
            prediction_proba = best_xgb_model_bc.predict_proba(processed_input)[0, 1]
            
            # Add to database
            patient_id = add_patient_to_database(
                patient_name, patient_age, prediction, prediction_proba, notes, contact_info
            )
            
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"**Prediction: Malignant** (Probability: {prediction_proba:.2f})")
                st.write("Based on the provided data, the model indicates a high likelihood of malignancy.")
            else:
                st.success(f"**Prediction: Benign** (Probability: {prediction_proba:.2f})")
                st.write("Based on the provided data, the model indicates a low likelihood of malignancy.")
            
            st.info(f"Patient record created with ID: {patient_id}")
            st.info("Disclaimer: This prediction should be used as a supplementary tool. Always rely on professional medical advice.")

elif page == "Patient Database":
    st.header("Patient Database")
    
    if st.session_state.patient_database.empty:
        st.info("No patients in database yet. Add patients by making predictions.")
    else:
        # Search and filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            search_name = st.text_input("Search by Name")
        with col2:
            filter_prediction = st.selectbox("Filter by Prediction", ["All", "Benign", "Malignant"])
        with col3:
            filter_risk = st.selectbox("Filter by Risk Level", ["All", "Low Risk", "Moderate Risk", "High Risk"])
        
        # Date range filter
        col4, col5 = st.columns(2)
        with col4:
            start_date = st.date_input("Start Date", value=date.today().replace(day=1))
        with col5:
            end_date = st.date_input("End Date", value=date.today())
        
        # Apply filters
        filtered_df = st.session_state.patient_database.copy()
        
        if search_name:
            filtered_df = filtered_df[filtered_df['name'].str.contains(search_name, case=False, na=False)]
        
        if filter_prediction != "All":
            filtered_df = filtered_df[filtered_df['prediction'] == filter_prediction]
        
        if filter_risk != "All":
            filtered_df = filtered_df[filtered_df['risk_level'] == filter_risk]
        
        # Date filter
        filtered_df['date_of_prediction'] = pd.to_datetime(filtered_df['date_of_prediction'])
        filtered_df = filtered_df[
            (filtered_df['date_of_prediction'].dt.date >= start_date) & 
            (filtered_df['date_of_prediction'].dt.date <= end_date)
        ]
        
        # Display results
        st.subheader(f"Found {len(filtered_df)} patients")
        
        if not filtered_df.empty:
            # Display summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Patients", len(filtered_df))
            with col2:
                malignant_count = len(filtered_df[filtered_df['prediction'] == 'Malignant'])
                st.metric("Malignant Cases", malignant_count)
            with col3:
                high_risk_count = len(filtered_df[filtered_df['risk_level'] == 'High Risk'])
                st.metric("High Risk Patients", high_risk_count)
            with col4:
                avg_age = filtered_df['age'].mean()
                st.metric("Average Age", f"{avg_age:.1f}")
            
            # Prepare data for editing - convert date strings to proper format
            display_df = filtered_df.copy()
            
            # Convert follow_up_date to datetime for proper date editing
            display_df['follow_up_date'] = pd.to_datetime(display_df['follow_up_date'], errors='coerce')
            
            # Editable dataframe
            st.subheader("Patient Records")
            edited_df = st.data_editor(
                display_df,
                column_config={
                    "patient_id": st.column_config.NumberColumn("ID", disabled=True),
                    "probability": st.column_config.NumberColumn("Probability", format="%.3f"),
                    "follow_up_date": st.column_config.DateColumn("Follow-up Date"),
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Update database with edits
            if not edited_df.equals(display_df):
                # Convert dates back to strings for storage
                edited_df['follow_up_date'] = edited_df['follow_up_date'].dt.strftime('%Y-%m-%d').fillna('')
                
                # Update the main database with edited records
                for idx, row in edited_df.iterrows():
                    patient_id = row['patient_id']
                    mask = st.session_state.patient_database['patient_id'] == patient_id
                    if mask.any():
                        st.session_state.patient_database.loc[mask] = row
                
                st.success("Patient records updated!")
            
            # Export functionality
            st.subheader("Export Data")
            csv_data = export_patient_data()
            if csv_data:
                st.download_button(
                    label="Download Patient Data as CSV",
                    data=csv_data,
                    file_name=f"patient_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

elif page == "Risk Management":
    st.header("Risk Management Dashboard")
    
    if st.session_state.patient_database.empty:
        st.info("No patients in database yet. Add patients by making predictions.")
    else:
        df = st.session_state.patient_database.copy()
        
        # Risk level distribution
        st.subheader("Risk Level Distribution")
        risk_counts = df['risk_level'].value_counts()
        
        fig_risk = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Patient Risk Distribution",
            color_discrete_map={
                'Low Risk': '#90EE90',
                'Moderate Risk': '#FFD700',
                'High Risk': '#FF6B6B'
            }
        )
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # High-risk patients requiring follow-up
        st.subheader("High-Risk Patients Requiring Attention")
        high_risk_patients = df[df['risk_level'] == 'High Risk'].copy()
        
        if not high_risk_patients.empty:
            st.warning(f"⚠️ {len(high_risk_patients)} high-risk patients require attention!")
            
            # Show high-risk patients
            st.dataframe(
                high_risk_patients[['patient_id', 'name', 'age', 'date_of_prediction', 'probability', 'follow_up_date']],
                use_container_width=True
            )
            
            # Bulk follow-up date setting
            st.subheader("Set Follow-up Dates")
            follow_up_date = st.date_input("Set follow-up date for high-risk patients")
            if st.button("Update Follow-up Dates"):
                st.session_state.patient_database.loc[
                    st.session_state.patient_database['risk_level'] == 'High Risk', 
                    'follow_up_date'
                ] = follow_up_date.strftime('%Y-%m-%d')
                st.success("Follow-up dates updated for high-risk patients!")
        else:
            st.success("✅ No high-risk patients currently in database.")
        
        # Risk trends over time
        st.subheader("Risk Trends Over Time")
        df['date_of_prediction'] = pd.to_datetime(df['date_of_prediction'])
        
        # Group by date and risk level
        risk_timeline = df.groupby(['date_of_prediction', 'risk_level']).size().unstack(fill_value=0)
        
        if not risk_timeline.empty:
            fig_timeline = px.line(
                risk_timeline,
                title="Risk Level Trends Over Time",
                labels={'value': 'Number of Patients', 'date_of_prediction': 'Date'}
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

elif page == "Analytics":
    st.header("Advanced Analytics")
    
    if st.session_state.patient_database.empty:
        st.info("No patients in database yet. Add patients by making predictions.")
    else:
        df = st.session_state.patient_database.copy()
        df['date_of_prediction'] = pd.to_datetime(df['date_of_prediction'])
        
        # Summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_patients = len(df)
            st.metric("Total Patients", total_patients)
        
        with col2:
            malignant_rate = len(df[df['prediction'] == 'Malignant']) / total_patients * 100
            st.metric("Malignant Rate", f"{malignant_rate:.1f}%")
        
        with col3:
            avg_probability = df['probability'].mean()
            st.metric("Avg Probability", f"{avg_probability:.3f}")
        
        with col4:
            high_risk_rate = len(df[df['risk_level'] == 'High Risk']) / total_patients * 100
            st.metric("High Risk Rate", f"{high_risk_rate:.1f}%")
        
        # Age distribution analysis
        st.subheader("Age Distribution by Prediction")
        fig_age = px.histogram(
            df, 
            x='age', 
            color='prediction',
            title="Age Distribution by Prediction Result",
            marginal="box"
        )
        st.plotly_chart(fig_age, use_container_width=True)
        
        # Probability distribution
        st.subheader("Probability Distribution")
        fig_prob = px.histogram(
            df,
            x='probability',
            color='prediction',
            title="Probability Distribution by Prediction Result",
            nbins=20
        )
        st.plotly_chart(fig_prob, use_container_width=True)
        
        # Predictions over time
        st.subheader("Predictions Over Time")
        daily_predictions = df.groupby([df['date_of_prediction'].dt.date, 'prediction']).size().unstack(fill_value=0)
        
        if not daily_predictions.empty:
            fig_daily = px.bar(
                daily_predictions,
                title="Daily Predictions",
                labels={'value': 'Number of Patients', 'date_of_prediction': 'Date'}
            )
            st.plotly_chart(fig_daily, use_container_width=True)
        
        # Risk level by age groups
        st.subheader("Risk Level by Age Groups")
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 70, 100], labels=['<30', '30-50', '50-70', '70+'])
        age_risk = df.groupby(['age_group', 'risk_level']).size().unstack(fill_value=0)
        
        if not age_risk.empty:
            fig_age_risk = px.bar(
                age_risk,
                title="Risk Level Distribution by Age Group",
                labels={'value': 'Number of Patients'},
                barmode='stack'
            )
            st.plotly_chart(fig_age_risk, use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown("💖 **Breast Cancer Prediction AI Assistant** - Always consult with healthcare professionals for medical decisions.")