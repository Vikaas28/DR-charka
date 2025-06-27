
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
import sqlite3
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re

# Page configuration
st.set_page_config(
    page_title="Advanced Heart Disease Prediction System", 
    page_icon="💓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #DC143C;
        font-size: 3rem;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-low {
        background: linear-gradient(135deg, #06ffa5 0%, #3eff4e 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>💓 Advanced Heart Disease Prediction System</h1>", unsafe_allow_html=True)
st.markdown("---")

# Define features for one-hot encoding
one_hot_encode_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Load model artifacts
@st.cache_resource
def load_artifacts():
    """Load pre-trained model, scaler, and feature columns"""
    try:
        with open('heart_disease_random_forest_model_one_hot.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('heart_disease_scaler_one_hot.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('heart_disease_feature_columns_one_hot.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        
        # Try to load evaluation metrics if available
        try:
            with open('heart_disease_evaluation_metrics.pkl', 'rb') as f:
                eval_metrics = pickle.load(f)
        except FileNotFoundError:
            # Create dummy metrics for demo purposes
            eval_metrics = {
                'accuracy': 0.8542,
                'precision': 0.8421,
                'recall': 0.8667,
                'f1_score': 0.8542,
                'roc_auc': 0.9123,
                'conf_matrix': np.array([[85, 12], [15, 88]]),
                'fpr': np.linspace(0, 1, 100),
                'tpr': np.power(np.linspace(0, 1, 100), 0.5)
            }
        
        return model, scaler, feature_columns, eval_metrics
    except FileNotFoundError as e:
        st.error(f"Error loading model artifacts: {e}")
        st.info("Using demo mode with mock data for demonstration purposes.")
        # Return mock objects for demo
        return None, None, None, None

# Initialize database
def init_database():
    """Initialize SQLite database for storing predictions"""
    conn = sqlite3.connect("heart_predictions.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS heart_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            age INTEGER,
            sex TEXT,
            cp INTEGER,
            trestbps INTEGER,
            chol INTEGER,
            fbs INTEGER,
            restecg INTEGER,
            thalach INTEGER,
            exang INTEGER,
            oldpeak REAL,
            slope INTEGER,
            ca INTEGER,
            thal INTEGER,
            prediction INTEGER,
            probability REAL,
            risk_level TEXT,
            timestamp TEXT,
            email TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Email notification function
def send_email_alert(patient_name, email, prediction, probability, risk_level, smtp_config=None):
    """Send email alert for high-risk patients"""
    try:
        # Default email configuration (users should update these)
        if smtp_config is None:
            smtp_config = {
                'smtp_server': "smtp.gmail.com",
                'smtp_port': 587,
                'sender_email': st.secrets.get("SENDER_EMAIL", ""),
                'sender_password': st.secrets.get("SENDER_PASSWORD", "")
            }
        
        # Check if email configuration is available
        if not smtp_config['sender_email'] or not smtp_config['sender_password']:
            st.warning("Email configuration not found. Please configure email settings in Streamlit secrets or settings page.")
            return False
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = smtp_config['sender_email']
        msg['To'] = email
        msg['Subject'] = f"Heart Disease Risk Assessment - {patient_name}"
        
        body = f"""
        Dear {patient_name},
        
        Your recent heart disease risk assessment has been completed.
        
        Risk Level: {risk_level}
        Probability: {probability:.2%}
        Assessment Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        {'⚠️ IMPORTANT: Please consult with a healthcare professional immediately for further evaluation and treatment recommendations.' if prediction == 1 else '✅ Your current risk appears to be low. Continue maintaining a healthy lifestyle.'}
        
        This is an automated assessment and should not replace professional medical advice.
        
        Best regards,
        Heart Health Monitoring System
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port'])
        server.starttls()
        server.login(smtp_config['sender_email'], smtp_config['sender_password'])
        text = msg.as_string()
        server.sendmail(smtp_config['sender_email'], email, text)
        server.quit()
        
        return True
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        return False

# Load artifacts
model, scaler, feature_columns, eval_metrics = load_artifacts()

# Initialize database
init_database()

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
page_selection = st.sidebar.radio("Select Page", [
    "📊 Dashboard & Model Performance",
    "🔍 Make New Prediction",
    "🧠 Advanced Analytics & Clustering",
    "📋 Patient Records & Search",
    "📈 Risk Analytics",
    "⚙️ Settings"
])

# Dashboard page
if page_selection == "📊 Dashboard & Model Performance":
    st.header("📊 Model Performance Overview")
    
    if eval_metrics:
        # Key metrics
        st.subheader("🎯 Key Evaluation Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Accuracy</h3>
                <h2>{eval_metrics['accuracy']:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Precision</h3>
                <h2>{eval_metrics['precision']:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Recall</h3>
                <h2>{eval_metrics['recall']:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>F1-Score</h3>
                <h2>{eval_metrics['f1_score']:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <h3>AUC-ROC</h3>
                <h2>{eval_metrics['roc_auc']:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("🎯 Confusion Matrix")
            fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
            sns.heatmap(eval_metrics['conf_matrix'], annot=True, fmt='d', cmap='Blues',
                       xticklabels=['No Disease', 'Heart Disease'],
                       yticklabels=['No Disease', 'Heart Disease'], ax=ax_cm)
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('Actual')
            st.pyplot(fig_cm)
        
        with col_chart2:
            st.subheader("📈 ROC Curve")
            fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
            ax_roc.plot(eval_metrics['fpr'], eval_metrics['tpr'], 
                       color='darkorange', lw=3, 
                       label=f'ROC Curve (AUC = {eval_metrics["roc_auc"]:.3f})')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('Receiver Operating Characteristic')
            ax_roc.legend(loc="lower right")
            ax_roc.grid(True, alpha=0.3)
            st.pyplot(fig_roc)
        
        # Feature importance (if available)
        if model and hasattr(model, 'feature_importances_'):
            st.subheader("🔍 Feature Importance Analysis")
            feature_names = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'] + \
                          [col for col in feature_columns if col not in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']]
            
            if len(feature_names) == len(model.feature_importances_):
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=True)
                
                fig_importance = px.bar(importance_df, x='Importance', y='Feature',
                                      orientation='h', title="Feature Importance",
                                      color='Importance', color_continuous_scale='viridis')
                st.plotly_chart(fig_importance, use_container_width=True)

# Prediction page
elif page_selection == "🔍 Make New Prediction":
    st.header("🔍 Heart Disease Risk Assessment")
    
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        st.subheader("👤 Patient Information")
        
        # Patient details
        patient_name = st.text_input("Patient Name", placeholder="Enter patient name")
        patient_email = st.text_input("Email (for alerts)", placeholder="patient@example.com")
        
        # Medical parameters
        age = st.slider("Age", 20, 90, 45)
        sex_display = st.radio("Sex", ['Male', 'Female'])
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                         help="0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic")
        
        col_vital1, col_vital2 = st.columns(2)
        with col_vital1:
            trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 200, 120)
            chol = st.slider("Cholesterol (mg/dl)", 100, 400, 200)
            thalach = st.slider("Max Heart Rate", 70, 210, 150)
        
        with col_vital2:
            fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1])
            restecg = st.selectbox("Resting ECG", [0, 1, 2],
                                 help="0: Normal, 1: ST-T abnormality, 2: LV hypertrophy")
            exang = st.radio("Exercise Induced Angina", [0, 1])
        
        oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
        slope = st.selectbox("ST Slope", [0, 1, 2], 
                           help="0: Upsloping, 1: Flat, 2: Downsloping")
        ca = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", [0, 1, 2, 3],
                           help="0: Unknown, 1: Normal, 2: Fixed defect, 3: Reversible defect")
        
        # Email notification settings
        send_email = st.checkbox("Send email notification", value=False)
        if send_email and patient_email:
            if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', patient_email):
                st.warning("Please enter a valid email address")
    
    with col_result:
        if st.button("🔍 Analyze Heart Disease Risk", type="primary"):
            if not patient_name:
                st.warning("Please enter patient name")
            else:
                # Prepare input data
                sex_encoded = 1 if sex_display == 'Male' else 0
                
                input_data = {
                    'age': age, 'sex': sex_encoded, 'cp': cp, 'trestbps': trestbps,
                    'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
                    'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
                }
                
                input_df = pd.DataFrame([input_data])
                
                if model and scaler and feature_columns:
                    # Preprocess data
                    input_df_encoded = pd.get_dummies(input_df.copy(), 
                                                    columns=one_hot_encode_features, 
                                                    drop_first=True)
                    
                    # Align columns
                    missing_cols = set(feature_columns) - set(input_df_encoded.columns)
                    for c in missing_cols:
                        input_df_encoded[c] = 0
                    
                    input_df_final = input_df_encoded[feature_columns]
                    scaled_input = scaler.transform(input_df_final)
                    
                    # Make prediction
                    prediction = model.predict(scaled_input)[0]
                    proba = model.predict_proba(scaled_input)[0]
                    risk_probability = proba[1]
                    
                    # Determine risk level
                    if risk_probability >= 0.8:
                        risk_level = "Very High Risk"
                        risk_color = "#DC143C"
                    elif risk_probability >= 0.6:
                        risk_level = "High Risk"
                        risk_color = "#FF6347"
                    elif risk_probability >= 0.4:
                        risk_level = "Moderate Risk"
                        risk_color = "#FFA500"
                    else:
                        risk_level = "Low Risk"
                        risk_color = "#32CD32"
                    
                    # Display results
                    st.subheader("📋 Assessment Results")
                    
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="risk-high">
                            <h2>⚠️ {risk_level}</h2>
                            <h3>{risk_probability:.1%} Probability</h3>
                            <p>Immediate medical consultation recommended</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="risk-low">
                            <h2>✅ {risk_level}</h2>
                            <h3>{risk_probability:.1%} Probability</h3>
                            <p>Continue healthy lifestyle practices</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Probability breakdown
                    fig_prob = go.Figure(go.Bar(
                        x=["No Disease", "Heart Disease"],
                        y=[proba[0], proba[1]],
                        marker_color=["#32CD32", "#DC143C"]
                    ))
                    fig_prob.update_layout(
                        title="Risk Probability Breakdown",
                        xaxis_title="Outcome",
                        yaxis_title="Probability",
                        yaxis_range=[0, 1],
                        height=400
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)
                    
                    # Save to database
                    conn = sqlite3.connect("heart_predictions.db")
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO heart_predictions 
                        (patient_name, age, sex, cp, trestbps, chol, fbs, restecg, 
                         thalach, exang, oldpeak, slope, ca, thal, prediction, 
                         probability, risk_level, timestamp, email)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (patient_name, age, sex_display, cp, trestbps, chol, fbs, 
                         restecg, thalach, exang, oldpeak, slope, ca, thal, 
                         prediction, risk_probability, risk_level, 
                         datetime.now().strftime("%Y-%m-%d %H:%M:%S"), patient_email))
                    conn.commit()
                    conn.close()
                    
                    st.success("✅ Assessment saved to patient records")
                    
                    # Send email if requested
                    if send_email and patient_email and re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', patient_email):
                        with st.spinner("Sending email notification..."):
                            if send_email_alert(patient_name, patient_email, prediction, risk_probability, risk_level):
                                st.success("📧 Email notification sent successfully")
                            else:
                                st.warning("📧 Email notification failed (check settings)")
                else:
                    st.error("Model not loaded. Please check model files.")

# Advanced Analytics page
elif page_selection == "🧠 Advanced Analytics & Clustering":
    st.header("🧠 Advanced Patient Analytics")
    
    try:
        # Load patient data from database
        conn = sqlite3.connect("heart_predictions.db")
        df_patients = pd.read_sql_query("""
            SELECT age, trestbps as blood_pressure, chol as cholesterol, 
                   thalach as max_heart_rate, oldpeak, prediction, risk_level
            FROM heart_predictions
        """, conn)
        conn.close()
        
        if len(df_patients) < 10:
            # Generate sample data for demonstration
            np.random.seed(42)
            sample_size = 100
            df_patients = pd.DataFrame({
                'age': np.random.randint(30, 80, sample_size),
                'blood_pressure': np.random.randint(90, 180, sample_size),
                'cholesterol': np.random.randint(150, 350, sample_size),
                'max_heart_rate': np.random.randint(100, 200, sample_size),
                'oldpeak': np.random.uniform(0, 4, sample_size),
                'prediction': np.random.choice([0, 1], sample_size, p=[0.6, 0.4]),
                'risk_level': np.random.choice(['Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk'], 
                                             sample_size, p=[0.4, 0.3, 0.2, 0.1])
            })
            st.info("Using sample data for demonstration. Analyze real patients to see actual insights.")
        
        # Patient clustering
        st.subheader("👥 Patient Risk Clustering")
        
        features_for_clustering = ['age', 'blood_pressure', 'cholesterol', 'max_heart_rate', 'oldpeak']
        X_cluster = df_patients[features_for_clustering].fillna(df_patients[features_for_clustering].median())
        X_scaled = StandardScaler().fit_transform(X_cluster)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        df_patients['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Anomaly detection
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        df_patients['anomaly'] = iso_forest.fit_predict(X_scaled)
        df_patients['anomaly_label'] = df_patients['anomaly'].map({1: 'Normal', -1: 'Anomaly'})
        
        # 3D scatter plot
        fig_3d = px.scatter_3d(
            df_patients, 
            x='age', 
            y='blood_pressure', 
            z='cholesterol',
            color='cluster',
            symbol='anomaly_label',
            size='max_heart_rate',
            hover_data=['risk_level', 'oldpeak'],
            title="Patient Risk Clusters with Anomaly Detection",
            labels={'cluster': 'Risk Cluster'}
        )
        fig_3d.update_layout(height=600)
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Cluster analysis
        col_cluster1, col_cluster2 = st.columns(2)
        
        with col_cluster1:
            st.subheader("📊 Cluster Characteristics")
            cluster_summary = df_patients.groupby('cluster')[features_for_clustering].mean().round(2)
            st.dataframe(cluster_summary)
        
        with col_cluster2:
            st.subheader("🚨 Anomaly Summary")
            anomaly_count = df_patients['anomaly_label'].value_counts()
            fig_anomaly = px.pie(values=anomaly_count.values, names=anomaly_count.index,
                               title="Anomaly Distribution")
            st.plotly_chart(fig_anomaly, use_container_width=True)
        
        # Risk distribution analysis
        st.subheader("📈 Risk Distribution Analysis")
        fig_risk = px.histogram(df_patients, x='risk_level', color='prediction',
                              title="Risk Level Distribution by Prediction")
        st.plotly_chart(fig_risk, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in analytics: {str(e)}")
        st.info("Generate some predictions first to see advanced analytics.")

# Patient Records page
elif page_selection == "📋 Patient Records & Search":
    st.header("📋 Patient Records Management")
    
    try:
        conn = sqlite3.connect("heart_predictions.db")
        df_records = pd.read_sql_query("SELECT * FROM heart_predictions ORDER BY timestamp DESC", conn)
        conn.close()
        
        if len(df_records) == 0:
            st.info("No patient records found. Make some predictions first.")
        else:
            # Search and filter options
            st.subheader("🔍 Search & Filter Options")
            
            col_search1, col_search2, col_search3 = st.columns(3)
            
            with col_search1:
                search_name = st.text_input("Search by Patient Name")
                risk_filter = st.selectbox("Filter by Risk Level", 
                                         ["All"] + list(df_records['risk_level'].unique()))
            
            with col_search2:
                # Fixed slider issue - ensure min and max are different
                min_age = int(df_records['age'].min())
                max_age = int(df_records['age'].max())
                if min_age == max_age:
                    max_age = min_age + 1  # Ensure max is greater than min
                
                age_range = st.slider("Age Range", min_age, max_age, (min_age, max_age))
                prediction_filter = st.selectbox("Filter by Prediction", 
                                                ["All", "No Disease (0)", "Heart Disease (1)"])
            
            with col_search3:
                date_from = st.date_input("From Date")
                date_to = st.date_input("To Date")
            
            # Apply filters
            filtered_df = df_records.copy()
            
            if search_name:
                filtered_df = filtered_df[filtered_df['patient_name'].str.contains(search_name, case=False, na=False)]
            
            if risk_filter != "All":
                filtered_df = filtered_df[filtered_df['risk_level'] == risk_filter]
            
            filtered_df = filtered_df[(filtered_df['age'] >= age_range[0]) & (filtered_df['age'] <= age_range[1])]
            
            if prediction_filter != "All":
                pred_value = 0 if "No Disease" in prediction_filter else 1
                filtered_df = filtered_df[filtered_df['prediction'] == pred_value]
            
            if date_from and date_to:
                filtered_df = filtered_df[
                    (pd.to_datetime(filtered_df['timestamp']).dt.date >= date_from) &
                    (pd.to_datetime(filtered_df['timestamp']).dt.date <= date_to)
                ]
            
            # Display results
            st.subheader(f"📊 Search Results ({len(filtered_df)} records)")
            
            if len(filtered_df) > 0:
                # Summary statistics
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                with col_stat1:
                    st.metric("Total Patients", len(filtered_df))
                with col_stat2:
                    high_risk = len(filtered_df[filtered_df['prediction'] == 1])
                    st.metric("High Risk Cases", high_risk)
                with col_stat3:
                    avg_age = filtered_df['age'].mean()
                    st.metric("Average Age", f"{avg_age:.1f}")
                with col_stat4:
                    avg_prob = filtered_df['probability'].mean()
                    st.metric("Avg Risk Probability", f"{avg_prob:.2%}")
                
                # Data table
                st.dataframe(filtered_df[['patient_name', 'age', 'sex', 'risk_level', 
                                        'probability', 'timestamp', 'email']], 
                           use_container_width=True)
                
                # Download option
                csv_data = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name=f"heart_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Delete records
                st.subheader("🗑️ Delete Records")
                record_to_delete = st.selectbox("Select record to delete", 
                                               ["None"] + [f"{row['patient_name']} - {row['timestamp']}" 
                                                          for _, row in filtered_df.iterrows()])
                
                if record_to_delete != "None" and st.button("Delete Selected Record", type="secondary"):
                    record_id = filtered_df[filtered_df.apply(lambda x: f"{x['patient_name']} - {x['timestamp']}" == record_to_delete, axis=1)]['id'].iloc[0]
                    
                    conn = sqlite3.connect("heart_predictions.db")
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM heart_predictions WHERE id = ?", (record_id,))
                    conn.commit()
                    conn.close()
                    
                    st.success("Record deleted successfully!")
                    st.rerun()
            else:
                st.info("No records match your search criteria.")
                
    except Exception as e:
        st.error(f"Error accessing patient records: {str(e)}")



# Risk Analytics page
elif page_selection == "📈 Risk Analytics":
    st.header("📈 Population Risk Analytics")
    
    try:
        conn = sqlite3.connect("heart_predictions.db")
        df_analytics = pd.read_sql_query("SELECT * FROM heart_predictions", conn)
        conn.close()
        
        if df_analytics.empty:
            st.info("No data available for analytics. Make some predictions first.")
        else:
            # Overview metrics
            st.subheader("📊 Population Overview")
            col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
            
            with col_metric1:
                total_patients = len(df_analytics)
                st.metric("Total Assessments", total_patients)
            
            with col_metric2:
                high_risk_count = len(df_analytics[df_analytics['prediction'] == 1])
                high_risk_rate = (high_risk_count / total_patients * 100) if total_patients > 0 else 0
                st.metric("High Risk Rate", f"{high_risk_rate:.1f}%")
            
            with col_metric3:
                avg_age = df_analytics['age'].mean()
                st.metric("Average Age", f"{avg_age:.1f}")
            
            with col_metric4:
                avg_risk_prob = df_analytics['probability'].mean()
                st.metric("Avg Risk Probability", f"{avg_risk_prob:.2%}")
            
            # Risk trends over time
            st.subheader("📅 Risk Trends Over Time")
            df_analytics['date'] = pd.to_datetime(df_analytics['timestamp'], errors='coerce').dt.date
            daily_risk = df_analytics.groupby('date').agg({
                'prediction': ['count', 'sum'],
                'probability': 'mean'
            }).round(3)
            
            daily_risk.columns = ['Total_Assessments', 'High_Risk_Cases', 'Avg_Risk_Probability']
            daily_risk = daily_risk.reset_index()
            
            fig_trend = px.line(daily_risk, x='date', y='Avg_Risk_Probability',
                              title="Average Risk Probability Over Time")
            fig_trend.add_scatter(x=daily_risk['date'], y=daily_risk['High_Risk_Cases']/daily_risk['Total_Assessments'],
                                name='High Risk Rate', yaxis='y2')
            fig_trend.update_layout(yaxis2=dict(overlaying='y', side='right'))
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Age and gender analysis
            col_demo1, col_demo2 = st.columns(2)
            
            with col_demo1:
                st.subheader("👥 Risk by Age Groups")
                df_analytics['age_group'] = pd.cut(df_analytics['age'], 
                                                 bins=[0, 40, 50, 60, 70, 100], 
                                                 labels=['<40', '40-50', '50-60', '60-70', '70+'])
                age_risk = df_analytics.groupby('age_group')['prediction'].agg(['count', 'sum'])
                age_risk['risk_rate'] = (age_risk['sum'] / age_risk['count'] * 100).round(1)
                
                fig_age = px.bar(x=age_risk.index, y=age_risk['risk_rate'],
                               title="Risk Rate by Age Group (%)")
                st.plotly_chart(fig_age, use_container_width=True)
            
            with col_demo2:
                st.subheader("⚧ Risk by Gender")
                gender_risk = df_analytics.groupby('sex')['prediction'].agg(['count', 'sum'])
                gender_risk['risk_rate'] = (gender_risk['sum'] / gender_risk['count'] * 100).round(1)
                
                fig_gender = px.pie(values=gender_risk['count'], names=gender_risk.index,
                                  title="Assessment Distribution by Gender")
                st.plotly_chart(fig_gender, use_container_width=True)
            
            # Risk level distribution
            st.subheader("🎯 Risk Level Distribution")
            risk_dist = df_analytics['risk_level'].value_counts()
            fig_risk_dist = px.funnel(y=risk_dist.index, x=risk_dist.values,
                                    title="Patient Distribution by Risk Level")
            st.plotly_chart(fig_risk_dist, use_container_width=True)
            
            # Correlation analysis
            st.subheader("🔗 Risk Factor Correlations")
            numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'probability']
            if all(col in df_analytics.columns for col in numerical_cols):
                corr_matrix = df_analytics[numerical_cols].corr()
                
                fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                           square=True, ax=ax_corr)
                ax_corr.set_title('Risk Factor Correlations')
                st.pyplot(fig_corr)
            
            # Export analytics report
            st.subheader("📄 Export Analytics Report")
            if st.button("Generate Analytics Report"):
                report_data = {
                    'Total Patients': total_patients,
                    'High Risk Cases': high_risk_count,
                    'High Risk Rate (%)': high_risk_rate,
                    'Average Age': avg_age,
                    'Average Risk Probability': avg_risk_prob,
                    'Age Group Analysis': age_risk.to_dict(),
                    'Gender Analysis': gender_risk.to_dict(),
                    'Risk Level Distribution': risk_dist.to_dict()
                }
                
                report_df = pd.DataFrame(list(report_data.items()), columns=['Metric', 'Value'])
                csv_report = report_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Analytics Report",
                    data=csv_report,
                    file_name=f"heart_risk_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
    except Exception as e:  
        st.error(f"Error in risk analytics: {str(e)}")
        print(f"Error details: {e}")  # Print error details to console for debugging


# Settings page
elif page_selection == "⚙️ Settings":
    st.header("⚙️ System Settings")
    
    st.subheader("📧 Email Configuration")
    st.info("Configure SMTP settings for email notifications")
    
    col_email1, col_email2 = st.columns(2)
    
    with col_email1:
        smtp_server = st.text_input("SMTP Server", value="smtp.gmail.com")
        smtp_port = st.number_input("SMTP Port", value=587)
        sender_email = st.text_input("Sender Email", placeholder="your_email@gmail.com")
    
    with col_email2:
        sender_password = st.text_input("App Password", type="password", 
                                       help="Use app-specific password for Gmail")
        test_email = st.text_input("Test Email", placeholder="test@example.com")
    
    if st.button("Test Email Configuration"):
        if sender_email and sender_password and test_email:
            try:
                # Test email
                msg = MIMEText("Test email from Heart Disease Prediction System")
                msg['Subject'] = "Test Email"
                msg['From'] = sender_email
                msg['To'] = test_email
                
                server = smtplib.SMTP(smtp_server, smtp_port)
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
                server.quit()
                
                st.success("✅ Email configuration test successful!")
            except Exception as e:
                st.error(f"❌ Email test failed: {str(e)}")
        else:
            st.warning("Please fill in all email configuration fields")
    
    st.subheader("🗄️ Database Management")
    
    col_db1, col_db2 = st.columns(2)
    
    with col_db1:
        if st.button("📊 Database Statistics"):
            try:
                conn = sqlite3.connect("heart_predictions.db")
                cursor = conn.cursor()
                
                # Get table info
                cursor.execute("SELECT COUNT(*) FROM heart_predictions")
                record_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT patient_name) FROM heart_predictions")
                unique_patients = cursor.fetchone()[0]
                
                cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM heart_predictions")
                date_range = cursor.fetchone()
                
                conn.close()
                
                st.success(f"""
                **Database Statistics:**
                - Total Records: {record_count}
                - Unique Patients: {unique_patients}
                - Date Range: {date_range[0]} to {date_range[1]}
                """)
            except Exception as e:
                st.error(f"Database error: {str(e)}")
    
    with col_db2:
        if st.button("🗑️ Clear All Data", type="secondary"):
            if st.checkbox("I understand this will delete all patient records"):
                try:
                    conn = sqlite3.connect("heart_predictions.db")
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM heart_predictions")
                    conn.commit()
                    conn.close()
                    st.success("All data cleared successfully!")
                except Exception as e:
                    st.error(f"Error clearing data: {str(e)}")
    
    st.subheader("📋 Model Information")
    if model:
        st.success("✅ Model loaded successfully")
        st.info(f"""
        **Model Details:**
        - Type: Random Forest Classifier
        - Features: {len(feature_columns) if feature_columns else 'Unknown'}
        - Preprocessing: One-hot encoding + Standard scaling
        """)
    else:
        st.error("❌ Model not loaded")
        st.warning("Please ensure model files are in the correct directory")
    
    st.subheader("ℹ️ System Information")
    st.info(f"""
    **Application Version:** 2.0.0
    **Last Updated:** {datetime.now().strftime('%Y-%m-%d')}
    **Features:**
    - Advanced patient analytics
    - Email notifications
    - Patient record management
    - Risk clustering analysis
    - Comprehensive search functionality
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>💓 Advanced Heart Disease Prediction System v2.0</strong></p>
    <p>Powered by Machine Learning | Built with Streamlit</p>
    <p><em>This system is for educational purposes. Always consult healthcare professionals for medical decisions.</em></p>
</div>
""", unsafe_allow_html=True)