# -------------------------------------------------------------------------
# Rice Grain Classification - Streamlit Web App (DEPLOYMENT READY)
# -------------------------------------------------------------------------
# This script is designed for deployment and performs inference only.
#
# To Deploy This App:
# 1. Train your model using the original script and ensure the
#    'best_rice_classifier.pkl' file is generated.
# 2. Place this 'app.py' file, your 'best_rice_classifier.pkl' file,
#    and a 'requirements.txt' file in a GitHub repository.
# 3. Connect your repository to a deployment platform like Streamlit Community Cloud.
# -------------------------------------------------------------------------

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import joblib
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# --- Page Configuration ---
st.set_page_config(
    page_title="Rice Grain Classifier",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Styling ---
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; }
    h1, h2, h3 { color: #1e293b; }
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
        max-width: 300px !important;
    }
    .stButton>button {
        background-color: #334155;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: background-color 0.3s;
    }
    .stButton>button:hover { background-color: #1e293b; }
    .card {
        background-color: #F7F3E9;
        border: 1px solid #E7E0D4;
        border-radius: 0.75rem;
        padding: 1rem;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        margin-bottom: 1rem;
    }
    .stMetric {
        background-color: #f1f5f9;
        border-left: 5px solid #64748b;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- GLOBAL VARIABLES & PATHS ---
MODEL_SAVE_PATH = "best_rice_classifier.pkl"
# IMPORTANT: The order of classes must match the order used during training.
# The original script used `sorted(os.listdir(TRAIN_PATH))`, so we sort them here.
CLASS_NAMES = sorted(["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"])

# --- CORE ML FUNCTIONS ---

@st.cache_resource
def load_model(path):
    """Loads a saved model from disk."""
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.error(f"Error loading the model: {e}")
            return None
    return None

def get_prediction(image_bytes):
    """Preprocesses an image and gets a prediction from the saved model."""
    model = load_model(MODEL_SAVE_PATH)
    if model is None:
        st.error("Model not found. Please ensure 'best_rice_classifier.pkl' is in the root directory.")
        return None, None

    # Preprocess the image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img_cv is None:
        st.error("Could not process image.")
        return None, None
    
    img_resized = cv2.resize(img_cv, (50, 50))
    img_flattened = img_resized.flatten()
    img_normalized = (img_flattened / 255.0).reshape(1, -1)

    # Get prediction and confidence
    try:
        predicted_index = model.predict(img_normalized)[0]
        confidence = model.predict_proba(img_normalized).max()
        predicted_class = CLASS_NAMES[predicted_index]
        return predicted_class, confidence
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None, None

# --- PAGE RENDERING FUNCTIONS ---

def home_page():
    st.title("🌾 AI-Powered Rice Grain Classification")
    st.subheader("Automating Quality and Purity Analysis with Machine Learning")
    
    st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:black;" /> """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Project Overview")
    st.write("""
        This application uses a machine learning model to classify rice grains into one of five varieties: 
        **Arborio, Basmati, Ipsala, Jasmine, and Karacadag**. The model was trained on a dataset of 
        5,000 images to identify rice based on features like shape, texture, and size.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Get Started")
    st.write("Navigate to the **Real-Time Classifier** page from the sidebar to upload your own rice grain image and get an instant classification from our trained model.")
    st.markdown('</div>', unsafe_allow_html=True)

def classifier_page():
    st.title("🔍 Real-Time Classifier")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Upload an Image to Classify")
    
    if not os.path.exists(MODEL_SAVE_PATH):
        st.error("🚨 No trained model found!")
        st.warning("Please ensure a `best_rice_classifier.pkl` file is present in the application's root directory. This app requires a pre-trained model to function.")
    else:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.image(Image.open(uploaded_file), caption='Uploaded Image', use_column_width=True)
            
            with col2:
                if st.button("Classify Grain"):
                    with st.spinner('Analyzing image...'):
                        image_bytes = uploaded_file.getvalue()
                        predicted_class, confidence = get_prediction(image_bytes)
                        
                        if predicted_class:
                            st.success("Classification Complete!")
                            st.metric("Predicted Rice Variety", value=predicted_class)
                            st.write("Confidence:")
                            st.progress(float(confidence))
                            st.markdown(f"**{confidence:.2%}**")
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Main App Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Real-Time Classifier"])
st.sidebar.markdown("---")
st.sidebar.info("This app uses a pre-trained machine learning model to classify rice varieties in real-time.")

# Display the selected page
if page == "Home":
    home_page()
elif page == "Real-Time Classifier":
    classifier_page()
