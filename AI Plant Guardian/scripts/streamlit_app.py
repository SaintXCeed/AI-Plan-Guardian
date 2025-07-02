import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="wide"
)

# Class labels dictionary (from your model)
CLASS_LABELS = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Blueberry___healthy',
    5: 'Cherry_(including_sour)___Powdery_mildew',
    6: 'Cherry_(including_sour)___healthy',
    7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    8: 'Corn_(maize)___Common_rust_',
    9: 'Corn_(maize)___Northern_Leaf_Blight',
    10: 'Corn_(maize)___healthy',
    11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)',
    13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)',
    16: 'Peach___Bacterial_spot',
    17: 'Peach___healthy',
    18: 'Pepper,_bell___Bacterial_spot',
    19: 'Pepper,_bell___healthy',
    20: 'Potato___Early_blight',
    21: 'Potato___Late_blight',
    22: 'Potato___healthy',
    23: 'Raspberry___healthy',
    24: 'Soybean___healthy',
    25: 'Squash___Powdery_mildew',
    26: 'Strawberry___Leaf_scorch',
    27: 'Strawberry___healthy',
    28: 'Tomato___Bacterial_spot',
    29: 'Tomato___Early_blight',
    30: 'Tomato___Late_blight',
    31: 'Tomato___Leaf_Mold',
    32: 'Tomato___Septoria_leaf_spot',
    33: 'Tomato___Spider_mites Two-spotted_spider_mite',
    34: 'Tomato___Target_Spot',
    35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    36: 'Tomato___Tomato_mosaic_virus',
    37: 'Tomato___healthy'
}

@st.cache_resource
def load_trained_model():
    """Load the pre-trained model"""
    try:
        # You need to upload your fine_tuned_model.h5 file
        model = load_model('fine_tuned_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please make sure 'fine_tuned_model.h5' is in the same directory as this script.")
        return None

def preprocess_image(img):
    """Preprocess image for model prediction"""
    # Resize image to 224x224
    img_resized = cv2.resize(img, (224, 224))
    
    # Convert to array and expand dimensions
    img_array = np.expand_dims(img_resized, axis=0)
    
    # Preprocess using MobileNetV2 preprocessing
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    return img_preprocessed

def predict_disease(model, img):
    """Make prediction on preprocessed image"""
    if model is None:
        return None, None
    
    # Preprocess image
    processed_img = preprocess_image(img)
    
    # Make prediction
    predictions = model.predict(processed_img, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    
    # Get class label
    predicted_class = CLASS_LABELS.get(predicted_class_idx, 'Unknown')
    
    return predicted_class, confidence

def main():
    st.title("🌱 Plant Disease Detection System")
    st.markdown("---")
    
    # Load model
    model = load_trained_model()
    
    if model is None:
        st.stop()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode:", 
                                   ["Image Upload", "Live Camera", "About"])
    
    if app_mode == "Image Upload":
        image_upload_mode(model)
    elif app_mode == "Live Camera":
        live_camera_mode(model)
    elif app_mode == "About":
        about_section()

def image_upload_mode(model):
    st.header("📸 Upload Image for Disease Detection")
    
    uploaded_file = st.file_uploader(
        "Choose a plant image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a plant leaf for disease detection"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            image_pil = Image.open(uploaded_file)
            st.image(image_pil, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.subheader("Prediction Results")
            
            # Convert PIL to OpenCV format
            img_array = np.array(image_pil)
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_cv = img_array
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                predicted_class, confidence = predict_disease(model, img_cv)
            
            if predicted_class:
                # Display results
                st.success("Analysis Complete!")
                
                # Parse the prediction
                parts = predicted_class.split('___')
                plant_type = parts[0].replace('_', ' ')
                disease_status = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'
                
                # Display formatted results
                st.markdown(f"**Plant Type:** {plant_type}")
                st.markdown(f"**Disease Status:** {disease_status}")
                st.markdown(f"**Confidence:** {confidence:.2%}")
                
                # Progress bar for confidence
                st.progress(confidence)
                
                # Health status indicator
                if 'healthy' in disease_status.lower():
                    st.success("🌿 Plant appears to be healthy!")
                else:
                    st.warning(f"⚠️ Disease detected: {disease_status}")
                    st.info("💡 Consider consulting with an agricultural expert for treatment recommendations.")

def live_camera_mode(model):
    st.header("📹 Live Camera Disease Detection")
    
    # Camera input
    camera_input = st.camera_input("Take a picture of the plant")
    
    if camera_input is not None:
        # Display captured image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Captured Image")
            st.image(camera_input, caption="Camera Capture", use_column_width=True)
        
        with col2:
            st.subheader("Real-time Analysis")
            
            # Convert to OpenCV format
            image_pil = Image.open(camera_input)
            img_array = np.array(image_pil)
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Make prediction
            with st.spinner("Processing..."):
                predicted_class, confidence = predict_disease(model, img_cv)
            
            if predicted_class:
                # Display results
                st.success("Analysis Complete!")
                
                # Parse the prediction
                parts = predicted_class.split('___')
                plant_type = parts[0].replace('_', ' ')
                disease_status = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'
                
                # Display formatted results with larger text
                st.markdown(f"### **Plant:** {plant_type}")
                st.markdown(f"### **Status:** {disease_status}")
                st.markdown(f"### **Confidence:** {confidence:.2%}")
                
                # Progress bar
                st.progress(confidence)
                
                # Health indicator
                if 'healthy' in disease_status.lower():
                    st.balloons()
                    st.success("🌿 Healthy plant detected!")
                else:
                    st.warning(f"⚠️ Disease detected: {disease_status}")
    
    # Instructions
    st.markdown("---")
    st.markdown("""
    ### 📋 Instructions for Best Results:
    - Ensure good lighting conditions
    - Focus on the leaf with clear details
    - Avoid blurry or dark images
    - Hold the camera steady
    - Fill the frame with the plant leaf
    """)

def about_section():
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ### 🌱 Plant Disease Detection System
    
    This application uses a Convolutional Neural Network (CNN) based on MobileNetV2 architecture 
    to detect diseases in plant leaves. The model has been trained on the PlantVillage dataset 
    and can identify 38 different classes of plant diseases and healthy conditions.
    
    ### 🔬 Technical Details:
    - **Model Architecture:** MobileNetV2 with Transfer Learning
    - **Input Size:** 224x224 pixels
    - **Number of Classes:** 38
    - **Supported Plants:** Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato
    
    ### 🎯 Supported Disease Types:
    - Bacterial infections
    - Fungal diseases
    - Viral infections
    - Pest damage
    - Healthy plant identification
    
    ### 🚀 Features:
    - **Image Upload:** Upload photos for analysis
    - **Live Camera:** Real-time detection using camera
    - **High Accuracy:** Fine-tuned model with excellent performance
    - **Fast Inference:** Optimized for real-time applications
    
    ### 📊 Model Performance:
    The model has been fine-tuned and achieves high accuracy on the test dataset, 
    making it suitable for practical agricultural applications.
    
    ### ⚠️ Disclaimer:
    This tool is for educational and research purposes. For critical agricultural decisions, 
    please consult with professional agronomists or plant pathologists.
    """)
    
    # Model statistics (you can update these with your actual results)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Classes Supported", "38")
    with col2:
        st.metric("Model Size", "~10MB")
    with col3:
        st.metric("Inference Speed", "<1s")

if __name__ == "__main__":
    main()
