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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
import base64

# Configure Streamlit page
st.set_page_config(
    page_title="🚀 AI Plant Guardian",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS for futuristic UI (same as before)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #533483 100%);
        color: #ffffff;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        background: rgba(255, 255, 255, 0.1);
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #00d4ff, #7b2cbf);
        border-radius: 10px;
    }
    
    /* Main Header */
    .main-header {
        font-family: 'Orbitron', monospace;
        font-size: 4rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, #00d4ff, #7b2cbf, #ff006e, #00f5ff);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 3s ease-in-out infinite;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
        margin-bottom: 2rem;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .sub-header {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2rem;
        font-weight: 600;
        color: #00d4ff;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        margin-bottom: 1.5rem;
    }
    
    .subtitle {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.3rem;
        text-align: center;
        color: #a0a0a0;
        margin-bottom: 3rem;
        letter-spacing: 2px;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(0, 212, 255, 0.2);
        border-color: rgba(0, 212, 255, 0.3);
    }
    
    /* Neon Prediction Boxes */
    .neon-healthy {
        background: linear-gradient(135deg, rgba(0, 255, 127, 0.1), rgba(0, 255, 127, 0.05));
        border: 2px solid #00ff7f;
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 
            0 0 20px rgba(0, 255, 127, 0.3),
            inset 0 0 20px rgba(0, 255, 127, 0.1);
        animation: healthyGlow 2s ease-in-out infinite alternate;
    }
    
    @keyframes healthyGlow {
        from { box-shadow: 0 0 20px rgba(0, 255, 127, 0.3), inset 0 0 20px rgba(0, 255, 127, 0.1); }
        to { box-shadow: 0 0 30px rgba(0, 255, 127, 0.5), inset 0 0 30px rgba(0, 255, 127, 0.2); }
    }
    
    .neon-disease {
        background: linear-gradient(135deg, rgba(255, 20, 147, 0.1), rgba(255, 20, 147, 0.05));
        border: 2px solid #ff1493;
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 
            0 0 20px rgba(255, 20, 147, 0.3),
            inset 0 0 20px rgba(255, 20, 147, 0.1);
        animation: diseaseGlow 2s ease-in-out infinite alternate;
    }
    
    @keyframes diseaseGlow {
        from { box-shadow: 0 0 20px rgba(255, 20, 147, 0.3), inset 0 0 20px rgba(255, 20, 147, 0.1); }
        to { box-shadow: 0 0 30px rgba(255, 20, 147, 0.5), inset 0 0 30px rgba(255, 20, 147, 0.2); }
    }
    
    /* Sidebar Styling - FIXED */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(12, 12, 12, 0.95), rgba(26, 26, 46, 0.95));
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(0, 212, 255, 0.2);
    }

    /* Sidebar text visibility fix */
    .css-1d391kg .stMarkdown, 
    .css-1d391kg .stSelectbox label,
    .css-1d391kg .stMarkdown p,
    .css-1d391kg .stMarkdown h1,
    .css-1d391kg .stMarkdown h2,
    .css-1d391kg .stMarkdown h3,
    .css-1d391kg .stMarkdown h4 {
        color: #ffffff !important;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.3) !important;
    }
    
    /* Futuristic Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #00d4ff, #7b2cbf);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 15px 30px;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0, 212, 255, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(0, 212, 255, 0.5);
        background: linear-gradient(45deg, #7b2cbf, #00d4ff);
    }
    
    /* Metrics Cards */
    .metric-card {
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        background: rgba(0, 212, 255, 0.2);
        transform: scale(1.05);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(45deg, #00d4ff, #7b2cbf, #ff006e);
        background-size: 200% 200%;
        animation: progressGlow 2s ease-in-out infinite;
    }
    
    @keyframes progressGlow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* File Uploader */
    .stFileUploader > div > div {
        background: rgba(0, 212, 255, 0.05);
        border: 2px dashed rgba(0, 212, 255, 0.3);
        border-radius: 20px;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div:hover {
        border-color: rgba(0, 212, 255, 0.6);
        background: rgba(0, 212, 255, 0.1);
    }
    
    /* Sidebar species list styling */
    .sidebar-species {
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: #ffffff !important;
    }

    .sidebar-species h3 {
        color: #00d4ff !important;
        font-family: 'Rajdhani', sans-serif !important;
        margin-bottom: 10px !important;
    }

    .sidebar-species ul {
        list-style: none;
        padding: 0;
    }

    .sidebar-species li {
        color: #ffffff !important;
        padding: 3px 0;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Class labels dictionary - SESUAI DENGAN TRAINING ANDA
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
        model = load_model('fine_tuned_model.h5')
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("📝 Please make sure 'fine_tuned_model.h5' is in the same directory as this script.")
        return None

def preprocess_image_training_exact(uploaded_file):
    """
    Preprocessing PERSIS seperti di training Anda:
    - Menggunakan keras image.load_img dengan target_size=(224, 224)
    - image.img_to_array()
    - np.expand_dims()
    - mobilenet_v2.preprocess_input()
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Load image PERSIS seperti di training
        img = image.load_img(tmp_path, target_size=(224, 224))
        
        # Convert to array
        img_array = image.img_to_array(img)
        
        # Expand dimensions
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess - SAMA PERSIS dengan training
        img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        return img_preprocessed
        
    finally:
        # Clean up temp file
        os.unlink(tmp_path)

def preprocess_camera_image(camera_input):
    """Preprocess camera image dengan metode yang sama"""
    # Convert camera input to PIL Image
    pil_image = Image.open(camera_input)
    
    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        pil_image.save(tmp_file.name)
        tmp_path = tmp_file.name
    
    try:
        # Load dengan keras image.load_img (sama dengan training)
        img = image.load_img(tmp_path, target_size=(224, 224))
        
        # Convert to array
        img_array = image.img_to_array(img)
        
        # Expand dimensions
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess
        img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        return img_preprocessed
        
    finally:
        # Clean up temp file
        os.unlink(tmp_path)

def predict_disease(model, processed_img):
    """Make prediction on preprocessed image"""
    if model is None:
        return None, None, None
    
    # GUNAKAN PREDIKSI ASLI MODEL ANDA
    predictions = model.predict(processed_img, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    
    # Get class label
    predicted_class = CLASS_LABELS.get(predicted_class_idx, 'Unknown')
    
    return predicted_class, confidence, predictions[0]

def create_futuristic_chart(predictions, top_n=5):
    """Create a futuristic confidence chart"""
    # Get top N predictions
    top_indices = np.argsort(predictions)[-top_n:][::-1]
    top_classes = [CLASS_LABELS[i].split('___')[1].replace('_', ' ') for i in top_indices]
    top_confidences = [predictions[i] for i in top_indices]
    
    # Create futuristic bar chart
    fig = go.Figure()
    
    # Add bars with gradient colors
    colors = ['#00d4ff', '#7b2cbf', '#ff006e', '#00f5ff', '#ff7b00']
    
    for i, (conf, cls, color) in enumerate(zip(top_confidences, top_classes, colors)):
        fig.add_trace(go.Bar(
            x=[conf],
            y=[cls],
            orientation='h',
            marker=dict(
                color=color,
                line=dict(color='rgba(255,255,255,0.3)', width=1)
            ),
            name=cls,
            hovertemplate=f'<b>{cls}</b><br>Confidence: {conf:.2%}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text="🔬 Quantum Analysis Results",
            font=dict(family="Orbitron", size=20, color="#00d4ff")
        ),
        xaxis=dict(
            title="Neural Confidence Level",
            titlefont=dict(family="Rajdhani", size=14, color="#ffffff"),
            tickfont=dict(color="#ffffff"),
            gridcolor="rgba(0, 212, 255, 0.2)",
            showgrid=True
        ),
        yaxis=dict(
            titlefont=dict(family="Rajdhani", size=14, color="#ffffff"),
            tickfont=dict(color="#ffffff")
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False
    )
    
    return fig

def main():
    # Futuristic Header
    st.markdown('<h1 class="main-header">🤖 AI PLANT GUARDIAN</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">NEXT-GENERATION QUANTUM PLANT DISEASE DETECTION SYSTEM</p>', unsafe_allow_html=True)
    
    # Loading animation for model
    with st.spinner("🔄 Initializing Quantum Neural Networks..."):
        model = load_trained_model()
    
    if model is None:
        st.error("🚨 SYSTEM FAILURE: Neural Network Offline")
        st.stop()
    else:
        st.success("✅ QUANTUM AI SYSTEM ONLINE")
    
    # Futuristic Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="glass-card">
            <h2 style="color: #00d4ff; font-family: 'Orbitron', monospace; text-align: center;">
                🎛️ CONTROL PANEL
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        app_mode = st.selectbox(
            "🚀 SELECT OPERATION MODE:",
            ["🖼️ QUANTUM IMAGE ANALYSIS", "📹 LIVE NEURAL SCAN"],
            help="Choose your preferred analysis method"
        )
        
        st.markdown("---")
        
        # System Status
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #00d4ff; font-family: 'Rajdhani', sans-serif;">⚡ SYSTEM STATUS</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4 style="color: #00ff7f;">AI CORES</h4>
                <h2 style="color: #ffffff;">38</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4 style="color: #ff006e;">ACCURACY</h4>
                <h2 style="color: #ffffff;">96%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Supported Species
        st.markdown("""
<div class="sidebar-species">
    <h3>🧬 SPECIES DATABASE</h3>
    <ul>
        <li>🍎 Malus domestica (Apple)</li>
        <li>🫐 Vaccinium corymbosum (Blueberry)</li>
        <li>🍒 Prunus avium (Cherry)</li>
        <li>🌽 Zea mays (Corn)</li>
        <li>🍇 Vitis vinifera (Grape)</li>
        <li>🍊 Citrus sinensis (Orange)</li>
        <li>🍑 Prunus persica (Peach)</li>
        <li>🌶️ Capsicum annuum (Pepper)</li>
        <li>🥔 Solanum tuberosum (Potato)</li>
        <li>🍓 Fragaria × ananassa (Strawberry)</li>
        <li>🍅 Solanum lycopersicum (Tomato)</li>
    </ul>
</div>
""", unsafe_allow_html=True)
    
    # Main content based on selected mode
    if app_mode == "🖼️ QUANTUM IMAGE ANALYSIS":
        quantum_image_analysis(model)
    elif app_mode == "📹 LIVE NEURAL SCAN":
        live_neural_scan(model)

def quantum_image_analysis(model):
    st.markdown('<h2 class="sub-header">🖼️ QUANTUM IMAGE ANALYSIS</h2>', unsafe_allow_html=True)
    
    # File uploader with futuristic styling
    uploaded_file = st.file_uploader(
        "🛸 UPLOAD SPECIMEN FOR QUANTUM ANALYSIS",
        type=['jpg', 'jpeg', 'png'],
        help="Upload high-resolution plant specimen image for AI analysis"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            <div class="glass-card">
                <h3 style="color: #00d4ff; font-family: 'Rajdhani', sans-serif;">📡 SPECIMEN SCAN</h3>
            </div>
            """, unsafe_allow_html=True)
            
            image_pil = Image.open(uploaded_file)
            st.image(image_pil, caption="Quantum Specimen Analysis", use_column_width=True)
            
            # Image metadata
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: #00ff7f;">📊 SCAN METADATA</h4>
                <div class="tech-text">
            """, unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"🔍 **Dimensions:** {image_pil.size[0]} × {image_pil.size[1]}")
                st.write(f"🎨 **Color Mode:** {image_pil.mode}")
            with col_b:
                st.write(f"📁 **Format:** {image_pil.format}")
                st.write(f"💾 **Size:** {len(uploaded_file.getvalue()) / 1024:.1f} KB")
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="glass-card">
                <h3 style="color: #00d4ff; font-family: 'Rajdhani', sans-serif;">🧠 AI ANALYSIS RESULTS</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Quantum Analysis Button
            if st.button("🚀 INITIATE QUANTUM ANALYSIS", type="primary"):
                # Futuristic loading animation
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate quantum processing stages
                stages = [
                    "🔄 Initializing Quantum Processors...",
                    "🧬 Analyzing Cellular Structure...",
                    "🔬 Scanning Pathogen Signatures...",
                    "🤖 Running Neural Networks...",
                    "⚡ Finalizing Quantum Analysis..."
                ]
                
                for i, stage in enumerate(stages):
                    status_text.text(stage)
                    progress_bar.progress((i + 1) * 20)
                    time.sleep(0.5)
                
                status_text.empty()
                progress_bar.empty()
                
                # Preprocess dengan metode training yang tepat
                processed_img = preprocess_image_training_exact(uploaded_file)
                
                # Get prediction
                predicted_class, confidence, all_predictions = predict_disease(model, processed_img)
                
                if predicted_class:
                    # Parse prediction
                    parts = predicted_class.split('___')
                    plant_type = parts[0].replace('_', ' ').title()
                    disease_status = parts[1].replace('_', ' ').title() if len(parts) > 1 else 'Unknown'
                    
                    # Display results with futuristic styling
                    if 'healthy' in disease_status.lower():
                        st.markdown(f"""
                        <div class="neon-healthy">
                            <h2 style="text-align: center; font-family: 'Orbitron', monospace;">
                                🌿 SPECIMEN STATUS: OPTIMAL
                            </h2>
                            <div style="text-align: center; margin-top: 20px;">
                                <h3>Species: {plant_type}</h3>
                                <h3>Health Index: {disease_status}</h3>
                                <h3>Confidence Level: {confidence:.1%}</h3>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                    else:
                        st.markdown(f"""
                        <div class="neon-disease">
                            <h2 style="text-align: center; font-family: 'Orbitron', monospace;">
                                ⚠️ PATHOGEN DETECTED
                            </h2>
                            <div style="text-align: center; margin-top: 20px;">
                                <h3>Species: {plant_type}</h3>
                                <h3>Pathogen: {disease_status}</h3>
                                <h3>Detection Confidence: {confidence:.1%}</h3>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Quantum confidence visualization
                    st.progress(confidence, text=f"Neural Certainty: {confidence:.1%}")
                    
                    # Detailed quantum analysis
                    with st.expander("🔬 DETAILED QUANTUM ANALYSIS", expanded=True):
                        # Confidence chart
                        fig = create_futuristic_chart(all_predictions)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show top 5 predictions
                        st.markdown("### 🏆 Top 5 Predictions:")
                        top_5_indices = np.argsort(all_predictions)[-5:][::-1]
                        for i, idx in enumerate(top_5_indices):
                            class_name = CLASS_LABELS.get(idx, f"Unknown_{idx}")
                            conf = all_predictions[idx]
                            plant = class_name.split('___')[0].replace('_', ' ')
                            disease = class_name.split('___')[1].replace('_', ' ') if '___' in class_name else 'Unknown'
                            st.write(f"{i+1}. **{plant}** - {disease}: {conf:.2%}")

def live_neural_scan(model):
    st.markdown('<h2 class="sub-header">📹 LIVE NEURAL SCAN</h2>', unsafe_allow_html=True)
    
    # Camera input with futuristic styling
    camera_input = st.camera_input("📸 ACTIVATE NEURAL SCANNER")
    
    if camera_input is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            <div class="glass-card">
                <h3 style="color: #00d4ff;">📡 LIVE SPECIMEN CAPTURE</h3>
            </div>
            """, unsafe_allow_html=True)
            st.image(camera_input, caption="Neural Scan Capture", use_column_width=True)
        
        with col2:
            st.markdown("""
            <div class="glass-card">
                <h3 style="color: #00d4ff;">🤖 REAL-TIME ANALYSIS</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Real-time processing
            with st.spinner("⚡ Processing Neural Scan..."):
                # Preprocess camera image
                processed_img = preprocess_camera_image(camera_input)
                predicted_class, confidence, all_predictions = predict_disease(model, processed_img)
            
            if predicted_class:
                # Parse prediction
                parts = predicted_class.split('___')
                plant_type = parts[0].replace('_', ' ').title()
                disease_status = parts[1].replace('_', ' ').title() if len(parts) > 1 else 'Unknown'
                
                # Real-time results display
                if 'healthy' in disease_status.lower():
                    st.markdown(f"""
                    <div class="neon-healthy">
                        <h1 style="text-align: center; font-family: 'Orbitron', monospace;">
                            🌿 HEALTHY
                        </h1>
                        <h2 style="text-align: center;">{plant_type}</h2>
                        <h3 style="text-align: center;">Confidence: {confidence:.1%}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.success("✅ OPTIMAL SPECIMEN DETECTED")
                else:
                    st.markdown(f"""
                    <div class="neon-disease">
                        <h1 style="text-align: center; font-family: 'Orbitron', monospace;">
                            ⚠️ PATHOGEN
                        </h1>
                        <h2 style="text-align: center;">{plant_type}</h2>
                        <h3 style="text-align: center;">{disease_status}</h3>
                        <h3 style="text-align: center;">Alert Level: {confidence:.1%}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.warning("🚨 DISEASE SIGNATURE DETECTED")
        
        # Live analysis visualization
        st.markdown("### 📊 LIVE NEURAL NETWORK ANALYSIS")
        fig = create_futuristic_chart(all_predictions)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
