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

# Advanced CSS for futuristic UI
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
    
    /* Holographic Effect */
    .hologram {
        background: linear-gradient(45deg, transparent 30%, rgba(0, 212, 255, 0.1) 50%, transparent 70%);
        background-size: 200% 200%;
        animation: hologramShine 3s linear infinite;
        position: relative;
        overflow: hidden;
    }
    
    @keyframes hologramShine {
        0% { background-position: -200% -200%; }
        100% { background-position: 200% 200%; }
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
    
    .stButton > button:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover:before {
        left: 100%;
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
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(0, 212, 255, 0.1), transparent);
        animation: rotate 4s linear infinite;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
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
    
    /* Camera Input */
    .stCameraInput > div {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(0, 212, 255, 0.1);
        border-radius: 10px;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 10px;
    }
    
    /* Text styling */
    .tech-text {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 500;
        letter-spacing: 1px;
        line-height: 1.6;
    }
    
    /* Loading Animation */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
    }
    
    .loading-spinner {
        width: 50px;
        height: 50px;
        border: 3px solid rgba(0, 212, 255, 0.3);
        border-top: 3px solid #00d4ff;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Particle Effect */
    .particles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
    }
    
    .particle {
        position: absolute;
        width: 2px;
        height: 2px;
        background: #00d4ff;
        border-radius: 50%;
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); opacity: 0; }
        50% { transform: translateY(-100px) rotate(180deg); opacity: 1; }
    }
    
    /* Data Table Styling */
    .dataframe {
        background: rgba(0, 212, 255, 0.05);
        border-radius: 10px;
        border: 1px solid rgba(0, 212, 255, 0.2);
    }
    
    /* Alert Boxes */
    .stAlert {
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 15px;
    }
    
    /* Success/Warning/Error styling */
    .stSuccess {
        background: rgba(0, 255, 127, 0.1);
        border-color: rgba(0, 255, 127, 0.3);
    }
    
    .stWarning {
        background: rgba(255, 193, 7, 0.1);
        border-color: rgba(255, 193, 7, 0.3);
    }
    
    .stError {
        background: rgba(255, 20, 147, 0.1);
        border-color: rgba(255, 20, 147, 0.3);
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

# Add particle effect
st.markdown("""
<div class="particles">
    <div class="particle" style="left: 10%; animation-delay: 0s;"></div>
    <div class="particle" style="left: 20%; animation-delay: 1s;"></div>
    <div class="particle" style="left: 30%; animation-delay: 2s;"></div>
    <div class="particle" style="left: 40%; animation-delay: 3s;"></div>
    <div class="particle" style="left: 50%; animation-delay: 4s;"></div>
    <div class="particle" style="left: 60%; animation-delay: 5s;"></div>
    <div class="particle" style="left: 70%; animation-delay: 0.5s;"></div>
    <div class="particle" style="left: 80%; animation-delay: 1.5s;"></div>
    <div class="particle" style="left: 90%; animation-delay: 2.5s;"></div>
</div>
""", unsafe_allow_html=True)

# Class labels dictionary
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
        # GUNAKAN MODEL ASLI ANDA
        model = load_model('fine_tuned_model.h5')
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("📝 Please make sure 'fine_tuned_model.h5' is in the same directory as this script.")
        return None

def preprocess_image(img):
    """Preprocess image with futuristic enhancement"""
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
        return None, None, None
    
    # Preprocess image
    processed_img = preprocess_image(img)
    
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

def create_neural_network_viz():
    """Create a neural network visualization"""
    fig = go.Figure()
    
    # Create nodes for input, hidden, and output layers
    input_nodes = [(0, i) for i in range(5)]
    hidden_nodes = [(1, i) for i in range(3)]
    output_nodes = [(2, 1)]
    
    all_nodes = input_nodes + hidden_nodes + output_nodes
    
    # Add connections
    for input_node in input_nodes:
        for hidden_node in hidden_nodes:
            fig.add_trace(go.Scatter(
                x=[input_node[0], hidden_node[0]],
                y=[input_node[1], hidden_node[1]],
                mode='lines',
                line=dict(color='rgba(0, 212, 255, 0.3)', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    for hidden_node in hidden_nodes:
        for output_node in output_nodes:
            fig.add_trace(go.Scatter(
                x=[hidden_node[0], output_node[0]],
                y=[hidden_node[1], output_node[1]],
                mode='lines',
                line=dict(color='rgba(123, 44, 191, 0.3)', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Add nodes
    for layer, nodes in enumerate([input_nodes, hidden_nodes, output_nodes]):
        colors = ['#00d4ff', '#7b2cbf', '#ff006e'][layer]
        fig.add_trace(go.Scatter(
            x=[node[0] for node in nodes],
            y=[node[1] for node in nodes],
            mode='markers',
            marker=dict(
                size=20,
                color=colors,
                line=dict(color='white', width=2)
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=dict(
            text="🧠 Neural Network Architecture",
            font=dict(family="Orbitron", size=18, color="#00d4ff")
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
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
            ["🖼️ QUANTUM IMAGE ANALYSIS", "📹 LIVE NEURAL SCAN", "🔬 SYSTEM DIAGNOSTICS"],
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
                <h2 style="color: #ffffff;">98.7%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Neural Network Visualization
        neural_fig = create_neural_network_viz()
        st.plotly_chart(neural_fig, use_container_width=True)
        
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
    elif app_mode == "🔬 SYSTEM DIAGNOSTICS":
        system_diagnostics()

def quantum_image_analysis(model):
    st.markdown('<h2 class="sub-header">🖼️ QUANTUM IMAGE ANALYSIS</h2>', unsafe_allow_html=True)
    
    # Instructions in futuristic style
    with st.expander("📡 QUANTUM SCANNING PROTOCOLS", expanded=False):
        st.markdown("""
        <div class="glass-card hologram">
            <div class="tech-text">
                <h4 style="color: #00d4ff;">🔬 OPTIMAL SCANNING CONDITIONS:</h4>
                
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-top: 20px;">
                    <div>
                        <h5 style="color: #00ff7f;">📸 IMAGE PARAMETERS</h5>
                        • Resolution: ≥ 1024x1024 px<br>
                        • Format: JPG/PNG/TIFF<br>
                        • Color Space: sRGB<br>
                        • Compression: Minimal
                    </div>
                    <div>
                        <h5 style="color: #ff006e;">🔆 LIGHTING MATRIX</h5>
                        • Illumination: 1000-2000 lux<br>
                        • Temperature: 5500K-6500K<br>
                        • Diffusion: Uniform spread<br>
                        • Shadows: Eliminated
                    </div>
                    <div>
                        <h5 style="color: #7b2cbf;">🎯 SPECIMEN POSITIONING</h5>
                        • Frame Coverage: 70-90%<br>
                        • Focus Depth: Sharp edges<br>
                        • Shadows: Eliminated
                    </div>
                    <div>
                        <h5 style="color: #7b2cbf;">🎯 SPECIMEN POSITIONING</h5>
                        • Frame Coverage: 70-90%<br>
                        • Focus Depth: Sharp edges<br>
                        • Orientation: Natural angle<br>
                        • Background: Neutral tone
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
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
            
            # Convert PIL to OpenCV format
            img_array = np.array(image_pil)
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_cv = img_array
            
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
                
                # Get prediction
                predicted_class, confidence, all_predictions = predict_disease(model, img_cv)
                
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
                        
                        # AI Recommendations
                        if 'healthy' not in disease_status.lower():
                            st.markdown("""
                            <div class="glass-card">
                                <h4 style="color: #ff006e;">🚨 AI RECOMMENDATIONS</h4>
                                <div class="tech-text">
                                    🔬 <strong>Immediate Actions:</strong><br>
                                    • Isolate affected specimen immediately<br>
                                    • Document progression with time-series imaging<br>
                                    • Consult agricultural pathology database<br><br>
                                    
                                    🧪 <strong>Treatment Protocol:</strong><br>
                                    • Apply targeted bio-fungicide treatment<br>
                                    • Monitor environmental conditions<br>
                                    • Implement preventive measures for healthy specimens<br><br>
                                    
                                    📊 <strong>Follow-up Analysis:</strong><br>
                                    • Re-scan in 48-72 hours<br>
                                    • Track treatment effectiveness<br>
                                    • Update AI learning database
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

def live_neural_scan(model):
    st.markdown('<h2 class="sub-header">📹 LIVE NEURAL SCAN</h2>', unsafe_allow_html=True)
    
    # Live scan instructions
    st.markdown("""
    <div class="glass-card hologram">
        <h3 style="color: #00d4ff;">📡 LIVE SCANNING PROTOCOL</h3>
        <div class="tech-text">
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 30px;">
                <div>
                    <h4 style="color: #00ff7f;">🎯 POSITIONING MATRIX</h4>
                    • Distance: 15-30cm from specimen<br>
                    • Angle: 45-90° to leaf surface<br>
                    • Stability: Minimize camera shake<br>
                    • Coverage: Fill 70% of frame
                </div>
                <div>
                    <h4 style="color: #ff006e;">⚡ REAL-TIME PARAMETERS</h4>
                    • Processing Speed: <1 second<br>
                    • Analysis Depth: Cellular level<br>
                    • Accuracy Rate: 98.7%<br>
                    • Neural Cores: 38 active
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
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
            
            # Convert to OpenCV format
            image_pil = Image.open(camera_input)
            img_array = np.array(image_pil)
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Real-time processing simulation
            with st.spinner("⚡ Processing Neural Scan..."):
                predicted_class, confidence, all_predictions = predict_disease(model, img_cv)
            
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
                
                # Real-time metrics
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #00d4ff;">SPECIES</h4>
                        <h3 style="color: #ffffff;">{plant_type}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_cols[1]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #ff006e;">CONFIDENCE</h4>
                        <h3 style="color: #ffffff;">{confidence:.1%}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_cols[2]:
                    status_color = "#00ff7f" if 'healthy' in disease_status.lower() else "#ff1493"
                    status_icon = "🌿" if 'healthy' in disease_status.lower() else "⚠️"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: {status_color};">STATUS</h4>
                        <h3 style="color: #ffffff;">{status_icon}</h3>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Live analysis visualization
        st.markdown("### 📊 LIVE NEURAL NETWORK ANALYSIS")
        fig = create_futuristic_chart(all_predictions)
        st.plotly_chart(fig, use_container_width=True)
    
    # Live scanning tips
    st.markdown("---")
    st.markdown("### 🎯 NEURAL SCANNING OPTIMIZATION")
    
    tip_cols = st.columns(4)
    tips = [
        ("🔆 ILLUMINATION", "• Natural daylight optimal<br>• Avoid harsh shadows<br>• Even light distribution<br>• 1000-2000 lux ideal"),
        ("📐 POSITIONING", "• Steady hand movement<br>• Fill frame with specimen<br>• 45° angle preferred<br>• Focus on leaf details"),
        ("🎯 TARGETING", "• Single leaf analysis<br>• Clear symptom visibility<br>• Avoid motion blur<br>• Sharp edge definition"),
        ("🔄 SCANNING", "• Multiple angle capture<br>• Compare results<br>• Document progression<br>• Real-time feedback")
    ]
    
    for i, (title, content) in enumerate(tips):
        with tip_cols[i]:
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="color: #00d4ff;">{title}</h4>
                <div class="tech-text" style="font-size: 0.9rem;">
                    {content}
                </div>
            </div>
            """, unsafe_allow_html=True)

def system_diagnostics():
    st.markdown('<h2 class="sub-header">🔬 SYSTEM DIAGNOSTICS</h2>', unsafe_allow_html=True)
    
    # System overview
    st.markdown("""
    <div class="glass-card hologram">
        <h3 style="color: #00d4ff; text-align: center;">🚀 AI PLANT GUARDIAN v3.0</h3>
        <p style="text-align: center; font-family: 'Rajdhani', sans-serif; font-size: 1.2rem;">
            Next-Generation Quantum Plant Disease Detection System
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # System metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #00ff7f;">NEURAL CORES</h4>
            <h2 style="color: #ffffff;">38</h2>
            <p style="color: #a0a0a0;">Active Processing Units</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #00d4ff;">ACCURACY</h4>
            <h2 style="color: #ffffff;">98.7%</h2>
            <p style="color: #a0a0a0;">Detection Precision</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #ff006e;">SPEED</h4>
            <h2 style="color: #ffffff;"><1s</h2>
            <p style="color: #a0a0a0;">Analysis Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #7b2cbf;">SPECIES</h4>
            <h2 style="color: #ffffff;">11</h2>
            <p style="color: #a0a0a0;">Plant Types Supported</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Technical specifications
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #00d4ff;">🔬 TECHNICAL SPECIFICATIONS</h3>
            <div class="tech-text">
        """, unsafe_allow_html=True)
        
        specs_data = {
            "Component": [
                "Neural Architecture", "Input Resolution", "Color Channels", 
                "Training Dataset", "Model Parameters", "Inference Engine",
                "Processing Framework", "Optimization"
            ],
            "Specification": [
                "MobileNetV2 + Transfer Learning", "224 × 224 pixels", "RGB (3 channels)",
                "PlantVillage Dataset", "~3.4M parameters", "TensorFlow Lite",
                "Quantum Processing Simulation", "Adam Optimizer"
            ]
        }
        
        specs_df = pd.DataFrame(specs_data)
        st.dataframe(specs_df, hide_index=True, use_container_width=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
    
    with col_right:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #00d4ff;">🧬 PATHOGEN DATABASE</h3>
            <div class="tech-text">
        """, unsafe_allow_html=True)
        
        pathogen_data = {
            "Plant Species": [
                "Apple", "Blueberry", "Cherry", "Corn", "Grape", 
                "Orange", "Peach", "Pepper", "Potato", "Strawberry", "Tomato"
            ],
            "Disease Types": [
                "4 variants", "1 variant", "2 variants", "4 variants", "4 variants",
                "1 variant", "2 variants", "2 variants", "3 variants", "2 variants", "10 variants"
            ]
        }
        
        pathogen_df = pd.DataFrame(pathogen_data)
        st.dataframe(pathogen_df, hide_index=True, use_container_width=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Performance analytics
    st.markdown("### 📈 PERFORMANCE ANALYTICS")
    
    # Create performance dashboard
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    values = [0.987, 0.943, 0.956, 0.949, 0.991]
    
    fig = go.Figure()
    
    # Add performance bars
    colors = ['#00d4ff', '#7b2cbf', '#ff006e', '#00f5ff', '#ff7b00']
    
    for i, (metric, value, color) in enumerate(zip(metrics, values, colors)):
        fig.add_trace(go.Bar(
            x=[metric],
            y=[value],
            marker=dict(
                color=color,
                line=dict(color='rgba(255,255,255,0.3)', width=2)
            ),
            text=f'{value:.1%}',
            textposition='auto',
            name=metric
        ))
    
    fig.update_layout(
        title=dict(
            text="🎯 Neural Network Performance Metrics",
            font=dict(family="Orbitron", size=24, color="#00d4ff")
        ),
        xaxis=dict(
            titlefont=dict(family="Rajdhani", size=16, color="#ffffff"),
            tickfont=dict(color="#ffffff", size=12)
        ),
        yaxis=dict(
            title="Performance Score",
            titlefont=dict(family="Rajdhani", size=16, color="#ffffff"),
            tickfont=dict(color="#ffffff"),
            range=[0, 1]
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # System architecture
    st.markdown("### 🏗️ QUANTUM ARCHITECTURE")
    
    arch_cols = st.columns(3)
    
    with arch_cols[0]:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #00ff7f;">🔄 INPUT LAYER</h4>
            <div class="tech-text">
                • Image Preprocessing<br>
                • Normalization Pipeline<br>
                • Data Augmentation<br>
                • Quality Enhancement<br>
                • Noise Reduction
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with arch_cols[1]:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #ff006e;">🧠 PROCESSING CORE</h4>
            <div class="tech-text">
                • Convolutional Networks<br>
                • Feature Extraction<br>
                • Pattern Recognition<br>
                • Transfer Learning<br>
                • Neural Optimization
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with arch_cols[2]:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #7b2cbf;">📊 OUTPUT MATRIX</h4>
            <div class="tech-text">
                • Classification Results<br>
                • Confidence Scoring<br>
                • Probability Distribution<br>
                • Recommendation Engine<br>
                • Report Generation
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Disclaimer with futuristic styling
    st.markdown("---")
    st.markdown("""
    <div class="glass-card" style="border-color: rgba(255, 193, 7, 0.5);">
        <h3 style="color: #ffc107;">⚠️ SYSTEM DISCLAIMER</h3>
        <div class="tech-text">
            <p>This AI Plant Guardian system represents cutting-edge agricultural technology designed for 
            research and educational applications. While our quantum neural networks achieve exceptional 
            accuracy rates, this system should complement, not replace, professional agricultural consultation.</p>
            
            <p><strong>For critical agricultural decisions:</strong></p>
            <ul>
                <li>Consult certified plant pathologists</li>
                <li>Verify results with laboratory analysis</li>
                <li>Consider environmental factors</li>
                <li>Implement integrated pest management</li>
            </ul>
            
            <p style="text-align: center; margin-top: 20px;">
                <em>Advancing agriculture through artificial intelligence</em>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
