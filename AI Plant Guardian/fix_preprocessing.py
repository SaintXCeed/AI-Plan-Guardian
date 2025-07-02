"""
🔧 PERBAIKAN PREPROCESSING
=========================

Jika prediksi masih salah, kemungkinan masalah di preprocessing.
Coba variasi preprocessing ini:
"""

import tensorflow as tf
import numpy as np
import cv2

def preprocess_option_1(img):
    """Preprocessing Option 1: MobileNetV2 standard"""
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_preprocessed

def preprocess_option_2(img):
    """Preprocessing Option 2: Manual normalization 0-1"""
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)
    img_preprocessed = img_array.astype('float32') / 255.0
    return img_preprocessed

def preprocess_option_3(img):
    """Preprocessing Option 3: Manual normalization -1 to 1"""
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)
    img_preprocessed = (img_array.astype('float32') / 127.5) - 1.0
    return img_preprocessed

def preprocess_option_4(img):
    """Preprocessing Option 4: RGB conversion + MobileNetV2"""
    # Convert BGR to RGB (jika dari OpenCV)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_preprocessed

# Ganti fungsi preprocess_image di streamlit_app.py dengan salah satu option di atas
# yang sesuai dengan preprocessing saat training
