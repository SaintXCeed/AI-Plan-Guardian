"""
🔍 DEBUG SCRIPT - Untuk mengecek masalah prediksi
=================================================

Jalankan script ini untuk mengidentifikasi masalah:
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image

# Class labels yang sama dengan training
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

def debug_model():
    """Debug model loading dan prediksi"""
    print("🔍 DEBUGGING AI PLANT GUARDIAN")
    print("=" * 50)
    
    # 1. Cek model file
    try:
        model = load_model('fine_tuned_model.h5')
        print("✅ Model loaded successfully")
        print(f"📊 Model input shape: {model.input_shape}")
        print(f"📊 Model output shape: {model.output_shape}")
        print(f"📊 Number of parameters: {model.count_params():,}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # 2. Cek preprocessing function
    def preprocess_image_debug(img_path):
        """Debug preprocessing"""
        print(f"\n🔬 Preprocessing: {img_path}")
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print("❌ Image loading failed")
            return None
            
        print(f"📐 Original shape: {img.shape}")
        
        # Resize
        img_resized = cv2.resize(img, (224, 224))
        print(f"📐 Resized shape: {img_resized.shape}")
        
        # Convert to array and expand dimensions
        img_array = np.expand_dims(img_resized, axis=0)
        print(f"📐 Array shape: {img_array.shape}")
        
        # Preprocess - PENTING: Gunakan preprocessing yang sama dengan training!
        img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        print(f"📐 Preprocessed shape: {img_preprocessed.shape}")
        print(f"📊 Preprocessed range: [{img_preprocessed.min():.3f}, {img_preprocessed.max():.3f}]")
        
        return img_preprocessed
    
    # 3. Test dengan gambar sample
    # Ganti dengan path gambar test Anda
    test_image_path = "test_image.jpg"  # Ganti dengan gambar test Anda
    
    try:
        processed_img = preprocess_image_debug(test_image_path)
        if processed_img is not None:
            # Prediksi
            print("\n🤖 Making prediction...")
            predictions = model.predict(processed_img, verbose=1)
            
            print(f"📊 Prediction shape: {predictions.shape}")
            print(f"📊 Prediction sum: {predictions.sum():.6f}")
            
            # Top 5 predictions
            top_5_indices = np.argsort(predictions[0])[-5:][::-1]
            print("\n🏆 Top 5 Predictions:")
            for i, idx in enumerate(top_5_indices):
                class_name = CLASS_LABELS.get(idx, f"Unknown_{idx}")
                confidence = predictions[0][idx]
                print(f"{i+1}. {class_name}: {confidence:.4f} ({confidence*100:.2f}%)")
                
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
    
    # 4. Cek class mapping
    print(f"\n📋 Total classes in model: {len(CLASS_LABELS)}")
    print("📋 Sample class mappings:")
    for i in range(0, min(10, len(CLASS_LABELS))):
        print(f"   {i}: {CLASS_LABELS[i]}")

def check_preprocessing_consistency():
    """Cek konsistensi preprocessing dengan training"""
    print("\n🔍 CHECKING PREPROCESSING CONSISTENCY")
    print("=" * 50)
    
    print("❓ Pertanyaan untuk Anda:")
    print("1. Apakah saat training Anda menggunakan:")
    print("   - tf.keras.applications.mobilenet_v2.preprocess_input() ?")
    print("   - Resize ke (224, 224) ?")
    print("   - Normalisasi manual (0-1 atau -1 to 1) ?")
    print("")
    print("2. Apakah urutan class labels sama dengan saat training?")
    print("3. Apakah format input gambar sama (RGB vs BGR)?")
    print("")
    print("💡 Jika ada perbedaan, itu penyebab prediksi salah!")

if __name__ == "__main__":
    debug_model()
    check_preprocessing_consistency()
