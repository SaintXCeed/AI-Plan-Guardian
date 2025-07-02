"""
🔍 DEBUG SCRIPT YANG BENAR - Berdasarkan kode training Anda
============================================================
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import os

def get_class_labels_from_training():
    """
    Berdasarkan kode training Anda, class labels dibuat dengan:
    class_labels_dict = {class_label: idx for idx, class_label in enumerate(df['label'].unique())}
    
    Ini berarti urutan class tergantung pada urutan folder di dataset.
    """
    # Class labels sesuai urutan di kode training Anda
    class_labels_dict = {
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
    return class_labels_dict

def preprocess_image_exact_training(img_path):
    """
    Preprocessing PERSIS seperti di training Anda:
    1. Load dengan keras image.load_img (bukan cv2)
    2. Convert ke array
    3. Expand dims
    4. mobilenet_v2.preprocess_input
    """
    from tensorflow.keras.preprocessing import image
    
    print(f"🔬 Processing: {img_path}")
    
    # Load image seperti di training (target_size=(224, 224))
    img = image.load_img(img_path, target_size=(224, 224))
    print(f"✅ Image loaded successfully")
    
    # Convert to array
    img_array = image.img_to_array(img)
    print(f"📐 Image array shape: {img_array.shape}")
    
    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)
    print(f"📐 Expanded shape: {img_array.shape}")
    
    # Preprocess - SAMA PERSIS dengan training
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    print(f"📐 Preprocessed shape: {img_preprocessed.shape}")
    print(f"📊 Value range: [{img_preprocessed.min():.3f}, {img_preprocessed.max():.3f}]")
    
    return img_preprocessed

def test_model_with_sample():
    """Test model dengan gambar sample"""
    print("🔍 TESTING MODEL WITH EXACT TRAINING PREPROCESSING")
    print("=" * 60)
    
    # Load model
    try:
        model = load_model('fine_tuned_model.h5')
        print("✅ Model loaded successfully")
        print(f"📊 Input shape: {model.input_shape}")
        print(f"📊 Output shape: {model.output_shape}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # Cari gambar test di folder current
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        import glob
        test_images.extend(glob.glob(ext))
    
    if not test_images:
        print("❌ No test images found in current directory")
        print("💡 Please add a test image (jpg/png) to current folder")
        return
    
    # Test dengan gambar pertama yang ditemukan
    test_image = test_images[0]
    print(f"🖼️ Testing with: {test_image}")
    
    try:
        # Preprocess dengan metode yang sama dengan training
        processed_img = preprocess_image_exact_training(test_image)
        
        # Predict
        print("\n🤖 Making prediction...")
        predictions = model.predict(processed_img, verbose=0)
        
        # Get class labels
        class_labels = get_class_labels_from_training()
        
        # Show top 5 predictions
        top_5_indices = np.argsort(predictions[0])[-5:][::-1]
        print("\n🏆 TOP 5 PREDICTIONS:")
        print("-" * 50)
        for i, idx in enumerate(top_5_indices):
            class_name = class_labels.get(idx, f"Unknown_{idx}")
            confidence = predictions[0][idx]
            plant = class_name.split('___')[0]
            disease = class_name.split('___')[1] if '___' in class_name else 'Unknown'
            print(f"{i+1}. {plant} - {disease}")
            print(f"   Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            print()
            
        # Show predicted class
        predicted_idx = np.argmax(predictions[0])
        predicted_class = class_labels.get(predicted_idx, f"Unknown_{predicted_idx}")
        predicted_confidence = predictions[0][predicted_idx]
        
        print("🎯 FINAL PREDICTION:")
        print("=" * 30)
        print(f"Class: {predicted_class}")
        print(f"Confidence: {predicted_confidence:.4f} ({predicted_confidence*100:.2f}%)")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()

def create_test_image():
    """Create a simple test image if none exists"""
    import numpy as np
    from PIL import Image
    
    # Create a simple green leaf-like image
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)
    img_array[:, :, 1] = 100  # Green channel
    img_array[50:174, 50:174, 1] = 200  # Brighter green center
    
    img = Image.fromarray(img_array)
    img.save('test_leaf.jpg')
    print("✅ Created test_leaf.jpg for testing")

if __name__ == "__main__":
    # Check if test images exist, if not create one
    import glob
    test_images = glob.glob('*.jpg') + glob.glob('*.jpeg') + glob.glob('*.png')
    if not test_images:
        create_test_image()
    
    test_model_with_sample()
