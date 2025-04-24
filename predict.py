import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import cv2

def load_trained_model(model_path='efficientnetB0.keras'):
    """
    Load the trained EfficientNetB0 model
    """
    try:
        model = load_model(model_path)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def preprocess_image(img_path, target_size=(150, 150)):
    """
    Preprocess the image for prediction - matches EfficientNetB0 training exactly
    """
    try:
        # Load image using OpenCV
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image at {img_path}")
            
        # Convert BGR to RGB (OpenCV loads as BGR, but model expects RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        # Resize to match training size
        img = cv2.resize(img, target_size)
        
        # Convert to float32 and add batch dimension
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply EfficientNet preprocessing
        img_array = preprocess_input(img_array)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        raise

def predict_tumor(model, img_path):
    """
    Make prediction on a single image - matches notebook implementation exactly
    """
    try:
        # Preprocess the image
        processed_image = preprocess_image(img_path)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)  # Disable prediction progress bar
        
        # Get class names in the same order as notebook
        class_names = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']
        
        # Research paper links for each tumor type
        research_links = {
            'glioma_tumor': 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2887670/',  # Glioma-specific research
            'meningioma_tumor': 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8068132/',  # Meningioma-specific research
            'pituitary_tumor': 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8068132/',  # Pituitary-specific research
            'no_tumor': None
        }
        
        # Get the predicted class and confidence
        predicted_class_index = np.argmax(prediction[0])
        predicted_class = class_names[predicted_class_index]
        confidence = float(prediction[0][predicted_class_index])
        
        # Get probabilities for all classes
        class_probabilities = {}
        for i, class_name in enumerate(class_names):
            class_probabilities[class_name] = float(prediction[0][i])
        
        return {
            'prediction': predicted_class,
            'confidence': confidence,
            'class_probabilities': class_probabilities,
            'research_link': research_links[predicted_class]
        }
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        raise

def process_batch(model, image_paths):
    """
    Process multiple images and return predictions
    """
    results = []
    for img_path in image_paths:
        try:
            result = predict_tumor(model, img_path)
            result['image_path'] = img_path
            results.append(result)
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            continue
    return results

if __name__ == "__main__":
    # Example usage
    model = load_trained_model()
    if model:
        # Test prediction
        test_image = input("Enter the path to your MRI image: ")
        if os.path.exists(test_image):
            result = predict_tumor(model, test_image)
            print("\nPrediction Results:")
            print(f"Predicted Class: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print("\nClass Probabilities:")
            for class_name, prob in result['class_probabilities'].items():
                print(f"{class_name}: {prob:.2%}")
            if result['research_link']:
                print(f"\nResearch Paper: {result['research_link']}")
        else:
            print(f"Test image not found: {test_image}") 