from predict import load_trained_model, process_batch
import os

def main():
    # Load the model
    model = load_trained_model()
    if not model:
        return

    # Get the directory containing MRI images
    image_dir = input("Enter the directory containing MRI images: ")
    
    # Get all image files in the directory
    image_extensions = ('.jpg', '.jpeg', '.png', '.tiff', '.bmp')
    image_paths = [
        os.path.join(image_dir, f) 
        for f in os.listdir(image_dir) 
        if f.lower().endswith(image_extensions)
    ]

    if not image_paths:
        print("No image files found in the specified directory")
        return

    # Process all images
    results = process_batch(model, image_paths)

    # Print results
    print("\nBatch Prediction Results:")
    for result in results:
        print(f"\nImage: {result['image_path']}")
        print(f"Predicted Class: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("Class Probabilities:")
        for class_name, prob in result['class_probabilities'].items():
            print(f"  {class_name}: {prob:.2%}")

if __name__ == "__main__":
    main() 