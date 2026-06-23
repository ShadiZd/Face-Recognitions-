"""
Script to recognize faces in images (OpenCV DNN version)
Detects faces, matches them against the database, and saves output

Usage:
    python recognize_image.py <image_path>
    python recognize_image.py unknown_faces/test.jpg
"""

import os
import sys
import cv2
from face_utils import (
    load_encoded_faces,
    recognize_faces_in_image,
    draw_face_boxes,
    load_image,
    save_image
)
import config


def recognize_image(image_path):
    """
    Recognize faces in a single image
    
    Args:
        image_path: Path to image file
    """
    print(f"Processing image: {image_path}")
    
    # Load the image
    image = load_image(image_path)
    if image is None:
        print("Error: Could not load image")
        return False
    
    print(f"  Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Load known faces database
    known_encodings, known_names = load_encoded_faces()
    if len(known_encodings) == 0:
        print("Error: No known faces in database. Run encode_faces.py first.")
        return False
    
    print(f"  Database contains: {len(set(known_names))} known people, {len(known_encodings)} total encodings")
    
    # Recognize faces in image
    print("  Detecting and recognizing faces...")
    face_locations, face_labels, face_distances = recognize_faces_in_image(
        image, known_encodings, known_names
    )
    
    print(f"  Faces found: {len(face_locations)}")
    
    # Draw boxes and labels on image
    output_image = draw_face_boxes(image, face_locations, face_labels, 
                                   show_distance=True, face_distances=face_distances)
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(config.OUTPUT_DIR, f"{base_name}_recognized.jpg")
    
    # Save output image
    if save_image(output_image, output_path):
        print(f"Result saved to: {output_path}")
        
        # Print recognition results
        print("\n  Recognition Results:")
        for i, (label, distance) in enumerate(zip(face_labels, face_distances)):
            print(f"    Face {i+1}: {label} (distance: {distance:.4f})")
        
        return True
    else:
        print("Error: Failed to save output image")
        return False


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python recognize_image.py <image_path>")
        print("\nExamples:")
        print("  python recognize_image.py unknown_faces/test.jpg")
        print("  python recognize_image.py photo.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    success = recognize_image(image_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
