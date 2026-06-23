#!/usr/bin/env python3
"""
Quick test to verify Face Recognition App installation and setup
Run this to ensure everything is configured correctly
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    modules_to_test = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('imutils', 'Imutils'),
        ('sklearn', 'scikit-learn'),
        ('scipy', 'SciPy'),
    ]
    
    all_ok = True
    for module_name, display_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError as e:
            print(f"  ✗ {display_name}: {e}")
            all_ok = False
    
    return all_ok

def test_directories():
    """Test that required directories exist"""
    print("\nChecking directories...")
    required_dirs = [
        'known_faces',
        'unknown_faces',
        'output',
        'encodings'
    ]
    
    all_ok = True
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ (not found)")
            all_ok = False
    
    return all_ok

def test_files():
    """Test that required Python files exist"""
    print("\nChecking Python files...")
    required_files = [
        'main.py',
        'config.py',
        'face_utils.py',
        'encode_faces.py',
        'recognize_image.py',
        'recognize_webcam.py',
    ]
    
    all_ok = True
    for file_name in required_files:
        if os.path.isfile(file_name):
            print(f"  ✓ {file_name}")
        else:
            print(f"  ✗ {file_name} (not found)")
            all_ok = False
    
    return all_ok

def main():
    """Run all tests"""
    print("=" * 60)
    print("FACE RECOGNITION APP - INSTALLATION TEST")
    print("=" * 60)
    print()
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Directories", test_directories()))
    results.append(("Files", test_files()))
    
    print("\n" + "=" * 60)
    if all(r[1] for r in results):
        print("✓ ALL TESTS PASSED!")
        print("\nYou can now run:")
        print("  python main.py              # Main menu")
        print("  python encode_faces.py      # Encode known faces")
        print("  python recognize_image.py   # Recognize faces in image")
        print("  python recognize_webcam.py  # Real-time webcam recognition")
        print("\nNext steps:")
        print("1. Add your reference images to known_faces/person_name/")
        print("2. Run encode_faces.py to create the encoding database")
        print("3. Run recognize_webcam.py or recognize_image.py")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease fix the issues above and try again:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
