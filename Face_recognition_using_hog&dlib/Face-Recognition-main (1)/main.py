"""
Main entry point for Face Recognition Application
Provides menu interface to encode faces, recognize from images, or use webcam

Usage:
    python main.py
"""

import os
import sys
import subprocess

# ---------------------------------------------------------------------
# Anchor every sibling-script path to THIS file's own folder, not the
# current working directory. Without this, subprocess.run(["encode_faces.py"])
# only finds that file if you happen to launch python from this exact
# folder - run it from an IDE, a shortcut, or any other cwd, and it
# fails with "No such file or directory" even though the file is sitting
# right next to main.py on disk.
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")


def script_path(filename):
    """Builds an absolute path to a sibling script next to this file."""
    return os.path.join(BASE_DIR, filename)


def print_menu():
    """Display main menu"""
    print("\n" + "=" * 60)
    print("         FACE RECOGNITION APPLICATION")
    print("=" * 60)
    print("1. Encode new faces (from known_faces/ directory)")
    print("2. Recognize faces in an image")
    print("3. Recognize faces from webcam (real-time)")
    print("4. View configuration")
    print("5. Exit")
    print("=" * 60)


def encode_menu():
    """Handle encoding faces"""
    print("\n[ENCODING FACES]")
    print("This will scan the 'known_faces' directory and create a database")
    print("of face encodings.\n")
    print("Directory structure should be:")
    print(f"  {KNOWN_FACES_DIR}\\")
    print("    ├── person1/")
    print("    │   ├── photo1.jpg")
    print("    │   ├── photo2.jpg")
    print("    │   └── ...")
    print("    ├── person2/")
    print("    │   ├── photo1.jpg")
    print("    │   └── ...")
    print("    └── ...\n")

    if not os.path.isdir(KNOWN_FACES_DIR):
        print(f"Note: that folder doesn't exist yet at {KNOWN_FACES_DIR}")
        print("Create it and add a subfolder per person before encoding.\n")

    response = input("Continue with encoding? (yes/no): ").strip().lower()
    if response in ['yes', 'y']:
        print("Starting encoding process...\n")
        encode_script = script_path("encode_faces.py")
        try:
            subprocess.run([sys.executable, encode_script], check=True)
        except subprocess.CalledProcessError:
            print("Error: Encoding process failed")
        except FileNotFoundError:
            print(f"Error: encode_faces.py not found at {encode_script}")
    else:
        print("Cancelled.")


def recognize_image_menu():
    """Handle image recognition"""
    print("\n[RECOGNIZE FACES IN IMAGE]")
    image_path = input("Enter path to image file: ").strip().strip('"')

    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return

    print("Processing image...\n")
    recognize_script = script_path("recognize_image.py")
    try:
        subprocess.run([sys.executable, recognize_script, image_path], check=True)
    except subprocess.CalledProcessError:
        print("Error: Image recognition failed")
    except FileNotFoundError:
        print(f"Error: recognize_image.py not found at {recognize_script}")


def recognize_webcam_menu():
    """Handle webcam recognition"""
    print("\n[RECOGNIZE FACES FROM WEBCAM]")
    print("Starting real-time face recognition from webcam...")
    print("Controls:")
    print("  ESC or 'q' - Exit")
    print("  Space - Pause/Resume")
    print()

    webcam_script = script_path("recognize_webcam.py")
    try:
        subprocess.run([sys.executable, webcam_script], check=True)
    except subprocess.CalledProcessError:
        pass  # User likely just exited normally
    except FileNotFoundError:
        print(f"Error: recognize_webcam.py not found at {webcam_script}")


def view_config_menu():
    """Display current configuration"""
    print("\n[CURRENT CONFIGURATION]")

    # BASE_DIR is already on sys.path automatically (Python adds the
    # main script's own folder to sys.path), so this import works
    # regardless of current working directory.
    try:
        import config
    except ImportError:
        print(f"Error: config.py not found in {BASE_DIR}")
        return

    settings = [
        ("Known Faces Directory", config.KNOWN_FACES_DIR),
        ("Encodings File", config.ENCODINGS_FILE),
        ("Face Detection Model", config.MODEL),
        ("Tolerance (0-1)", config.TOLERANCE),
        ("Process Every N Frames", config.PROCESS_EVERY_N_FRAMES),
        ("Frame Size", f"{config.FRAME_WIDTH}x{config.FRAME_HEIGHT}"),
    ]

    print("\nCurrent Settings:")
    for setting_name, value in settings:
        print(f"  {setting_name:.<40} {value}")

    print("\nTo change settings, edit config.py")


def main():
    """Main menu loop"""
    print("\n" + "=" * 60)
    print("Face Recognition App - Starting up...")
    print("=" * 60)

    # Check dependencies
    try:
        import face_recognition
        import cv2
        import imutils
        print("✓ All dependencies found")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    print()

    while True:
        try:
            print_menu()
            choice = input("Select option (1-5): ").strip()

            if choice == '1':
                encode_menu()
            elif choice == '2':
                recognize_image_menu()
            elif choice == '3':
                recognize_webcam_menu()
            elif choice == '4':
                view_config_menu()
            elif choice == '5':
                print("\nExiting Face Recognition App. Goodbye!")
                break
            else:
                print("Invalid option. Please enter 1-5.")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Returning to menu...")


if __name__ == "__main__":
    main()