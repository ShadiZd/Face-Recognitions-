"""
main.py
--------
Interactive menu for the KNN face-recognition project.

This file does NOT reimplement any recognition logic. It only calls
the functions that already exist in T_T.py:
    - T_T.train()
    - T_T.predict()
    - T_T.show_prediction_labels_on_image()

The only extra package used anywhere here is `imageio-ffmpeg`, and only
for ONE job: locating the ffmpeg binary so we can grab a single still
frame from the webcam by calling it directly. Nothing in face_recognition
or PIL can open a camera by itself, so something has to do that. Once
the snapshot is saved as a .jpg, it's handed straight to T_T.predict()
and T_T.show_prediction_labels_on_image() exactly like any other photo.
No cv2, no extra face-detection library.

Folder structure expected (same as T_T.py's docstring):
    knn_examples/
    ├── train/
    │   ├── person1/
    │   │   ├── img1.jpg
    │   │   └── img2.jpg
    │   └── person2/
    │       └── img1.jpg
    └── test/
        ├── photo1.jpg
        └── photo2.jpg
"""

import os
from PIL import Image
import T_T  # T_T.py must be in the same folder as this file

# ---------------------------------------------------------------------
# Paths - anchored to this script's own folder, NOT the current working
# directory. Without this, running main.py from a different folder (an
# IDE's "Run" button, a shortcut, a terminal cd'd elsewhere, etc.) would
# make these relative paths point to the wrong place even though the
# knn_examples folder is right next to main.py on disk.
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, "knn_examples", "train")
TEST_DIR = os.path.join(BASE_DIR, "knn_examples", "test")
MODEL_PATH = os.path.join(BASE_DIR, "trained_knn_model.clf")
SNAPSHOT_PATH = os.path.join(BASE_DIR, "webcam_snapshot.jpg")

# Minimum confidence (0-100) a face needs to be labeled with a name
# instead of "unknown". Raise this to be stricter, lower it to be
# more lenient.
MIN_CONFIDENCE = 70


def train_and_test():
    print("\n--- Training the KNN model ---")

    if not os.path.isdir(TRAIN_DIR):
        print(f"Could not find training folder: {TRAIN_DIR}")
        print("Expected structure: knn_examples/train/<person_name>/<photos>.jpg")
        return

    T_T.train(TRAIN_DIR, model_save_path=MODEL_PATH, n_neighbors=2, verbose=True)
    print("Training complete. Model saved to:", MODEL_PATH)

    if not os.path.isdir(TEST_DIR):
        print(f"No test folder found at {TEST_DIR}, skipping the test step.")
        return

    print("\n--- Testing the model on the test folder ---")
    for image_file in os.listdir(TEST_DIR):
        full_path = os.path.join(TEST_DIR, image_file)
        if not os.path.isfile(full_path):
            continue

        print(f"\nLooking for faces in {image_file}")
        predictions = T_T.predict(full_path, model_path=MODEL_PATH, min_confidence=MIN_CONFIDENCE)

        if not predictions:
            print(" - No faces found.")
            continue

        for name, (top, right, bottom, left), confidence in predictions:
            print(f" - Found {name} at ({left}, {top}) - {confidence:.1f}% confidence")

        T_T.show_prediction_labels_on_image(full_path, predictions)


def test_on_photo():
    print("\n--- Test on a single photo ---")

    if not os.path.isfile(MODEL_PATH):
        print(f"No trained model found at {MODEL_PATH}. Run option 1 first.")
        return

    photo_path = input("Enter the full path to the photo: ").strip().strip('"')

    if not os.path.isfile(photo_path):
        print("That file does not exist. Check the path and try again.")
        return

    predictions = T_T.predict(photo_path, model_path=MODEL_PATH, min_confidence=MIN_CONFIDENCE)

    if not predictions:
        print("No faces found in this photo.")
        return

    for name, (top, right, bottom, left), confidence in predictions:
        print(f" - Found {name} at ({left}, {top}) - {confidence:.1f}% confidence")

    T_T.show_prediction_labels_on_image(photo_path, predictions)


def list_dshow_video_devices():
    """
    Asks ffmpeg (via imageio-ffmpeg) which Windows DirectShow video
    devices actually exist. Needed because '<video0>' is not reliable:
    Windows often lists virtual/software entries (e.g. 'ffdshow') ahead
    of the real webcam, so the index does not match the real camera.
    """
    import subprocess
    import imageio_ffmpeg

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run(
        [ffmpeg_exe, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
        stderr=subprocess.PIPE, text=True
    )

    devices = []
    for line in result.stderr.splitlines():
        if '"' in line and "(video)" in line:
            devices.append(line.split('"')[1])
    return devices


def capture_webcam_photo(camera_name, save_path, warmup_seconds=1):
    """
    Grabs exactly one frame from a named DirectShow webcam by calling
    ffmpeg directly (the binary bundled with imageio-ffmpeg) and piping
    a single JPEG frame back over stdout. This sidesteps imageio's own
    '<video=...>' URI handling, which has proven unreliable with some
    camera name/driver combinations on Windows.

    warmup_seconds tells ffmpeg to read a moment before grabbing the
    frame, so the camera's auto-exposure/focus has time to settle
    instead of capturing a dark frame from the instant it powers on.
    """
    import subprocess
    import io
    import imageio_ffmpeg

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg_exe,
        "-f", "dshow",
        "-i", f"video={camera_name}",
        "-ss", str(warmup_seconds),
        "-frames:v", "1",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "-",
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0 or not result.stdout:
        error_lines = result.stderr.decode(errors="ignore").strip().splitlines()
        raise RuntimeError(error_lines[-1] if error_lines else "Unknown ffmpeg error")

    image = Image.open(io.BytesIO(result.stdout)).convert("RGB")
    image.save(save_path)


def test_on_webcam():
    print("\n--- Test using a webcam snapshot ---")

    if not os.path.isfile(MODEL_PATH):
        print(f"No trained model found at {MODEL_PATH}. Run option 1 first.")
        return

    try:
        import imageio_ffmpeg  # noqa: F401 - just confirming it's installed
    except ImportError:
        print("This feature needs the 'imageio-ffmpeg' package.")
        print("Install with:   pip install imageio-ffmpeg")
        return

    try:
        devices = list_dshow_video_devices()
    except Exception as e:
        print("Could not list camera devices:", e)
        return

    # Skip known virtual/software entries that aren't real cameras.
    blocked_keywords = ("ffdshow", "screen-capture-recorder", "obs", "virtual")
    real_devices = [d for d in devices if not any(b in d.lower() for b in blocked_keywords)]

    if not real_devices:
        print("No real webcam found. Devices seen by ffmpeg:", devices)
        return

    if len(real_devices) == 1:
        camera_name = real_devices[0]
    else:
        print("Multiple cameras found:")
        for i, device_name in enumerate(real_devices, start=1):
            print(f"  {i}. {device_name}")
        choice = input(f"Pick a camera (1-{len(real_devices)}): ").strip()
        try:
            camera_name = real_devices[int(choice) - 1]
        except (ValueError, IndexError):
            print("Invalid choice.")
            return

    print(f"Using camera: {camera_name}")

    import time
    print("Get ready, taking the photo in...")
    for n in (3, 2, 1):
        print(n)
        time.sleep(1)

    try:
        capture_webcam_photo(camera_name, SNAPSHOT_PATH, warmup_seconds=1)
    except Exception as e:
        print("Could not access the webcam:", e)
        print("Check that a camera is connected and not in use by another app.")
        return

    print("Snapshot saved to:", SNAPSHOT_PATH)

    predictions = T_T.predict(SNAPSHOT_PATH, model_path=MODEL_PATH, min_confidence=MIN_CONFIDENCE)

    if not predictions:
        print("No faces found in the snapshot.")
        return

    for name, (top, right, bottom, left), confidence in predictions:
        print(f" - Found {name} at ({left}, {top}) - {confidence:.1f}% confidence")

    T_T.show_prediction_labels_on_image(SNAPSHOT_PATH, predictions)


def main_menu():
    while True:
        print("\n========== Face Recognition (KNN) ==========")
        print("1. Train and test the model")
        print("2. Test the model on a photo (enter path)")
        print("3. Test using webcam (snapshot)")
        print("4. Exit")

        choice = input("Choose an option (1-4): ").strip()

        if choice == "1":
            train_and_test()
        elif choice == "2":
            test_on_photo()
        elif choice == "3":
            test_on_webcam()
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Please enter a number between 1 and 4.")


if __name__ == "__main__":
    main_menu()