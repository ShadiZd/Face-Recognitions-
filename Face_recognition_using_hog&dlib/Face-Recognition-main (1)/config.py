"""
Configuration file for Face Recognition App
Adjust these parameters based on your needs and hardware
"""

import os

# ---------------------------------------------------------------------
# BASE_DIR anchors every path below to the folder this config.py file
# actually lives in - not the current working directory. Without this,
# "known_faces" only resolves correctly if you happen to launch python
# from this exact folder; running main.py from an IDE, a shortcut, or
# any other cwd would make these paths point to the wrong place even
# though the folders are sitting right here on disk.
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path settings
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
UNKNOWN_FACES_DIR = os.path.join(BASE_DIR, "unknown_faces")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
ENCODINGS_DIR = os.path.join(BASE_DIR, "encodings")
ENCODINGS_FILE = os.path.join(ENCODINGS_DIR, "encodings.pkl")

# Face recognition settings
# Tolerance: lower = stricter matching, higher = more permissive
# Range: 0.4-0.7, default 0.6
TOLERANCE = 0.6

# Model type for face detection
# "hog" = faster (10-30 FPS), suitable for CPU
# "cnn" = slower (1-5 FPS) but more accurate, requires GPU
MODEL = "cnn"

# Webcam settings
PROCESS_EVERY_N_FRAMES = 1  # Process every Nth frame for speed (1 = all frames, 5 = every 5th)
FRAME_WIDTH = 800  # Reduced resolution for real-time processing
FRAME_HEIGHT = 600

# Display settings
SHOW_FACE_LABELS = True
LABEL_FONT_SIZE = 0.6
LABEL_THICKNESS = 2
BOX_COLOR = (0, 255, 0)  # Green (BGR format)
TEXT_COLOR = (0, 255, 0)  # Green