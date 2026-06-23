"""
Script for real-time face recognition from webcam (OpenCV DNN version)
Captures video from default webcam and recognizes faces in real-time

Usage:
    python recognize_webcam.py

Controls:
    ESC or 'q' - Exit
    Space - Toggle pause/play
"""

import cv2
import time
from imutils.video import VideoStream
from face_utils import (
    load_encoded_faces,
    recognize_faces_in_image,
    draw_face_boxes
)
import config


def recognize_webcam():
    """
    Start real-time face recognition from webcam
    """
    print("Initializing webcam...")
    
    # Load known faces database
    known_encodings, known_names = load_encoded_faces()
    if len(known_encodings) == 0:
        print("Error: No known faces in database. Run encode_faces.py first.")
        return False
    
    print(f"Database loaded: {len(set(known_names))} known people")
    print("Starting video stream...")
    
    # Start video stream
    try:
        vs = VideoStream(src=0).start()
        time.sleep(2.0)  # Allow camera to warm up
    except Exception as e:
        print(f"Error: Could not access webcam: {e}")
        print("Trying alternative method...")
        try:
            vs = cv2.VideoCapture(0)
            if not vs.isOpened():
                print("Could not open webcam with alternative method")
                return False
        except:
            return False
    
    frame_count = 0
    paused = False
    start_time = time.time()
    fps = 0
    
    print("\nVideo stream started. Press ESC or 'q' to exit, Space to pause/play")
    print("-" * 60)
    
    try:
        while True:
            # Read frame from stream
            frame = vs.read()

            if frame is None:
                print("Error: Failed to read frame from webcam")
                break
            
            if frame is None:
                print("Error: Failed to read frame from webcam")
                break
            
            frame_count += 1
            
            # Process every Nth frame for performance
            if frame_count % config.PROCESS_EVERY_N_FRAMES == 0 and not paused:
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
                
                # Recognize faces
                try:
                    face_locations, face_labels, face_distances = recognize_faces_in_image(
                        small_frame, known_encodings, known_names
                    )
                    
                    # Scale coordinates back to original frame size
                    scale_y = frame.shape[0] / config.FRAME_HEIGHT
                    scale_x = frame.shape[1] / config.FRAME_WIDTH
                    
                    scaled_locations = []
                    for (x, y, w, h) in face_locations:
                        scaled_locations.append((
                            int(x * scale_x),
                            int(y * scale_y),
                            int(w * scale_x),
                            int(h * scale_y)
                        ))
                    
                    frame = draw_face_boxes(frame, scaled_locations, face_labels)
                
                except Exception as e:
                    print(f"Error during face recognition: {e}")
            
            # Calculate FPS
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = frame_count / elapsed
            
            # Add info text
            status_text = "PAUSED" if paused else f"FPS: {fps:.1f}"
            cv2.putText(frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "ESC/q: exit, Space: pause",
                       (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Display frame
            cv2.imshow("Face Recognition - Webcam", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q'
                print("\nExiting...")
                break
            elif key == ord(' '):  # Space to pause
                paused = not paused
                status = "PAUSED" if paused else "RESUMED"
                print(f"Playback {status}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during video capture: {e}")
    
    finally:
        print("Cleaning up...")
        cv2.destroyAllWindows()

        try:
            vs.stop()
        except:
            pass

        try:
            vs.release()
        except:
            pass

        print("Done!")


def main():
    """Main entry point"""
    recognize_webcam()


if __name__ == "__main__":
    main()
