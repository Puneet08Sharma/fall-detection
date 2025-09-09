# fall_detection_realtime.py
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# === Parameters ===
VIDEO_PATH = "subject 2_fall_19.mp4"        # Input video
OUTPUT_PATH = "output_fall_detection.avi" # Optional: save output
MODEL_PATH = "fall_detection_model.h5"
NUM_KEYFRAMES = 10
KEYPOINT_DIM = 33*2

# === Load model ===
model = load_model(MODEL_PATH)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# === Setup MediaPipe ===
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=True)

# === Functions ===
def extract_keypoints(frame):
    """Extract 33 pose keypoints as flattened array, fallback to zeros."""
    results = pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        kps = np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark])
        return kps.flatten()
    else:
        return np.zeros(KEYPOINT_DIM)

# === Video Capture ===
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Optional: setup video writer
save_output = True
if save_output:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# === Sliding window buffers ===
window = []
original_frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    kp = extract_keypoints(frame)
    window.append(kp)
    original_frames.append(frame.copy())

    if len(window) == NUM_KEYFRAMES:
        # Predict
        seq_input = np.expand_dims(np.array(window), axis=0)
        pred = model.predict(seq_input)[0][0]
        label = "Fall" if pred >= 0.3 else "Non-Fall"

        # Overlay label on last frame
        overlay_frame = original_frames[-1].copy()
        color = (0,0,255) if label=="Fall" else (0,255,0)
        cv2.putText(overlay_frame, f"{label} ({pred:.2f})",
                    (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Display
        cv2.imshow("Fall Detection", overlay_frame)
        if save_output:
            out.write(overlay_frame)

        # Slide window
        window.pop(0)
        original_frames.pop(0)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if save_output:
    out.release()

print("✅ Inference finished, output saved to:", OUTPUT_PATH if save_output else "Not saved")
