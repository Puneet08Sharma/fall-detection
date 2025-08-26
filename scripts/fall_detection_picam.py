import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

# ==== Parameters ====
NUM_KEYFRAMES = 10
KEYPOINT_DIM = 33*2
CONFIDENCE_THRESHOLD = 0.5

# ==== Load TFLite model ====
interpreter = tf.lite.Interpreter(model_path="fall_detection_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==== MediaPipe Pose ====
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# ==== Pi Camera Setup ====
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 15
rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(1)  # Allow camera to warm up

# ==== Sliding window buffer ====
keypoints_buffer = []

def extract_keypoints(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose_detector.process(frame_rgb)
    keypoints = np.full((33,2), -1.0)
    if result.pose_landmarks:
        for i, lm in enumerate(result.pose_landmarks.landmark):
            if i < 33:
                keypoints[i] = [lm.x, lm.y]
    return keypoints.flatten()

def temporal_interpolate(seq):
    seq = np.array(seq)
    for j in range(33):
        for k in range(2):
            vals = seq[:, j*2+k]
            mask = vals != -1
            if np.sum(mask) >= 2:
                seq[:, j*2+k] = np.interp(np.arange(len(vals)), np.where(mask)[0], vals[mask])
    return seq

# ==== Real-time loop ====
print("Starting Pi Camera fall detection...")
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array

    # Extract keypoints
    kp = extract_keypoints(image)
    keypoints_buffer.append(kp)

    if len(keypoints_buffer) > NUM_KEYFRAMES:
        keypoints_buffer.pop(0)

    if len(keypoints_buffer) == NUM_KEYFRAMES:
        window = temporal_interpolate(keypoints_buffer)
        window_input = np.expand_dims(window, axis=0).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], window_input)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])[0][0]

        label = "Fall" if pred >= CONFIDENCE_THRESHOLD else "Non-Fall"
        color = (0,0,255) if pred >= CONFIDENCE_THRESHOLD else (0,255,0)
        cv2.putText(image, f"{label}: {pred:.3f}", (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show real-time image
    cv2.imshow("Fall Detection PiCam", image)
    rawCapture.truncate(0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.close()
cv2.destroyAllWindows()
