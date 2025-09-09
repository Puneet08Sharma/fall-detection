import cv2
import numpy as np
import time
import os
from tensorflow.lite.python.interpreter import Interpreter
from picamera2 import Picamera2

# ==== Parameters ====
NUM_KEYFRAMES = 10
KEYPOINT_DIM = 33 * 2
CONFIDENCE_THRESHOLD = 0.5
SMOOTHING_SIZE = 5
CAM_RESOLUTION = (320, 240)
FPS = 10

# ==== Load Fall Detection LSTM Model (TFLite) ====
fall_interpreter = Interpreter(model_path="models/fall_detection_model.tflite")
fall_interpreter.allocate_tensors()
fall_in = fall_interpreter.get_input_details()
fall_out = fall_interpreter.get_output_details()

# ==== Load Pose Landmark Model (TFLite) ====
lm_interpreter = Interpreter(model_path="models/pose_landmark_lite.tflite")
lm_interpreter.allocate_tensors()
lm_in = lm_interpreter.get_input_details()
lm_out = lm_interpreter.get_output_details()

# ==== Buffers ====
keypoints_buffer = []
smoothing_window = []

# ==== Keypoint extraction + smoothing ====
def extract_keypoints_pose_landmark(frame, lm_interpreter, lm_input_details, lm_output_details, smoothing_window, SMOOTHING_SIZE=5):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    lm_input = cv2.resize(img_rgb, (256, 256))
    lm_input = np.expand_dims(lm_input, axis=0).astype(np.float32)/255.0

    lm_interpreter.set_tensor(lm_input_details[0]['index'], lm_input)
    lm_interpreter.invoke()
    lm_out_data = lm_interpreter.get_tensor(lm_output_details[0]['index'])[0]

    keypoints = []
    for i in range(33):
        x = lm_out_data[i*5]
        y = lm_out_data[i*5 + 1]
        keypoints.append([x, y])
    keypoints = np.array(keypoints, dtype=np.float32)

    # smoothing
    smoothing_window.append(keypoints)
    if len(smoothing_window) > SMOOTHING_SIZE:
        smoothing_window.pop(0)
    smoothed = np.mean(smoothing_window, axis=0)

    return smoothed.flatten()

# ==== Camera Setup ====
cam = Picamera2()
config = cam.create_preview_configuration(main={"size": CAM_RESOLUTION})
cam.configure(config)
cam.start()

print("✅ Starting Fall Detection... Press 'r' to start/stop recording, 'q' to quit")

# ==== Recording state ====
recording = False
out = None

while True:
    frame = cam.capture_array()

    # ==== Extract keypoints (pass all required args) ====
    kp = extract_keypoints_pose_landmark(
        frame,
        lm_interpreter=lm_interpreter,
        lm_input_details=lm_in,
        lm_output_details=lm_out,
        smoothing_window=smoothing_window,
        SMOOTHING_SIZE=SMOOTHING_SIZE
    )

    keypoints_buffer.append(kp)
    if len(keypoints_buffer) > NUM_KEYFRAMES:
        keypoints_buffer.pop(0)

    # ==== Fall detection ====
    if len(keypoints_buffer) == NUM_KEYFRAMES:
        window = np.array(keypoints_buffer)
        window_input = np.expand_dims(window, axis=0).astype(np.float32)

        fall_interpreter.set_tensor(fall_in[0]['index'], window_input)
        fall_interpreter.invoke()
        pred = fall_interpreter.get_tensor(fall_out[0]['index'])[0][0]

        label = "Fall" if pred >= CONFIDENCE_THRESHOLD else "Non-Fall"
        color = (0, 0, 255) if pred >= CONFIDENCE_THRESHOLD else (0, 255, 0)

        cv2.putText(frame, f"{label}: {pred:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        print(f"[Prediction] {label} | Confidence: {pred:.3f}")

    # ==== Write frame if recording ====
    if recording and out is not None:
        out.write(frame)

    # ==== Show window ====
    cv2.imshow("Fall Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        if not recording:
            print("▶️ Start Recording...")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('fall_detection_output.avi', fourcc, FPS, CAM_RESOLUTION)
            recording = True
        else:
            print("⏹ Stop Recording...")
            recording = False
            if out is not None:
                out.release()
                out = None

    if key == ord('q'):
        break

# ==== Cleanup ====
cam.stop()
if out is not None:
    out.release()
cv2.destroyAllWindows()
