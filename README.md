====================================
Fall Detection Using Pose Estimation
====================================

Project Overview
----------------

This project implements a lightweight vision-based fall detection system using human pose estimation. It uses CNN-LSTM architecture to detect fall/non-fall events in real-time. The system can run on:
   Raspberry Pi with Pi Camera for edge deployment.
   Laptop/PC 

 The pipeline includes:
   1.Video preprocessing & frame extraction.
   2.Pose estimation using MediaPipe Pose.
   3.Keyframe selection based on motion.
   4.Temporal interpolation of missing keypoints.
   5.Training a CNN-LSTM model on keypoints.
   6.Converting the trained model to TensorFlow Lite for edge deployment.
   7.Real-time inference on Raspberry Pi or webcam.

Dataset
-------

Dataset Used:

   UR Fall Detection Dataset (URFD): RGB videos containing falls and Activities of Daily Living (ADL).
   https://fenix.ur.edu.pl/~mkepski/ds/uf.html
   Each video includes a single person performing either a fall or normal activity.
   Dataset is divided into:
    1.Fall videos: Different types of falls (forward, backward, side, etc.).
    2.ADL videos: Non-fall activities (walking, sitting, bending, picking objects).

Data Requirements:

   Minimum 10-15 videos per class for small-scale training.
   More data improves accuracy and robustness.

Dataset is preprocessed into:
   Frames (PNG images)
   Pose keypoints (33 keypoints × 2 coordinates per frame)
   Keyframes (selected frames with maximum motion)

Installation
------------

   Clone Repository
     git clone https://github.com/Puneet08Sharma/fall-detection.git
     cd fall-detection

   Install Dependencies
     pip install -r requirements.txt

   Dependencies include:
     numpy, scipy → numerical computation
     opencv-python → video/frame processing
     mediapipe → pose estimation
     tensorflow → model training and inference
     tflite-runtime → lightweight inference on Raspberry Pi
     pandas → dataset handling
     matplotlib → visualization

Pipeline Details
----------------

  1.Video Preprocessing
    Extract all frames from videos.
    Use MediaPipe Pose to detect 33 keypoints per frame.
    Save frames and keypoints to dataset/.

  2.Keyframe Selection
    For each video, select NUM_KEYFRAMES (e.g., 10) with highest motion.
    Motion calculated based on midpoint of left & right shoulders.
    Helps reduce sequence length and speeds up training.
 
  3.Temporal Interpolation
    Fill missing keypoints (not detected in some frames) via linear interpolation.
    Ensures fixed-length sequences for LSTM input.

  4.Dataset Construction
    Each video converted into:
    Sequence of keypoints: (NUM_KEYFRAMES, 66) → 33 keypoints × (x, y)
    Label: 1 for fall, 0 for non-fall
    Exported to Excel (multimodal_dataset_summary.xlsx) for verification.

Model Training Architecture
---------------------------

  CNN-LSTM:
  Conv1D(128) → extracts spatial features
  LSTM(64) → learns temporal dependencies
  Dense(64) + Dense(1, Sigmoid) → binary classification

  Training:
  python scripts/train_model.py
  Input: (NUM_KEYFRAMES, 66) keypoints
  Output: Fall / Non-Fall
  Loss: Binary Crossentropy
  Optimizer: Adam
  Batch size: 8
  Epochs: 100

  Saving Model:
  model.save("models/fall_detection_model.h5")

  Model Conversion to TFLite:
  python scripts/h5_to_tflite.py
  Converts .h5 to .tflite for Raspberry Pi deployment.
  Supports lightweight real-time inference.
  Use converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS] if LSTM conversion fails.

  Inference:
  python scripts/inference_pi.py
  Captures live video from PiCam.
  Extracts keypoints per frame.
  Predicts fall/non-fall in real-time.
  Overlay results on video frame:
  Green text: Non-Fall
  Red text: Fall
  Supports adjustable confidence threshold.


  

