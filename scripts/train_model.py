import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

NUM_KEYFRAMES = 10
KEYPOINT_DIM = 33*2
DATA_ROOT = "dataset_processed"

def load_dataset(data_root):
    X, y = [], []
    for label_name in ['fall', 'non_fall']:
        label = 1 if label_name == 'fall' else 0
        label_dir = os.path.join(data_root, label_name)
        for video_id in os.listdir(label_dir):
            kp_dir = os.path.join(label_dir, video_id, 'keypoints')
            keypoint_seq = []
            for i in range(NUM_KEYFRAMES):
                npy_path = os.path.join(kp_dir, f"{i}.npy")
                if not os.path.exists(npy_path):
                    break
                keypoints = np.load(npy_path).flatten()
                keypoint_seq.append(keypoints)
            if len(keypoint_seq) == NUM_KEYFRAMES:
                X.append(keypoint_seq)
                y.append(label)
    return np.array(X), np.array(y)

X, y = load_dataset(DATA_ROOT)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

def build_model(input_shape):
    model = models.Sequential([
        layers.Conv1D(128, 3, padding='same', activation='relu', input_shape=input_shape),
        layers.LSTM(64),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model((NUM_KEYFRAMES, KEYPOINT_DIM))
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=8
)

model.save("models/fall_detection_model.h5")
print("✅ Model saved as .h5")
