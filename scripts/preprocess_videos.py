import os
import cv2
import numpy as np
from glob import glob
import pandas as pd
from scipy.signal import savgol_filter
import mediapipe as mp
from tqdm import tqdm

# Parameters
NUM_KEYFRAMES = 10
FRAME_ROOT = "dataset"  # path to URFD RGB frames
SAVE_ROOT = "dataset_processed"

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=True)

# Utility functions
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_all_frames_from_directory(dir_path):
    frame_paths = sorted(glob(os.path.join(dir_path, '*.png')))
    return [cv2.imread(f) for f in frame_paths]

def extract_keypoints_mediapipe(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose_detector.process(image_rgb)
    keypoints = np.full((33, 2), -1.0)
    if result.pose_landmarks:
        for i, lm in enumerate(result.pose_landmarks.landmark):
            if i < 33:
                keypoints[i] = [lm.x, lm.y]
    return keypoints

def temporal_interpolate(keypoint_seq):
    keypoint_seq = np.array(keypoint_seq)
    for j in range(33):
        for k in range(2):
            kp_values = keypoint_seq[:, j, k]
            mask = kp_values != -1
            if np.sum(mask) >= 2:
                keypoint_seq[:, j, k] = np.interp(
                    np.arange(len(kp_values)),
                    np.where(mask)[0],
                    kp_values[mask]
                )
    return keypoint_seq

def select_keyframes(frames, num_frames=NUM_KEYFRAMES):
    shoulder_midpoints = []
    for frame in frames:
        keypoints = extract_keypoints_mediapipe(frame)
        left_shoulder, right_shoulder = keypoints[11], keypoints[12]
        if -1 in left_shoulder or -1 in right_shoulder:
            midpoint = [-1, -1]
        else:
            midpoint = [(left_shoulder[0]+right_shoulder[0])/2,
                        (left_shoulder[1]+right_shoulder[1])/2]
        shoulder_midpoints.append(midpoint)
    shoulder_midpoints = np.array(shoulder_midpoints)
    valid_indices = ~np.any(shoulder_midpoints == -1, axis=1)
    if valid_indices.sum() < num_frames:
        return frames[:num_frames], list(range(min(num_frames, len(frames))))
    smooth_x = savgol_filter(shoulder_midpoints[valid_indices,0], 11, 3)
    smooth_y = savgol_filter(shoulder_midpoints[valid_indices,1], 11, 3)
    diffs = np.sqrt(np.diff(smooth_x)**2 + np.diff(smooth_y)**2)
    key_indices = np.argsort(diffs)[-num_frames:]
    key_indices = sorted(key_indices)
    frame_indices = np.where(valid_indices)[0][key_indices]
    return [frames[i] for i in frame_indices], frame_indices.tolist()

def process_video(video_id, label, df_rows):
    frame_dir = os.path.join(FRAME_ROOT, video_id)
    frames = load_all_frames_from_directory(frame_dir)
    if len(frames) < NUM_KEYFRAMES:
        print(f"Skipping {video_id}, not enough frames")
        return df_rows
    selected_frames, indices = select_keyframes(frames)
    keypoint_seq = []
    save_dir = os.path.join(SAVE_ROOT, label, video_id)
    frame_save_dir = os.path.join(save_dir, "frames")
    keypoint_save_dir = os.path.join(save_dir, "keypoints")
    ensure_dir(frame_save_dir)
    ensure_dir(keypoint_save_dir)
    for i, frame in enumerate(selected_frames):
        frame_path = os.path.join(frame_save_dir, f"{i}.png")
        cv2.imwrite(frame_path, frame)
        keypoints = extract_keypoints_mediapipe(frame)
        keypoint_seq.append(keypoints)
    keypoint_seq = temporal_interpolate(keypoint_seq)
    for i, keypoints in enumerate(keypoint_seq):
        np.save(os.path.join(keypoint_save_dir, f"{i}.npy"), keypoints)
        row = {"video_id": video_id, "frame_index": i, "label": label}
        for j in range(33):
            row[f"x_{j}"], row[f"y_{j}"] = keypoints[j][0], keypoints[j][1]
        df_rows.append(row)
    return df_rows

if __name__ == "__main__":
    adl_folders = [f for f in os.listdir(FRAME_ROOT) if f.lower().startswith("adl")]
    fall_folders = [f for f in os.listdir(FRAME_ROOT) if f.lower().startswith("fall")]
    dataset_rows = []
    for video_id in tqdm(fall_folders, desc="Processing fall"):
        dataset_rows = process_video(video_id, "fall", dataset_rows)
    for video_id in tqdm(adl_folders, desc="Processing non_fall"):
        dataset_rows = process_video(video_id, "non_fall", dataset_rows)
    df = pd.DataFrame(dataset_rows)
    df.to_excel("multimodal_dataset_summary.xlsx", index=False)
    print("✅ Preprocessing complete")
