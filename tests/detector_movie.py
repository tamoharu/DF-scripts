import sys
sys.path.append('../')

import cv2
import copy
from tqdm import tqdm
import numpy as np
import torch
from DeepFake.face_detection.detector import RTDETRv2
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_frame(frame, model):
    debug_image = copy.deepcopy(frame)
    output = model(debug_image)
    if not output:
        return debug_image
    for kps in output:
        for kp in kps:
            cx, cy = kp[0], kp[1]
            if not (np.isnan(cx) or np.isnan(cy)):
                color = (0, 255, 0)
                cv2.circle(debug_image, (int(cx), int(cy)), 5, color, -1)

    return debug_image

def main():
    model_file = '../models/rtdetrv2_large.onnx'
    video_path = './videos/bella.mp4'
    output_video_path = './outputs/output.mp4'
    providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    model = RTDETRv2(model_file, providers)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    with tqdm(total=total_frames, desc="Reading Video Frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            pbar.update(1)
    cap.release()

    processed_frames = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_frame = {executor.submit(process_frame, frame, model): i for i, frame in enumerate(frames)}
        with tqdm(total=total_frames, desc="Processing Video Frames") as pbar:
            for future in as_completed(future_to_frame):
                frame_index = future_to_frame[future]
                try:
                    processed_frame = future.result()
                    processed_frames.append((frame_index, processed_frame))
                except Exception as exc:
                    print(f"Frame {frame_index} generated an exception: {exc}")
                pbar.update(1)

    processed_frames.sort(key=lambda x: x[0])
    with tqdm(total=total_frames, desc="Writing Video Frames") as pbar:
        for _, frame in processed_frames:
            video_writer.write(frame)
            pbar.update(1)

    video_writer.release()
    cv2.destroyAllWindows()
    print("Video processing completed!")

if __name__ == "__main__":
    main()
