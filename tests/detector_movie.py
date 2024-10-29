import sys
sys.path.append('../')

import cv2
import copy
from tqdm import tqdm
import numpy as np
import torch
from DeepFake.face_detection.detector import RTDETRv2

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
    output_video_path = './output/output.mp4'
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
    with tqdm(total=total_frames, desc="Processing Video Frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = process_frame(frame, model)
            video_writer.write(processed_frame)
            pbar.update(1)
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print("Video processing completed!")

if __name__ == "__main__":
    main()
