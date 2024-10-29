import os
import sys
sys.path.append('..')
import torch
import cv2
import numpy as np
from DeepFake.face_detection.detector import YOLOX, RTDETRv2


image_path = './images/test_bella.png'
execution_providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
basename = os.path.basename(image_path)


def yolox():
    model_path = '../models/yolox.onnx'
    detector = YOLOX(model_path=model_path, execution_providers=execution_providers)
    run(detector)


def rtdetrv2():
    model_path = '../models/rtdetrv2_large.onnx'
    detector = RTDETRv2(model_path=model_path, execution_providers=execution_providers)
    run(detector)


def run(detector):
    frame = cv2.imread(image_path)
    output = detector(frame)
    for kps in output:
        for i, kp in enumerate(kps):
            x, y = kp
            if np.isnan(x) or np.isnan(y):
                continue
            if i == 0:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                cv2.putText(frame, 'L', (int(x) + 5, int(y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            elif i == 1:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                cv2.putText(frame, 'R', (int(x) + 5, int(y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            elif i == 2:
                cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)
            elif i == 3:
                cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 0), -1)
                cv2.putText(frame, 'L', (int(x) + 5, int(y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
            elif i == 4:
                cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 0), -1)
                cv2.putText(frame, 'R', (int(x) + 5, int(y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
    cv2.imwrite(f'./outputs/{basename}_detected.png', frame)


if __name__ == '__main__':
    rtdetrv2()