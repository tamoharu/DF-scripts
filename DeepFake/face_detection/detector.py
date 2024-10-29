# input: image[1, 3, H, W]
# output: kps[5, 2]
# kps: [left_eye, right_eye, nose, left_mouth, right_mouth]

from typing import List, Tuple
import threading

import cv2
import numpy as np

import config.instances as instances
from DeepFake.utils.basemodel import OnnxBaseModel


class Box2Point:
    '''
    input
    face_boxes: [N, 4]
    eye_boxes: [N, 4]
    nose_boxes: [N, 4]
    mouth_boxes: [N, 4]

    output
    kps: [N, 5, 2]
    '''
    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)
    

    def run(self, face_boxes, eye_boxes, nose_boxes, mouth_boxes):
        results = []
        if len(face_boxes) == 0:
            return results
        # len(face_boxes) >= 1
        for face_box in face_boxes:
            eye_left_point, eye_right_point, nose_point, mouth_left_point, mouth_right_point = None, None, None, None, None
            # 0 <= len(eye_boxes) <= 2
            eye_boxes_filtered = self._filter_boxes(eye_boxes, face_box, max_items=2)
            # 0 <= len(nose_boxes) <= 1
            nose_boxes_filtered = self._filter_boxes(nose_boxes, face_box, max_items=1)
            # 0 <= len(mouth_boxes) <= 1
            mouth_boxes_filtered = self._filter_boxes(mouth_boxes, face_box, max_items=1)
            # bottom to top, vertical angle
            face_angle = self._calc_face_angle(face_box, eye_boxes_filtered, nose_boxes_filtered, mouth_boxes_filtered)
            nose_point = self._nose_post(nose_boxes_filtered, face_angle)
            mouth_left_point, mouth_right_point = self._mouth_post(mouth_boxes_filtered, face_angle)
            eye_left_point, eye_right_point = self._eye_post(face_box, eye_boxes_filtered, nose_boxes_filtered , mouth_boxes_filtered, face_angle)
            kps = np.zeros((5, 2))
            kps[0] = eye_left_point
            kps[1] = eye_right_point
            kps[2] = nose_point
            kps[3] = mouth_left_point
            kps[4] = mouth_right_point
            results.append(kps)
        return results
    
    
    def _filter_boxes(self, boxes, face_box, max_items):
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = face_box[0], face_box[1], face_box[2], face_box[3]
        inface_boxes = []
        for box in boxes:
            box_cx = (box[0] + box[2]) / 2
            box_cy = (box[1] + box[3]) / 2
            if x1 <= box_cx <= x2 and y1 <= box_cy <= y2:
                inface_boxes.append(box)
        if len(inface_boxes) > max_items:
            distances = []
            face_cx = (x1 + x2) / 2
            face_cy = (y1 + y2) / 2
            for box in inface_boxes:
                box_cx = (box[0] + box[2]) / 2
                box_cy = (box[1] + box[3]) / 2
                distance = (box_cx - face_cx) ** 2 + (box_cy - face_cy) ** 2
                distances.append((distance, box))
            distances.sort(key=lambda x: x[0]) 
            filtered_boxes = [box for _, box in distances[:max_items]]
        else:
            filtered_boxes = inface_boxes
        return filtered_boxes


    def _calc_face_angle(self, face_box, eye_boxes, nose_boxes, mouth_boxes):
        eyes_center = None
        nose_center = None
        mouth_center = None
        if len(eye_boxes) == 2:
            eye_1_center =  [(eye_boxes[0][0] + eye_boxes[0][2]) / 2, (eye_boxes[0][1] + eye_boxes[0][3]) / 2]
            eye_2_center =  [(eye_boxes[1][0] + eye_boxes[1][2]) / 2, (eye_boxes[1][1] + eye_boxes[1][3]) / 2]
            eyes_center = [(eye_1_center[0] + eye_2_center[0]) / 2, (eye_1_center[1] + eye_2_center[1]) / 2]
        if len(nose_boxes) == 1:
            nose_box = nose_boxes[0]
            nose_center = [(nose_box[0] + nose_box[2]) / 2, (nose_box[1] + nose_box[3]) / 2]
        if len(mouth_boxes) == 1:
            mouth_box = mouth_boxes[0]
            mouth_center = [(mouth_box[0] + mouth_box[2]) / 2, (mouth_box[1] + mouth_box[3]) / 2]
        if eyes_center is not None:
            if nose_center is not None:
                dx = eyes_center[0] - nose_center[0]
                dy = eyes_center[1] - nose_center[1]
                return np.arctan2(dy, dx)
            if mouth_center is not None:
                dx = eyes_center[0] - mouth_center[0]
                dy = eyes_center[1] - mouth_center[1]
                return np.arctan2(dy, dx)
            dx = eyes_center[0] - face_box[0]
            dy = eyes_center[1] - face_box[1]
            return np.arctan2(dy, dx)
        if mouth_center is not None:
            if nose_center is not None:
                dx = nose_center[0] - mouth_center[0]
                dy = nose_center[1] - mouth_center[1]
                return np.arctan2(dy, dx)
            dx = face_box[0] - mouth_center[0]
            dy = face_box[1] - mouth_center[1]
            return np.arctan2(dy, dx)
        return np.pi / 2

    
    def _eye_post(self, face_box, eye_boxes, nose_boxes_filtered , mouth_boxes_filtered, face_angle):
        if len(eye_boxes) == 0:
            return None, None
        eyes_centers = []
        for eye in eye_boxes:
            eye_center = [(eye[0] + eye[2]) / 2, (eye[1] + eye[3]) / 2]
            eyes_centers.append(eye_center)
        face_center = [(face_box[0] + face_box[2]) / 2, (face_box[1] + face_box[3]) / 2]
        if len(eye_boxes) == 1:
            eye_center = eyes_centers[0]
            if len(nose_boxes_filtered) == 1:
                dx = eye_center[0] - nose_boxes_filtered[0][0]
                dy = eye_center[1] - nose_boxes_filtered[0][1]
            elif len(mouth_boxes_filtered) == 1:
                dx = eye_center[0] - mouth_boxes_filtered[0][0]
                dy = eye_center[1] - mouth_boxes_filtered[0][1]
            else:
                dx = eye_center[0] - face_center[0]
                dy = eye_center[1] - face_center[1]
            eye_angle = np.arctan2(dy, dx)
            if eye_angle > face_angle:
                left_eye_point = eye_center
                right_eye_point = None
            else:
                left_eye_point = None
                right_eye_point = eye_center
            return left_eye_point, right_eye_point
        if len(eye_boxes) == 2:
            angles = []
            for eye_center in eyes_centers:
                if len(mouth_boxes_filtered) == 1:
                    dx = eye_center[0] - mouth_boxes_filtered[0][0]
                    dy = eye_center[1] - mouth_boxes_filtered[0][1]
                elif len(nose_boxes_filtered) == 1:
                    dx = eye_center[0] - nose_boxes_filtered[0][0]
                    dy = eye_center[1] - nose_boxes_filtered[0][1]
                else:
                    dx = eye_center[0] - face_center[0]
                    dy = eye_center[1] - face_center[1]
                angle = np.arctan2(dy, dx)
                angles.append(angle)
            if angles[0] < angles[1]:
                left_eye_point, right_eye_point = eyes_centers
            else:
                left_eye_point, right_eye_point = eyes_centers[::-1]
            return left_eye_point, right_eye_point
        raise ValueError('Invalid eye_boxes')
        

    def _nose_post(self, nose_boxes, face_angle):
        if len(nose_boxes) == 0:
            return None
        coef = 0.25
        nose_box = nose_boxes[0]
        dx = np.cos(face_angle) * (nose_box[2] - nose_box[0]) / 2
        dy = np.sin(face_angle) * (nose_box[3] - nose_box[1]) / 2
        nose_center = [(nose_box[0] + nose_box[2]) / 2, (nose_box[1] + nose_box[3]) / 2]
        nose_point = [nose_center[0] - dx * coef, nose_center[1] - dy * coef]
        return nose_point
    

    def _mouth_post(self, mouth_boxes, face_angle):
        if len(mouth_boxes) == 0:
            return None, None
        mouth_box = mouth_boxes[0]
        h_angle = face_angle - np.pi / 2
        mouth_center = [(mouth_box[0] + mouth_box[2]) / 2, (mouth_box[1] + mouth_box[3]) / 2]
        dx = np.cos(h_angle) * (mouth_box[2] - mouth_box[0]) / 2
        dy = np.sin(h_angle) * (mouth_box[3] - mouth_box[1]) / 2
        mouth_left_point = [mouth_center[0] - dx, mouth_center[1] - dy]
        mouth_right_point = [mouth_center[0] + dx, mouth_center[1] + dy]
        if h_angle < -np.pi / 2 or h_angle > np.pi / 2:
            mouth_left_point, mouth_right_point = mouth_right_point, mouth_left_point
        return mouth_left_point, mouth_right_point


class YOLOX(OnnxBaseModel):
    '''
    input
    input: [1, 3, 640, 640]

    output
    batchno_classid_score_x1y1x2y2: ['N', 7]
    '''
    _lock = threading.Lock()


    def __init__(self, model_path: str, execution_providers: List[str]):
        with YOLOX._lock:
            if instances.detector is None:
                super().__init__(model_path, execution_providers)
                self.model_size = (640, 640)
                instances.detector = self
            else:
                self.__dict__ = instances.detector.__dict__


    def run(self, frame):
        frame, resize_data = self._preprocess(frame)
        output = self._forward(frame)
        results = self._postprocess(output, resize_data)
        return results


    def _preprocess(self, frame):
        frame_height, frame_width = frame.shape[:2]
        resize_ratio = min(self.model_size[0] / frame_height, self.model_size[1] / frame_width)
        resized_shape = int(round(frame_width * resize_ratio)), int(round(frame_height * resize_ratio))
        frame = cv2.resize(frame, resized_shape, interpolation=cv2.INTER_LINEAR)
        offset_height = (self.model_size[0] - resized_shape[1]) / 2
        offset_width = (self.model_size[1] - resized_shape[0]) / 2
        resize_data = tuple([offset_height, offset_width, resize_ratio])
        frame = cv2.copyMakeBorder(frame, round(offset_height - 0.1), round(offset_height + 0.1), round(offset_width - 0.1), round(offset_width + 0.1), cv2.BORDER_CONSTANT, value = (114, 114, 114))
        frame = frame.transpose(2, 0, 1)
        frame = frame.astype(np.float32)
        frame = np.expand_dims(frame, axis = 0)
        frame = np.ascontiguousarray(frame)
        return frame, resize_data


    def _forward(self, frame):
        with self.semaphore:
            output = self.session.run(None,
            {
                self.input_names[0]: frame
            })
        return output


    def _postprocess(self, detections, resize_data):
        offset_height, offset_width, resize_ratio = resize_data
        detections = np.array(detections)
        detections[:, :, 3::2] = (detections[:, :, 3::2] - offset_width) / resize_ratio
        detections[:, :, 4::2] = (detections[:, :, 4::2] - offset_height) / resize_ratio
        face_boxes = detections[detections[:, :, 1] == 3]
        if len(face_boxes) == 0:
            return []
        eye_boxes = detections[detections[:, :, 1] == 4]
        nose_boxes = detections[detections[:, :, 1] == 5]
        mouth_boxes = detections[detections[:, :, 1] == 6]
        face_boxes = face_boxes[:, 3:]
        eye_boxes = eye_boxes[:, 3:]
        nose_boxes = nose_boxes[:, 3:]
        mouth_boxes = mouth_boxes[:, 3:]
        box2point = Box2Point()
        results = box2point(face_boxes, eye_boxes, nose_boxes, mouth_boxes)
        return results

    
class RTDETRv2(OnnxBaseModel):
    '''
    input
    input_bgr: [1, 3, 'H', 'W']

    output
    label_xyxy_score: [1, 1250, 6]
    '''
    _lock = threading.Lock()


    def __init__(self, model_path: str, execution_providers: List[str]):
        with RTDETRv2._lock:
            if instances.recognizer is None:
                super().__init__(model_path, execution_providers)
                instances.recognizer = self
            else:
                self.__dict__ = instances.recognizer.__dict__
    

    def run(self, frame):
        post_frame = frame.copy()
        frame = self._preprocess(frame)
        output = self._forward(frame)
        results = self._postprocess(output, post_frame)
        return results
    

    def _preprocess(self, frame):
        frame = frame.astype(np.float32)
        frame = frame.transpose(2, 0, 1)
        frame = np.expand_dims(frame, axis = 0)
        frame = np.ascontiguousarray(frame)
        return frame
    

    def _forward(self, frame):
        with self.semaphore:
            output = self.session.run(None,
            {
                self.input_names[0]: frame
            })
        return output
    

    def _postprocess(self, detections, frame):
        score_threshold = 0.5
        detections = np.array(detections).squeeze()
        frame_height, frame_width = frame.shape[:2]
        if len(detections) == 0:
            return []
        scores = detections[:, 5:6]
        keep_idxs = scores[:, 0] > score_threshold
        boxes_keep = detections[keep_idxs, :]
        if len(boxes_keep) == 0:
            return []
        boxes_keep = boxes_keep[:, :5]
        boxes_keep[:, 1::2] = boxes_keep[:, 1::2] * frame_width
        boxes_keep[:, 2::2] = boxes_keep[:, 2::2] * frame_height
        face_boxes = boxes_keep[boxes_keep[:, 0] == 16]
        if len(face_boxes) == 0:
            return []
        eye_boxes = boxes_keep[boxes_keep[:, 0] == 17]
        nose_boxes = boxes_keep[boxes_keep[:, 0] == 18]
        mouth_boxes = boxes_keep[boxes_keep[:, 0] == 19]
        face_boxes = face_boxes[:, 1:]
        eye_boxes = eye_boxes[:, 1:]
        nose_boxes = nose_boxes[:, 1:]
        mouth_boxes = mouth_boxes[:, 1:]
        box2point = Box2Point()
        results = box2point(face_boxes, eye_boxes, nose_boxes, mouth_boxes)
        return results