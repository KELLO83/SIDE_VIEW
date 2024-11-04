import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm.auto import  tqdm
import os
import logging
import pandas as pd
import functools
import time
import fisheye2plane
import torch
from natsort import natsorted
from glob import glob
from typing import Union
from collections import Counter
import multiprocessing as mp
import traceback
from sit_recognition import SitRecognition
from typing import Dict, List, Tuple, Optional, Union
from collections import deque
import math
class YoloPoseEstimator:
    def __init__(self):
        self.pose_estimator = YOLO('yolo11x-pose.pt')
        # 하체 키포인트 인덱스 정의
        self.lower_body_keypoints = {
            0: 'nose',          # 코
            1: 'neck',          # 목
            2 : 'right_eye',    # 오른쪽 눈
            3 : 'left_eye',     # 왼쪽 눈
            5: 'left_shoulder', # 왼쪽 어깨
            6: 'right_shoulder',# 오른쪽 어깨
            11: 'left_hip',     # 왼쪽 엉덩이
            12: 'right_hip',    # 오른쪽 엉덩이
            13: 'left_knee',    # 왼쪽 무릎
            14: 'right_knee',   # 오른쪽 무릎
            15: 'left_ankle',   # 왼쪽 발목
            16: 'right_ankle',  # 오른쪽 발목
        }
        
        self.naming_keypoints = {  
            0: 'nos',           # 코
            1: 'nec',           # 목
            2: 'r_eye',         # 오른쪽 눈
            3: 'l_eye',         # 왼쪽 눈
            5: 'l_sho',         # 쪽 어깨
            6: 'r_sho',         # 오른쪽 어깨
            11: 'l_hip',        # 왼쪽 엉덩이
            12: 'r_hip',        # 오른쪽 엉덩이
            13: 'l_kne',        # 왼쪽 무릎
            14: 'r_kne',        # 오른쪽 무릎
            15: 'l_ank',        # 왼쪽 발목
            16: 'r_ank',        # 오른쪽 발목
        }
        
        self.KEYPOINT_INDEX = {v: k for k, v in self.lower_body_keypoints.items()}
        
        # 연결선 정의를 클래스 속성으로 이동
        self.connections = [
            (13, 15),  # 왼쪽 무릎-왼쪽 발목
            (14, 16),  # 오른쪽 무릎-오른쪽 발목
            (11, 13),  # 왼쪽 엉덩이-왼쪽 무릎
            (12, 14),  # 오른쪽 엉덩이-오른쪽 무릎
        ]
        
        # 상수 정의
        self.CONFIDENCE_THRESHOLD = 0.4
        self.LINE_COLOR = (0, 255, 0)    # 초록색
        self.POINT_COLOR = (255, 0, 0)   # 파란색
        self.TEXT_COLOR = (0, 0, 255)    # 빨간색
    def draw_skeleton(self, img, person_keypoints):
        """키포인트 간의 스켈레톤 그리기"""
        for start_idx, end_idx in self.connections:
            # is_valid_keypoints 함수로 한 번만 신뢰도 체크
            if self.is_valid_keypoints(person_keypoints, [start_idx, end_idx]):
                start_pos = tuple(map(int, person_keypoints[start_idx][:2]))
                end_pos = tuple(map(int, person_keypoints[end_idx][:2]))
                
                # 선 그리기
                cv2.line(img, start_pos, end_pos, self.LINE_COLOR, 2)
                
                # 선분의 중앙점 계산
                mid_x = int((start_pos[0] + end_pos[0]) / 2)
                mid_y = int((start_pos[1] + end_pos[1]) / 2)
                
                # 신뢰도 점수 가져오기
                start_conf = float(person_keypoints[start_idx][2])
                end_conf = float(person_keypoints[end_idx][2])
                
                # 신뢰도 텍스트 표시 (소수점 2자리까지)
                conf_text = f"{start_conf:.2f}/{end_conf:.2f}"
                cv2.putText(img, conf_text, (mid_x, mid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           self.TEXT_COLOR, 1)

    def draw_keypoints(self, img, person_keypoints):
        """키포인트 circle 표시 """
        for idx, label in self.naming_keypoints.items():
            kp = person_keypoints[idx]
            if kp[2] > self.CONFIDENCE_THRESHOLD:
                x, y = map(int, kp[:2])
                cv2.circle(img, (x, y), 4, self.POINT_COLOR, -1)
                cv2.putText(img, label, (x + 5, y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                          self.TEXT_COLOR, 1)
                
    
    def is_valid_keypoints(self, keypoints, indices):
        """주어진 키포인트들이 모두 신뢰도 임계값을 넘는지 확인"""
        for x , y , conf in keypoints[indices]:
            if conf < self.CONFIDENCE_THRESHOLD:
                return False
        return True
    
    def calculate_midpoint(self, keypoints, idx1, idx2):
        """두 키포인트의 중점 계산"""
        if self.is_valid_keypoints(keypoints, [idx1, idx2]):
            return (keypoints[idx1][:2] + keypoints[idx2][:2]) / 2
        return None
    
    def calculate_direction_angle(self, person_keypoints):
        """
        'LEFT HIP' 'RIGHT HIP' 'LEFT KNEE' 'RIGHT KNEE' 를 이용하여 사람이 바라보고 있는 방향 각도를 계산합니다.
        Returns:
            tuple: (angle, used_keypoints)
            angle: float - 수학적 각도 (-180 ~ 180도)
            used_keypoints: dict - 사용된 키포인트 정보
        """
        import math

        used_keypoints = {'method': None}  # 어떤 방법으로 계산되었는지 저장

        # 키포인트 추출
        left_hip = person_keypoints[self.KEYPOINT_INDEX.get('left_hip')].cpu().numpy()
        right_hip = person_keypoints[self.KEYPOINT_INDEX.get('right_hip')].cpu().numpy()
        left_knee = person_keypoints[self.KEYPOINT_INDEX.get('left_knee')].cpu().numpy()
        right_knee = person_keypoints[self.KEYPOINT_INDEX.get('right_knee')].cpu().numpy()
        left_ankle = person_keypoints[self.KEYPOINT_INDEX.get('left_ankle')].cpu().numpy()
        right_ankle = person_keypoints[self.KEYPOINT_INDEX.get('right_ankle')].cpu().numpy()

        # 신뢰도 체크
        if (all(hip[2] > self.CONFIDENCE_THRESHOLD and all(int(coord) > 0 for coord in hip[:2]) for hip in [left_hip, right_hip]) and
            all(knee[2] > self.CONFIDENCE_THRESHOLD and all(int(coord) > 0 for coord in knee[:2]) for knee in [left_knee, right_knee])):

            used_keypoints['method'] = 'hip_knee'
            used_keypoints.update({
                'left_hip': list(map(float, left_hip)),
                'right_hip': list(map(float, right_hip)),
                'left_knee': list(map(float, left_knee)),
                'right_knee': list(map(float, right_knee))
            })

            # 엉덩이 중점 계산
            hip_midpoint = [
                (left_hip[0] + right_hip[0]) / 2,
                (left_hip[1] + right_hip[1]) / 2
            ]

            # 무릎 중점 계산
            knee_midpoint = [
                (left_knee[0] + right_knee[0]) / 2,
                (left_knee[1] + right_knee[1]) / 2
            ]

            # 방향 벡터 계산 (엉덩이에서 무릎으로 향하는 벡터)
            dx = knee_midpoint[0] - hip_midpoint[0]
            dy = knee_midpoint[1] - hip_midpoint[1]

            # 각도 계산 (수평선 기준, 오른쪽이 0도)
            angle = math.degrees(math.atan2(dy, dx))
            
            # 각도를 -180 ~ 180 범위로 조정
            angle = angle % 360
            if angle > 180:
                angle -= 360
                
            return angle, used_keypoints
        
        elif (all(knee[2] > self.CONFIDENCE_THRESHOLD and all(int(coord) > 0 for coord in knee[:2]) for knee in [left_knee, right_knee]) and
              all(ankle[2] > self.CONFIDENCE_THRESHOLD and all(int(coord) > 0 for coord in ankle[:2]) for ankle in [left_ankle, right_ankle])):
                
            used_keypoints['method'] = 'knee_ankle'
            used_keypoints.update({
                'left_knee': list(map(float, left_knee)),
                'right_knee': list(map(float, right_knee)),
                'left_ankle': list(map(float, left_ankle)),
                'right_ankle': list(map(float, right_ankle))
            })
            
            knee_center = [
                (left_knee[0] + right_knee[0]) / 2,
                (left_knee[1] + right_knee[1]) / 2
            ]
            ankle_center = [
                (left_ankle[0] + right_ankle[0]) / 2,
                (left_ankle[1] + right_ankle[1]) / 2
            ]

            # 무릎에서 발목으로 가는 방향 벡터 계산
            direction_vector = (ankle_center[0] - knee_center[0], ankle_center[1] - knee_center[1])

            # 방향 각도 계산 (라디안 -> 도 단위 변환)
            angle = np.arctan2(direction_vector[1], direction_vector[0]) * (180 / np.pi)
            angle = angle % 360
            if angle > 180:
                angle -= 360
                
            return angle, used_keypoints
            
        else:
            return None, {'method': None}
        
    def calculate_head_direction_angle(self, person_keypoints):
        """ 머리 방향 각도 계산 """
        nose = person_keypoints[0].cpu().numpy()
        left_eye = person_keypoints[3].cpu().numpy()  # 왼쪽 눈 인덱스 3
        right_eye = person_keypoints[2].cpu().numpy() # 오른쪽 눈 인덱스 2
        
        # 코와 양쪽 눈의 신뢰도와 좌표 유효성 검사
        if (nose[2] > self.CONFIDENCE_THRESHOLD and 
            left_eye[2] > self.CONFIDENCE_THRESHOLD and 
            right_eye[2] > self.CONFIDENCE_THRESHOLD and
            all(int(coord) > 0 for coord in nose[:2]) and
            all(int(coord) > 0 for coord in left_eye[:2]) and
            all(int(coord) > 0 for coord in right_eye[:2])):
            
            # 양쪽 눈의 중점 계산
            eyes_center = np.array([
                (left_eye[0] + right_eye[0]) / 2,
                (left_eye[1] + right_eye[1]) / 2
            ])
            
            # 눈 중점에서 코까지의 방향 벡터 계산
            head_vector = nose[:2] - eyes_center
            
            # 방향 각도 계산 (라디안 -> 도)
            head_angle = math.degrees(math.atan2(head_vector[1], head_vector[0]))
            
            # 각도를 -180 ~ 180 범위로 조정
            head_angle = head_angle % 360
            if head_angle > 180:
                head_angle -= 360
            
            return head_angle
        else:
            return None
    
    def calculate_upper_body_direction(self, person_keypoints):
        """ 상체 방향 각도 계산 """
       
        left_hip = person_keypoints[11].cpu().numpy()
        right_hip = person_keypoints[12].cpu().numpy()
        left_shoulder = person_keypoints[5].cpu().numpy()
        right_shoulder = person_keypoints[6].cpu().numpy()
        
        if left_hip[2] > self.CONFIDENCE_THRESHOLD and right_hip[2] > self.CONFIDENCE_THRESHOLD and \
           left_shoulder[2] > self.CONFIDENCE_THRESHOLD and right_shoulder[2] > self.CONFIDENCE_THRESHOLD and \
           all(int(coord) > 0 for coord in left_hip[:2]) and all(int(coord) > 0 for coord in right_hip[:2]) and \
           all(int(coord) > 0 for coord in left_shoulder[:2]) and all(int(coord) > 0 for coord in right_shoulder[:2]):
            
            # 어깨 중심점과 엉덩이 중심점 계산
            shoulder_center = (left_shoulder[:2] + right_shoulder[:2]) / 2
            hip_center = (left_hip[:2] + right_hip[:2]) / 2
            
            # 상체 방향 벡터 계산 (엉덩이 중심에서 어깨 중심으로)
            upper_body_vector = shoulder_center - hip_center
            
            # 방향 각도 계산 (라디안 -> 도)
            upper_body_angle = math.degrees(math.atan2(upper_body_vector[1], upper_body_vector[0]))
        else:
            upper_body_angle = None
        
        return upper_body_angle

    def judge_pose_by_angles(self, person_keypoints):
        LEFT_HIP = 11
        LEFT_KNEE = 13
        LEFT_ANKLE = 15

        # 키포인트 추출 및 신뢰도 체크
        left_hip = person_keypoints[LEFT_HIP].cpu().numpy()
        left_knee = person_keypoints[LEFT_KNEE].cpu().numpy()
        left_ankle = person_keypoints[LEFT_ANKLE].cpu().numpy()
        
        if left_hip[2] > self.CONFIDENCE_THRESHOLD and left_knee[2] > self.CONFIDENCE_THRESHOLD and \
           left_ankle[2] > self.CONFIDENCE_THRESHOLD and all(int(coord) > 0 for coord in left_hip[:2]) and \
           all(int(coord) > 0 for coord in left_knee[:2]) and all(int(coord) > 0 for coord in left_ankle[:2]):
            
            # 엉덩이에서 무릎으로의 벡터
            hip_to_knee = left_knee[:2] - left_hip[:2]
            # 무릎에서 발목으로의 벡터  
            knee_to_ankle = left_ankle[:2] - left_knee[:2]
            
            # 각 벡터의 각도 계산
            hip_knee_angle = math.degrees(math.atan2(hip_to_knee[1], hip_to_knee[0]))
            knee_ankle_angle = math.degrees(math.atan2(knee_to_ankle[1], knee_to_ankle[0]))
            
            # 두 각도의 차이 계산
            angle_diff = hip_knee_angle - knee_ankle_angle

            # 각도를 -180 ~ 180 범위로 조정
            angle_diff = angle_diff % 360
            if angle_diff > 180:
                angle_diff -= 360
            return angle_diff
        
        else:
            return None
        
    def draw_angle_arrow(self, img, start_point, angle, length=50, color=(0, 165, 255), thickness=2):
        """
        각도를 나타내는 화살표를 그립니다.
        Args:
            angle: -180 ~ 180 범위의 각도
        """
        if angle is None:
            return img
        
        # 각도를 라디안으로 변환
        angle_rad = math.radians(angle)
        
        # 끝점 계산 (y축이 아래로 증가하므로 sin에 -를 곱함)
        end_x = int(start_point[0] + length * math.cos(angle_rad))
        end_y = int(start_point[1] + length * math.sin(angle_rad))
        end_point = (end_x, end_y)
        
        # 화살표 그리기
        cv2.putText(img, f'{int(angle)}', (start_point[0], start_point[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.arrowedLine(img, start_point, end_point, color, thickness, tipLength=0.3)
        
        return img
    
    def PoseEstimation(self, img):
        try:
            results = self.pose_estimator(img,
                                        half=False,
                                        iou=0.5,
                                        conf=self.CONFIDENCE_THRESHOLD,
                                        device='cuda:0',
                                        classes=[0])
            
            output_img = img.copy()

            for result in results:
                if not hasattr(result, 'keypoints') or result.keypoints is None:
                    continue
                    
                keypoints = result.keypoints.data
                xyxys = result.boxes.xyxy.data
                for index, (person_keypoints, xyxy) in enumerate(zip(keypoints, xyxys)):
                    x1, y1, x2, y2 = list(map(int, xyxy.cpu().numpy()))
                    center_x, center_y = int((x1+x2)//2), int((y1+y2)//2)
                    cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(output_img, f'{index}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    # 각도 계산
                    head_angle = self.calculate_head_direction_angle(person_keypoints)
                    upper_body_angle = self.calculate_upper_body_direction(person_keypoints)

                    # y 좌표 오프셋 초기화
                    y_offset = y1 - 20
                    
                    # 각 각도가 유효할 때만 텍스트 표시
                    if head_angle is not None:
                        cv2.putText(output_img,
                                  f'Head Angle: {int(head_angle)}',
                                  (x1, y_offset),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5,
                                  (0,0,255),
                                  2)
                        y_offset -= 20
                    
                    if upper_body_angle is not None:
                        cv2.putText(output_img,
                                  f'Upper Body: {int(upper_body_angle)}',
                                  (x1, y_offset),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5,
                                  (0,0,255),
                                  2)
                        y_offset -= 20
                    

                    # angle = self.judge_pose_by_angles(person_keypoints) if self.judge_pose_by_angles(person_keypoints) is not None else None
                    # output_img = self.draw_angle_arrow(output_img, (center_x, center_y), angle, color=(128, 0, 128))
                    angle, used_keypoints = self.calculate_direction_angle(person_keypoints)
                    if angle is not None:
                        # 텍스트 표시
                        print(f"index: {index} ,angle: {angle}, used_keypoints: {used_keypoints}")
                        
                        cv2.putText(output_img,
                                f'Lower Body: {int(angle)} ({used_keypoints["method"]})',
                                (x1, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0,0,255),
                                2)
                        # 화살표 그리기
                        output_img = self.draw_angle_arrow(output_img, (center_x, center_y), angle)
                    else:
                        angle = upper_body_angle if upper_body_angle is not None else None
                        if angle is not None:
                            output_img = self.draw_angle_arrow(output_img, (center_x, center_y), angle, color=(255, 192, 203))
                            
                    self.draw_skeleton(output_img, person_keypoints)
                    self.draw_keypoints(output_img, person_keypoints)
            return output_img
            
        except Exception as e:
            logging.error(f"Pose estimation error: {str(e)}")
            return img
        
if __name__ == "__main__":
    yolo_pose_estimator = YoloPoseEstimator()
    img = cv2.imread('tracking_images/000003_04.jpg')
    output_img = yolo_pose_estimator.PoseEstimation(img)
    cv2.namedWindow('output_img', cv2.WINDOW_NORMAL)
    cv2.imshow('output_img', output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()