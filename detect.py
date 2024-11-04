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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

logging.info("Yolo Detect run ...")

def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print("Model : ",args[0].model.model_name)
        print(f"Function '{func.__name__}' started at {time.strftime('%H:%M:%S', time.localtime(start_time))}")
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        print(f"Function '{func.__name__}' ended at {time.strftime('%H:%M:%S', time.localtime(end_time))}")
        print(f"Total execution time: {end_time - start_time:.4f} seconds")

        return result
    return wrapper

class YoloDetector:
    def __init__(self, SCEN='1'):
        self.model = YOLO('yolo11x.pt')
        self.LOCATION = Location()
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.SCEN = SCEN
        self.set_list = self._load_and_pair_images()
        self.seat_detector = SeatPositionDetector()
        self.visualizer = SitRecognition()
        
        self.pose_estimator = YoloPoseEstimator()
        self.segmentation = YoloSegmentation()
        self.tracking = YoloTracking()
        
        self.detected_person_cordinate = []
    

    def _load_and_pair_images(self):
        """이미지 로딩 및 페어링"""
        try:
            cam0_list = natsorted(glob(f'collect/scne{self.SCEN}_0.5origin/cam0/*.jpg'))
            cam2_list = natsorted(glob(f'collect/scne{self.SCEN}_0.5origin/cam2/*.jpg'))
            cam4_list = natsorted(glob(f'collect/scne{self.SCEN}_0.5origin/cam4/*.jpg'))
            return self.pair(cam0_list, cam2_list, cam4_list , flag = False)
        
        except Exception as e:
            logging.error(f"이미지 로딩 실패: {e}")
            return []

    def pair(self, cam0_list, cam2_list, cam4_list , flag=True):
        """이미 경로 페어링 메서드"""
        buffer = []
        # 딕셔너리 컴프리헨션에 에러 처리 추가
        try:
            cam0_dict = {os.path.basename(cam0_path): cam0_path for cam0_path in cam0_list}
            cam2_dict = {os.path.basename(cam2_path): cam2_path for cam2_path in cam2_list}
            cam4_dict = {os.path.basename(cam4_path): cam4_path for cam4_path in cam4_list}

            for path in cam2_list:
                name = os.path.basename(path)
                cam0_path = None
                cam4_path = None

                try:
                    if str(self.SCEN) in ['2', '3', '3x']:
                        index, _ = os.path.splitext(name)
                        adjusted_name = f"{str(int(index) + 10)}.jpg"
                        cam0_path = cam0_dict.get(adjusted_name)
                        cam4_path = cam4_dict.get(adjusted_name)
                    else:
                        cam0_path = cam0_dict.get(name)
                        cam4_path = cam4_dict.get(name)
                        
                    if flag and cam0_path is not None and cam4_path is not None:
                        buffer.append((cam0_path, path, cam4_path))
                    
                    elif flag == False and cam0_path is not None:
                        buffer.append((cam0_path, path))

                except ValueError as e:
                    logging.warning(f"파일명 '{name}' 처리 중 오류 발생: {e}")
                    continue

            return buffer

        except Exception as e:
            logging.error(f"페어링 처리 중 오류 발생: {e}")
            return []

    def prediction(self, img: Union[np.ndarray, torch.Tensor], flag: str) -> np.ndarray:
        try:
            result = self.model(img, classes=0)
            
            boxes = []
            scores = []
            for res in result:
                for box in res.boxes:
                    if int(box.cls) == 0:
                        x1, y1, w, h = box.xyxy[0].tolist()
                        score = box.conf
                        boxes.append([int(x1), int(y1), int(w - x1), int(h - y1)])
                        scores.append(float(score.detach().cpu().numpy()))

            filtered_boxes, filtered_scores = self.apply_nms(boxes, scores)
            nmx_boxes = list(map(self.nmx_box_to_cv2_loc, filtered_boxes))
            
            img_res = self.draw(nmx_boxes, img, flag)
            
            if flag == 'cam0':
                img_res , row_counter , detected_person_cordinate = self.LOCATION.cam0_run(img_res, nmx_boxes)
            elif flag == 'cam2':
                img_res , row_counter , detected_person_cordinate = self.LOCATION.cam2_run(img_res, nmx_boxes)
                
            
            return img_res , row_counter , detected_person_cordinate
        except Exception as e:
            
            logging.error(f"예측 중 오류 발생: {e}")
            raise e
    def test_run(self , image):
        result = self.model(image, classes=0 , device='cuda:0' , augment=True )
        result = result[0]
        if hasattr(result, 'plot'):
            return result.plot()
        else:
            return image
    def set_run(self) -> None:
        """실행 메서드"""
        try:
            if not self.set_list:
                logging.error("이미지 리스트가 비어있습니다.")
                return

            self.set_list = self.set_list[400:]

            
            for door_path, under_path in tqdm(self.set_list):
                door = cv2.imread(door_path, cv2.IMREAD_COLOR)
                under = cv2.imread(under_path, cv2.IMREAD_COLOR)
                
                if door is None or under is None:
                    logging.error(f"이미지 로드 실패: {door_path} or {under_path}")
                    continue

                door_plane = fisheye2plane.run(door, -40)
                under_plane = fisheye2plane.run(under, -40)
                under2_plane = fisheye2plane.run(under, 40 ,move = 5 , flag=False)
                cv2.namedWindow('under2', cv2.WINDOW_NORMAL)
                test_image = self.test_run(under2_plane)
                cv2.imshow('under2', test_image)
                # os.makedirs('under_plane', exist_ok=True)
                # cv2.imwrite(f'under_plane/under_{os.path.basename(under_path)}', under_plane)
                #tracking_image = self.tracking.tracking(under_plane.copy())
                #cv2.namedWindow('tracking', cv2.WINDOW_NORMAL)
                #cv2.imshow('tracking', tracking_image)
                under_prediction, position_details , _ = self.prediction(under_plane.copy(), flag='cam2')
                door_prediction, door_row_counter , detected_person_cordinate = self.prediction(door_plane.copy(), flag='cam0')
                
                cam0_detections = door_row_counter.copy()
                cam2_detections = position_details.copy()
                
             
                self.seat_detector.determine_seat_positions_cam2(cam2_detections)
                self.seat_detector.camera_calibration(cam0_detections, detected_person_cordinate)
                

                #visualization = self.visualizer.visualize_seats(self.seat_detector.get_seat_status())
                cv2.namedWindow("under", cv2.WINDOW_NORMAL)
                cv2.namedWindow('door', cv2.WINDOW_NORMAL)
                cv2.imshow('under', under_prediction)
                cv2.imshow('door', door_prediction)
                #cv2.imshow('visualization', visualization)
                
                tracking_image = self.tracking.tracking(under_plane.copy())
                cv2.namedWindow('tracking', cv2.WINDOW_NORMAL)
                cv2.imshow('tracking', tracking_image)
                
                pose_estimation = self.pose_estimator.PoseEstimation(under_plane.copy())
                cv2.namedWindow('pose_estimation', cv2.WINDOW_NORMAL)
                cv2.imshow('pose_estimation', pose_estimation)
                key = cv2.waitKey(0) & 0xFF
                if key == ord('c'):
                    while True:
                        if cv2.waitKey(0) & 0xFF == ord('c'):
                            break

        except Exception as e:
            logging.error(f"실행 중 오류 발생: {str(e)}")
            logging.error(f"오류 발생 위치:\n{traceback.format_exc()}")

    def nmx_box_to_cv2_loc(self, boxes):
        x1, y1, w, h = boxes
        x2 = x1 + w
        y2 = y1 + h
        return [x1, y1, x2, y2]

    def apply_nms(self, boxes, scores, iou_threshold=None):
        if iou_threshold is None:
            iou_threshold = self.iou_threshold
            
        if not boxes:
            return [], []
            
        # 박스와 점수를 numpy 배열로 변환
        boxes = np.array(boxes)
        scores = np.array(scores)
        
        # confidence threshold 적용
        mask = scores > self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        
        if len(boxes) == 0:
            return [], []
            
        # NMS 용
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 
                                 self.conf_threshold, iou_threshold)
        
        if len(indices) > 0:
            if isinstance(indices, np.ndarray):
                indices = indices.flatten()
            else:
                indices = [i[0] for i in indices]
                
            return boxes[indices].tolist(), scores[indices].tolist()
        
        return [], []

    def draw(self, nmx_boxes: list[int], img, flag=False) -> np.array:
        h, w, _ = img.shape
        for idx, i in enumerate(nmx_boxes):
            x1, y1, x2, y2 = i
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), thickness=2)
            cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)  
            cv2.putText(img, f'({center_x}, {center_y})', 
                        (center_x, center_y), 
                        cv2.FONT_HERSHEY_DUPLEX, 
                        0.3, 
                        (255, 255, 255), 
                        1)
        return img

class Location():
    
    right_points = [
        (299, 0),    # 시작점
        (299, 30),
        (299, 60),   # row1
        (299, 90),
        (299, 123),  # row2
        (297, 150),
        (295, 180),
        (293, 211),  # row3
        (292, 240),
        (291, 270),
        (291, 300),
        (290, 330),
        (290, 366),  # row4
        (289, 400),
        (288, 450),
        (287, 500),
        (287, 532),  # row5
        (288, 570),
        (289, 600),
        (290, 639)   # 끝점
    ]
    left_points = [
        (189, 0),    # 시작점
        (189, 30),
        (189, 60),   # row1
        (182, 90),
        (175, 123),  # row2
        (160, 150),
        (146, 180),
        (138, 211),  # row3
        (120, 240),
        (100, 270),
        (89, 300),
        (80, 330),
        (74, 366),   # row4
        (60, 400),
        (40, 450),
        (20, 500),
        (3, 532),    # row5
        (2, 570),
        (1, 600),
        (0, 639)     # 끝점
    ]
    
    y_limit = [0, 60, 123, 211, 366, 532]
    cam0_x_limit = [0, 93, 253, 427, 599]
    cam0_y_limit = [350 , 125 , 440] # uppper lower  발바닥 하한선
        
    side_seat_limit_x , side_seat_limit_y = [ 585 , 280  ] 
    side_seat_threshold_x = 415  # 좌석 9 10 노이즈제거 x1 기준 사용

    def cam2_run(self, img, boxes):
        h, w, _ = img.shape
        
        # Draw vertical line at x=585 in green
        cv2.line(img, (self.side_seat_limit_x, 0), (self.side_seat_limit_x, h), (0, 0, 255), 2)
        
        # Draw horizontal line at y=300 in yellow  
        cv2.line(img, (0, self.side_seat_limit_y), (w, self.side_seat_limit_y), (0, 255, 255), 2)
        cv2.line(img, (self.side_seat_threshold_x, 0), (self.side_seat_threshold_x, h), (0, 0, 255), 2)
        

        def find_intersection_x(y, points):
            """주어진 y좌표에서 선분들과 나는 x좌표 기"""
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]
                y1, y2 = p1[1], p2[1]
                
                if min(y1, y2) <= y <= max(y1, y2):
                    x1, x2 = p1[0], p2[0]
                    if y1 != y2:
                        x = x1 + (x2 - x1) * (y - y1) / (y2 - y1)
                        return x
                    else:
                        return x1
            return None

        memory = []
        self.position_info = []

        side_seats_occupied = {'side_seats': {'seat9': False, 'seat10': False}}
        for i in boxes:
            x1, y1, x2, y2 = i
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            if Location.side_seat_limit_x <= x2 and y2 >= Location.side_seat_limit_y and x1 >= Location.side_seat_threshold_x:
                if y2 <= 325:
                    side_seats_occupied['side_seats']['seat9'] = True
                    cv2.circle(img, (x2, y2), 5, (51, 51, 255), -1)
                    cv2.putText(img, "9", (x2-40, y2), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 1)  
                    continue
                else:
                    side_seats_occupied['side_seats']['seat10'] = True
                    cv2.circle(img, (x2, y2), 5, (51, 51, 255), -1)
                    cv2.putText(img, "10", (x2-40, y2), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 1)  
                    continue
                
            if  ( not center_x >= 300 and not y2 >= 570): # 570 cam2에서 마지막행 제외 
                idx = min(range(len(Location.y_limit)), key=lambda i: abs(Location.y_limit[i] - y1))
                memory.append(idx)
                
                left_x = find_intersection_x(center_y, Location.left_points)
                right_x = find_intersection_x(center_y, Location.right_points)
                
                if left_x is not None and right_x is not None:
                    dist_to_left = abs(center_x - left_x)
                    dist_to_right = abs(center_x - right_x)
                    position = "LEFT" if dist_to_left < dist_to_right else "RIGHT"
                    self.position_info.append((idx, position))
                    
                    cv2.circle(img, (int(left_x), center_y), 3, (255, 255, 0), -1)
                    cv2.circle(img, (int(right_x), center_y), 3, (255, 255, 0), -1)
                    
                    cv2.putText(img, f"{position} ({dist_to_left:.1f}, {dist_to_right:.1f})", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (0, 255, 255), 1)

        row_counter = Counter(memory)
        row_counter = dict(sorted(row_counter.items(), key=lambda x: x[0]))

        position_details = {}
        for idx, pos in self.position_info:
            if idx not in position_details:
                position_details[idx] = {"LEFT": 0, "RIGHT": 0}
            position_details[idx][pos] += 1


        for key in row_counter.keys():
            loc = Location.y_limit[key]
            left_count = position_details.get(key, {"LEFT": 0})["LEFT"]
            right_count = position_details.get(key, {"RIGHT": 0})["RIGHT"]
            total_count = row_counter[key]
            cv2.putText(img, f"Row{key}:L:{left_count} R:{right_count} (T:{total_count})", 
                       (100, loc + 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255))
            
        position_details = dict(sorted(position_details.items(), key=lambda x: x[0]))
        position_details['side_seat'] = side_seats_occupied['side_seats']
        return img, position_details , []
    
    def cam0_run(self, img, boxes):
        """
        cam0에서 감지된 사람들의 위치를 분석하고 시각화합니다.
        
        Args:
            img: 입력 이미지
            boxes: 감지된 바운딩 박 리스트 [[x1,y1,x2,y2], ...]
            
        Returns:
            tuple: (시각화된 이미지, row별 인원수, row별 좌표 정보)
        """
        def draw_guide_lines(img, h, w):
            # 가이드라인 그리기
            for x in self.cam0_x_limit:
                cv2.line(img, (x, 0), (x, h - 1), (255, 0, 0), 2)
            for y in self.cam0_y_limit:
                cv2.line(img, (0, y), (w - 1, y), (0, 0, 255), 2)
            return img

        def is_valid_detection(y1, y2, center_y):
            return (center_y <= self.cam0_y_limit[0] and 
                    y1 >= self.cam0_y_limit[1] and 
                    y2 <= self.cam0_y_limit[2])

        def determine_row(x1, lower_limit, upper_limit, current_row):
            if abs(lower_limit - x1) <= abs(upper_limit - x1):
                return current_row
            return current_row - 1 if current_row > 1 else None

        h, w, _ = img.shape
        img = draw_guide_lines(img, h, w)
        
        # 결과 저장을 위한 자료구조 초기화
        detected_person_coordinates = {f'row{i}': [] for i in range(1, 5)}
        detected_row_numbers = []

        # 각 감지된 박스 처리
        for x1, y1, x2, y2 in boxes:
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            if not is_valid_detection(y1, y2, center_y):
                continue
                
            cv2.rectangle(img, (x1, y1), (x2, y2), (199, 180, 200), 2)
            
            # row 번호 결정
            for idx in range(len(self.cam0_x_limit) - 1):
                lower_limit = self.cam0_x_limit[idx]
                upper_limit = self.cam0_x_limit[idx + 1]
                
                if lower_limit <= x1 <= upper_limit:
                    current_row = 4 - idx
                    row = determine_row(x1, lower_limit, upper_limit, current_row)
                    
                    if row:
                        detected_row_numbers.append(row)
                        detected_person_coordinates[f'row{row}'].append([center_x, center_y])
                    break

        # row별 인원수 계산 및 시각화
        row_counter = Counter(detected_row_numbers)
        row_counter.update({i: 0 for i in range(1, 5)})  # 빈 row 초기화
        
        # 인원수 표시
        for row_num, count in row_counter.items():
            loc = self.cam0_x_limit[4 - row_num]
            cv2.putText(img, f"T:{count}", 
                       (loc + 30, 30), 
                       cv2.FONT_HERSHEY_DUPLEX, 
                       0.7, (0, 0, 255))

        return img, dict(sorted(row_counter.items())), detected_person_coordinates
    
    def get_position_info(self):
        """position_info 반환"""
        return self.position_info
    
class SeatPositionDetector:
    def __init__(self):
        self.seat_positions = self._initialize_seats()
        
    def _initialize_seats(self):
        # 모든 좌석을 False로 초기화
        return {
            f'row{i}': {'left': False, 'right': False}
            for i in range(1, 5)
        } | {'side_seat': {'seat9': False, 'seat10': False}}

    def determine_seat_positions_cam2(self, cam2_detections):
        try:
            
            self.seat_positions = self._initialize_seats()
            
            for key, value in cam2_detections.items():
                if key == 'side_seat':
                    self._update_side_seats(value)
                    
              
                elif isinstance(key, int) and 1 <= key <= 4: # 1열 ~ 4열
                    row_name = f'row{key}'
                    self._update_seat_position(row_name, value)
                    
                elif isinstance(key , int) and  key == 0: # 0열은 추가적으로 1열으로 붙임
                    row_name = f'row{key + 1}'
                    self._update_seat_position(row_name, value)
                                        
                    
            # 디버깅을 위한 출력
            for i in range(1, 5):
                row_name = f'row{i}'
                print(f"{row_name} : {self.seat_positions[row_name]}")
            print(f"side_seat : {self.seat_positions['side_seat']}")
                    
        except Exception as e:
            logging.error(f"좌석 위치 결정 중 오류 발생: {e}")
            raise
        
    def camera_calibration(self, cam0_detections , detected_person_cordinate):
        """
        카메라 캘리브레이션 처리
        cam 0 와 cam 2 의 좌석 위치 정보를 통해 카메라 캘리브레이션 처리
        cam 0 형태 : {0: 1, 1: 1, 2: 1, 3: 1 , 4: 1}
        cam 2 형태 : {0: {'LEFT': 1, 'RIGHT': 0}, 1: {'LEFT': 1, 'RIGHT': 0}, 2: {'LEFT': 1, 'RIGHT': 0}, 3: {'LEFT': 1, 'RIGHT': 0}}
        detected_person_cordinate : {f'row{i}': [[center_x, center_y], [center_x, center_y]]}
        """
        # key를 row key로 변환
        row_detections = {}
        for key in cam0_detections:
            row_key = f'row{key}'
            row_detections[row_key] = cam0_detections[key]
                    
        for i in range(1,5): # 카메라 cam0 cam2를 이용한 좌석 판단 작성. # row1 ~ row4
            row_name = f'row{str(i)}'
            cam0_row_count = row_detections.get(row_name, 0) # 해당행의 사람명수 
            cam2_row_count_left = self.seat_positions[row_name].get('left', 0)
            cam2_row_count_right = self.seat_positions[row_name].get('right', 0)
            
            if cam0_row_count == 0 and cam2_row_count_left + cam2_row_count_right == 0:
                # 정상 케이스: 해당 열에 아무도 없음
                self.seat_positions[row_name]['left'] = 0
                self.seat_positions[row_name]['right'] = 0
                
            elif cam0_row_count == 0 and (cam2_row_count_left + cam2_row_count_right > 0):
                # cam2에서만 감지된 경우 - cam2 결과 사용
                self.seat_positions[row_name]['left'] = cam2_row_count_left 
                self.seat_positions[row_name]['right'] = cam2_row_count_right
                
            elif cam0_row_count == 1 and cam2_row_count_left + cam2_row_count_right == 0:
                # cam0에서만 감지된 경우 (사각지대) - 좌우 판단 불가로 pass 그러 row3번은 손잡이로인해 cam0이 하나 cam2가 0일떄 좌로가정
       
                try:
                    if detected_person_cordinate[row_name] and len(detected_person_cordinate[row_name]) > 0:
                        _, cy = detected_person_cordinate[row_name][0]
            
                        if cy >= 270:
                            self.seat_positions[row_name]['right'] = 1
                        else:
                            self.seat_positions[row_name]['left'] = 1
                    else:
                        raise Exception(f"{row_name}에 대한 좌표 데이터가 없습니다.")
                        
                except Exception as e:
                    print("예외발생")
                    print(f"Row name: {row_name}")
                    print('cam0_detections' , cam0_detections)
                    print('cam2_detections' , self.seat_positions)
                    print(f"Detected coordinates: {detected_person_cordinate}")
                    print('row_detections' , row_detections)
                    import pdb; pdb.set_trace() 
                    print("예외 메시지:", str(e))
                    print("상세 traceback:")
                    print(traceback.format_exc())
                        
            elif cam0_row_count == 1 and cam2_row_count_left + cam2_row_count_right == 1:
                # 정상 케이스: 한 이 정확히 감지됨
                self.seat_positions[row_name]['left'] = cam2_row_count_left
                self.seat_positions[row_name]['right'] = cam2_row_count_right
                
            elif cam0_row_count == 1 and cam2_row_count_left + cam2_row_count_right == 2:
                # cam2에서 2명 감지 - 좌우 모 착석으로 처리
                self.seat_positions[row_name]['left'] = 1
                self.seat_positions[row_name]['right'] = 1
                
            elif cam0_row_count == 2 and cam2_row_count_left + cam2_row_count_right == 0:
                # cam0에서 2명 감지 - 좌우 모두 착석으로 처리
                self.seat_positions[row_name]['left'] = 1
                self.seat_positions[row_name]['right'] = 1
                
            elif cam0_row_count == 2 and cam2_row_count_left + cam2_row_count_right == 1:
                # cam0에서 2명 감지 - 좌우 모두 착석으로 처리
                self.seat_positions[row_name]['left'] = 1
                self.seat_positions[row_name]['right'] = 1
                
            elif cam0_row_count == 2 and cam2_row_count_left + cam2_row_count_right == 2:
                # 양쪽 카메라에서 모두 2명 감지 - 좌우 모두 착석으로 처리
                self.seat_positions[row_name]['left'] = 1
                self.seat_positions[row_name]['right'] = 1
                
        # print("row_detections")
        # print(row_detections)
        # print("self.seat_positions")
        # print(self.seat_positions)    
        
        cam0_detections.clear()
        cam0_detections.update(row_detections)
        
    
    def _update_seat_position(self, row_name, detection_info):
        # LEFT와 RIGHT 값을 명시적으로 가져와서 처리
        left_occupied = detection_info.get('LEFT', 0) > 0
        right_occupied = detection_info.get('RIGHT', 0) > 0
        
        self.seat_positions[row_name]['left'] = left_occupied
        self.seat_positions[row_name]['right'] = right_occupied

    def _update_side_seats(self, side_seats):
        self.seat_positions['side_seat']['seat9'] = bool(side_seats.get('seat9', False))
        self.seat_positions['side_seat']['seat10'] = bool(side_seats.get('seat10', False))

    def get_seat_status(self):
        """
        현재 좌석 상태를 딕셔너리 형태로 반환
        
        Returns:
            dict: 각 좌석의 상태를 포함하는 딕셔너리
        """
        return self.seat_positions
    
    def print_seat_status(self):
        """
        모든 좌석의 태를 포맷팅하여 출력
        """
        print("\n=== 좌석 상태 ===")
        # 일반 좌석 출력
        for row in range(1, 5):
            row_name = f'row{row}'
            positions = self.seat_positions[row_name]
            print(f"\n{row}열:")
            print(f"  왼쪽 좌석: {'착석' if positions['left'] else '비어있음'}")
            print(f"  오른쪽 좌석: {'착석' if positions['right'] else '비어있음'}")
        
        # 측면 좌석 출력
        print("\n측면 좌석:")
        side_seats = self.seat_positions['side_seat']
        print(f"  9번 좌석: {'착석' if side_seats['seat9'] else '비어있음'}")
        print(f"  10번 좌석: {'착석' if side_seats['seat10'] else '비어있음'}")

class YoloPoseEstimator:
    def __init__(self):
        self.pose_estimator = YOLO('yolo11x-pose.pt')
        # 하체 키포인트 인덱스 정의
        self.lower_body_keypoints = {
            0: 'nose',          # 코
            1: 'neck',          # 목
            5: 'left_shoulder', # 왼쪽 어깨
            6: 'right_shoulder',# 오른쪽 어깨
            7 : 'left_elbow',   # 왼쪽 팔꿈치
            8 : 'right_elbow',  # 오른쪽 팔꿈치
            11: 'left_hip',     # 왼쪽 엉덩이
            12: 'right_hip',    # 오른쪽 엉덩이
            13: 'left_knee',    # 왼쪽 무릎
            14: 'right_knee',   # 오른쪽 무릎
            15: 'left_ankle',   # 왼쪽 발목
            16: 'right_ankle',  # 오른쪽 발목
        }
        
        self.naming_keypoints = {}
        
        for key , value in self.lower_body_keypoints.items():
            name = value.replace('_', ' ')
            if '_' in value:
                direction = value.split('_')[0][:1]
                name = value.split('_')[1][:3]
                full_name = f"{direction}_{name}"
            else:
                full_name = name[:]
            
            self.naming_keypoints.update({key: full_name})

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
                
                if start_pos[0] > 0 and start_pos[1] > 0 and end_pos[0] > 0 and end_pos[1] > 0: 
                    # 선 그리기
                    cv2.line(img, start_pos, end_pos, self.LINE_COLOR, 2)
                    
                    # 선분의 중앙점 계산
                    mid_x = int((start_pos[0] + end_pos[0]) / 2)
                    mid_y = int((start_pos[1] + end_pos[1]) / 2)
                else:
                    continue
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
        left_elbow = person_keypoints[7].cpu().numpy()
        right_elbow = person_keypoints[8].cpu().numpy()
        
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
            if left_shoulder[0] > right_shoulder[0]:
                upper_body_angle = 360 - upper_body_angle
                
            return upper_body_angle
        
        else:
            return None
        

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
                        
            output_img = img.copy()
            zero_image = np.zeros_like(img, dtype=np.uint8)
            zero_image[ : , 300:, :] = img[:, 300:, :]
            img = zero_image.copy()

            results = self.pose_estimator(img,
                                        half=False,
                                        iou=0.5,
                                        conf=self.CONFIDENCE_THRESHOLD,
                                        device='cuda:0',
                                        classes=[0])

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
                    # if head_angle is not None:
                    #     cv2.putText(output_img,
                    #               f'Head Angle: {int(head_angle)}',
                    #               (x1, y_offset),
                    #               cv2.FONT_HERSHEY_SIMPLEX,
                    #               0.5,
                    #               (0,0,255),
                    #               2)
                    #     y_offset -= 20
                    
                    # if upper_body_angle is not None:
                    #     cv2.putText(output_img,
                    #               f'Upper Body: {int(upper_body_angle)}',
                    #               (x1, y_offset),
                    #               cv2.FONT_HERSHEY_SIMPLEX,
                    #               0.5,
                    #               (0,0,255),
                    #               2)
                    #     y_offset -= 20
                    

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
                            cv2.putText(output_img,
                                    f'Upper Body: {int(angle)}',
                                    (x1, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0,0,255),
                                    2)
                            output_img = self.draw_angle_arrow(output_img, (center_x, center_y), angle, color=(255, 192, 203))
                            
                    self.draw_skeleton(output_img, person_keypoints)
                    self.draw_keypoints(output_img, person_keypoints)
            return output_img
            
        except Exception as e:
            logging.error(f"Pose estimation error: {str(e)}")
            return img

class YoloTracking:
    def __init__(self):
        self.tracking_model = YOLO('yolo11x.pt')
        self.dynamic_status = {}
        self.prev_frame_ids = set()  # 이전 프레임의 객체 ID 저장
        self.CONFIDENCE_AVG_THRESHOLD = 1
        
        self.image_idx = 0
        self.save_dir = 'tracking_images'
        os.makedirs(self.save_dir, exist_ok=True)
        
    def tracking(self, img):
        zero_image = np.zeros_like(img, dtype=np.uint8)
        zero_image[ : , 300:, :] = img[:, 300:, :]
        model_input_image = zero_image.copy()
        print("===============================tracking================================")
        results = self.tracking_model.track(
            model_input_image,
            persist=True,  # tracking 지속성 유지
            classes=[0],
            half=False,
            device='cuda:0',
            iou=0.4,
            conf=0.1,
            imgsz=640,
            augment=True,
            visualize=True,
        )
        results = results[0]

        if hasattr(results, 'show'):
            image_plot = results.plot()
        image_plot[ : , : 300 , :] = img[ : , : 300 , :]
        movement_status = {}  # 객체별 동적/정적 상태
        current_frame_ids = set()  # 현재 프레임의 객체 ID 저장

        for obj in results.boxes:
            if obj.id is None:  # ID가 없는 경우 건너뛰기
                continue
                
            obj_id = int(obj.id)  # 객체의 고유 ID를 정수로 변환
            current_frame_ids.add(obj_id)  # 현재 프레임 ID 기록
            
            x1 , y1 , x2 , y2 = list(map(int, obj.xyxy[0].cpu().numpy()))
            #crop_img = img[y1:y2, x1:x2]
            #cv2.imwrite(os.path.join(self.save_dir, f'{self.image_idx:06d}_{obj_id:02d}.jpg'), crop_img)
            self.image_idx += 1
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)
            # 해당 객체 ID의 좌표 리스트가 없으면 deque로 초기화 (최대 5개 유지)
            if obj_id not in self.dynamic_status:
                self.dynamic_status[obj_id] = deque(maxlen=5)

            # 새 좌표를 deque에 추가
            self.dynamic_status[obj_id].append((x2, y1))
            cv2.circle(image_plot, (x2, y1), 5, (0, 165, 255), -1)


            max_distance = 0
            avg_distance = 0
            # 최근 5개 좌표 변화 분석
            if len(self.dynamic_status[obj_id]) >= 5:
                max_distance = 0
                avg_distance = 0
                coordinates = list(self.dynamic_status[obj_id])
                
                for i in range(len(coordinates)):
                    for j in range(i+1, len(coordinates)):
                        distance = ((coordinates[i][0] - coordinates[j][0])**2 + 
                                  (coordinates[i][1] - coordinates[j][1])**2) ** 0.5
                        if distance > max_distance:
                            max_distance = distance
                else:
                    avg_distance = max_distance / len(coordinates)
                    
                if avg_distance > self.CONFIDENCE_AVG_THRESHOLD:
                    movement_status[obj_id] = ("MOVING", avg_distance)
                    cv2.putText(image_plot, "MOVING", (x_center, y_center), 
                              cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                else:
                    movement_status[obj_id] = ("STATIC",  avg_distance)
                    cv2.putText(image_plot, "STATIC", (x_center, y_center), 
                              cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
                
                cv2.putText(image_plot, f"{avg_distance:.2f}", (x1, y1 + 20), 
                              cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
            else:
                movement_status[obj_id] = ("PENDING", avg_distance)
                cv2.putText(image_plot, "PENDING", (x_center, y_center), 
                          cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)

        # 현재 프레임에 없는 객체의 기록 관리
        disappeared_ids = self.prev_frame_ids - current_frame_ids
        
        # 사라진 객체의 프레임 카운트 관리를 위한 딕셔너리 초기화
        if not hasattr(self, 'disappeared_counts'):
            self.disappeared_counts = {}
            
        # 사라진 객체들의 카운트 증가
        for disappeared_id in disappeared_ids:
            if disappeared_id not in self.disappeared_counts:
                self.disappeared_counts[disappeared_id] = 1
            else:
                self.disappeared_counts[disappeared_id] += 1
                
            # 5프레임 이상 사라진 객체의 기록 삭제
            if self.disappeared_counts[disappeared_id] >= 5:
                if disappeared_id in self.dynamic_status:
                    del self.dynamic_status[disappeared_id]
                del self.disappeared_counts[disappeared_id]
                
        # 현재 프레임에 다시 나타난 객체의 카운트 초기화
        for current_id in current_frame_ids:
            if current_id in self.disappeared_counts:
                del self.disappeared_counts[current_id]

        # 현재 프레임 ID를 이전 프레임 ID로 업데이트
        self.prev_frame_ids = current_frame_ids
        
        print("\n=== Movement Status ===")
        # sorted()는 리스트를 반환하므로, 직접 정렬된 항목들을 순회
        sorted_status = sorted(movement_status.items(), key=lambda x: x[0])
        for obj_id, (status, distance) in sorted_status:
            print(f"ID {obj_id:2d}: {status:8s} (distance: {distance:.2f})")

        return image_plot

class YoloSegmentation:
    colors = [
            (51, 255, 255),  # 노란색 (Yellow)
            (51, 255, 51),   # 초록색 (Green)
            (255, 51, 51),   # 파란색 (Blue)
            (153, 51, 51),   # 남색 (Navy)
            (255, 51, 255)   # 보라색 (Purple)
        ]
    def __init__(self):
        self.segmentation_model = YOLO('yolo11x-seg.pt')
  
    def segmentation(self, img):
        """
        이미지에서 객체를 감지하고 세그멘테이션을 수행합니다.
        
        Args:
            img (np.ndarray): 입력 이미지
            
        Returns:
            np.ndarray: 세그멘테이션 마스크가 적용된 이미지
        """

        try:
            results = self.segmentation_model(img)
            result_img = img.copy()
            results = results[0]
            
            # 클래스별 고유 색상 생성 (클래스 ID를 키로 사용)
            class_colors = {}
            if hasattr(results, 'masks'):
                plot_image = results.plot()
                return plot_image
            
            if hasattr(results, 'masks') and results.masks is not None:
                # 마스크와 클래스 정보 가져오기
                masks = results.masks.xyn
                classes = results.boxes.cls
                
                for mask, class_id in zip(masks, classes):
                    # 클래스별 색상 할당 (없으면 새로 생성)
                    if class_id not in class_colors:
                        class_colors[class_id] = YoloSegmentation.colors[class_id % len(YoloSegmentation.colors)]   

                    # 마스크를 이미지 크기에 맞게 조정
                    mask = mask.astype(np.uint8)
                    h, w = img.shape[:2]
                    mask = cv2.resize(mask, (w, h))
                    
                    # 마스크 적용
                    color = class_colors[class_id]
                    colored_mask = np.zeros_like(img)
                    colored_mask[mask > 0] = color
                    
                    # 투명도를 적용하여 원본 이미지와 블렌딩
                    alpha = 0.5
                    mask_area = mask > 0
                    result_img[mask_area] = cv2.addWeighted(
                        result_img[mask_area], 
                        1 - alpha,
                        colored_mask[mask_area], 
                        alpha, 
                        0
                    )
                    
                    # 클래스 이름 표시 (마스크의 상단에)
                    mask_points = np.where(mask > 0)
                    if len(mask_points[0]) > 0:
                        top_point = (int(np.mean(mask_points[1])), int(np.min(mask_points[0])))
                        class_name = f"Class {int(class_id)}"  # 클래스 이름이 있다면 그것을 사용
                        cv2.putText(
                            result_img,
                            class_name,
                            top_point,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2
                        )

            cv2.namedWindow("segmentation", cv2.WINDOW_NORMAL)
            cv2.imshow("segmentation", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            return result_img
            
        except Exception as e:
            logging.error(f"Segmentation 처리 중 오류 발생: {str(e)}")
            logging.error(traceback.format_exc())
            return img
    
if __name__ == "__main__":
    try:
        C = YoloDetector()
        C.set_run()
    except Exception as e:
        logging.error(f"메인 실행 중 오류 발생: {e}")
        
        
        