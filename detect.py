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
    def __init__(self, SCEN='2'):
        self.model = YOLO('yolo11x.pt')
        self.LOCATION = Location()
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.SCEN = SCEN
        self.set_list = self._load_and_pair_images()
        self.seat_detector = SeatPositionDetector()
        self.visualizer = SitRecognition()
    

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
        """이미지 경로 페어링 메서드"""
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
                img_res , row_counter , people_row_cordinate = self.LOCATION.cam0_run(img_res, nmx_boxes)
            elif flag == 'cam2':
                img_res , row_counter , people_row_cordinate = self.LOCATION.cam2_run(img_res, nmx_boxes)
            
            
            return img_res , row_counter, people_row_cordinate
        except Exception as e:
            
            logging.error(f"예측 중 오류 발생: {e}")
            raise e

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
                
                under_prediction, position_details , people_row_cordinate = self.prediction(under_plane, flag='cam2')
                door_prediction, door_row_counter , people_row_cordinate = self.prediction(door_plane, flag='cam0')
                
                cam0_detections = door_row_counter.copy()
                cam2_detections = position_details.copy()
                # print("position_details: ", position_details)
                # print("===cam2_detections===")
                # for key, value in cam2_detections.items():
                #     if key != 'side_seat':  # side_seat는 별도 처리
                #         print(f"Row {key}: {value}")
                # if 'side_seat' in cam2_detections:
                #     print(f"Side seats: {cam2_detections['side_seat']}")

                print("\n=== cam0_detections 상태 ===")
                print(cam0_detections)
                print('cam2_detections: ', cam2_detections)
                self.seat_detector.determine_seat_positions_cam2(cam2_detections)
                self.seat_detector.camera_calibration( cam0_detections , people_row_cordinate )
                #self.seat_detector.print_seat_status()
                
                visualization = self.visualizer.visualize_seats(self.seat_detector.get_seat_status())
                cv2.namedWindow("under", cv2.WINDOW_NORMAL)
                cv2.namedWindow('door', cv2.WINDOW_NORMAL)
                cv2.imshow('under', under_prediction)
                cv2.imshow('door', door_prediction)
                cv2.imshow('visualization', visualization)
                
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
            
        # NMS 적용
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
            """주어진 y좌표에서 선분들과 만나는 x좌표 ��기"""
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
        # print("\n=== 열별 전체 인원 ===")
        # print("Row counts:", row_counter)

        position_details = {}
        for idx, pos in self.position_info:
            if idx not in position_details:
                position_details[idx] = {"LEFT": 0, "RIGHT": 0}
            position_details[idx][pos] += 1

        # print("\n=== 열별 좌우 위치 상세 ===")
        # for row in sorted(position_details.keys()):
        #     left_count = position_details[row]["LEFT"]
        #     right_count = position_details[row]["RIGHT"]
        #     print(f"Row {row}: LEFT={left_count}, RIGHT={right_count}")

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
        h, w, _ = img.shape
        # 수직선 그리기
        for i in Location.cam0_x_limit:
            cv2.line(img, (i, 0), (i, h - 1), (255, 0, 0), 2)
            
        for i in Location.cam0_y_limit:
            cv2.line(img, (0, i), (w - 1, i), (0, 0, 255), 2)     
            
        people_row_cordinate = {f'row{i}': [] for i in range(len(Location.cam0_x_limit) - 1)}
        people_where = []
        
        # 각 사람의 위치 확인
        for x1, y1, x2, y2 in boxes:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            if center_y <= Location.cam0_y_limit[0] and y1 >= Location.cam0_y_limit[1] and y2 <= Location.cam0_y_limit[2]:
                cv2.rectangle(img, (x1, y1), (x2, y2), (199, 180, 200), 2)
                for idx in range(len(Location.cam0_x_limit) - 1):
                    lower_limit = Location.cam0_x_limit[idx]
                    upper_limit = Location.cam0_x_limit[idx + 1]

                    if lower_limit <= x1 <= upper_limit:
                        if abs(lower_limit - x1) <= abs(upper_limit - x1):
                            people_where.append(idx)
                            people_row_cordinate[f'row{idx}'].append([center_x, center_y])
                        else:
                            people_where.append(idx + 1)
                            people_row_cordinate[f'row{idx + 1}'].append([center_x, center_y])
                        break  

        # 각 열의 인원 수 계산
        row_counter = Counter(people_where)
        
        # 빈 row_counter인 경우에도 1-4 키를 모두 포함하도록 초기화
        default_counter = {i: 0 for i in range(1, 5)}
        if row_counter:
            row_counter = {4 - key: value for key, value in row_counter.items()}
        else:
            row_counter = default_counter

        print("row_counter")
        print(row_counter)
        
        # 각 열에 인원 수 표시
        for key, value in row_counter.items():
            loc = Location.cam0_x_limit[key - 1]
            cv2.putText(img, f"T:{value}", 
                       (loc + 30, 30), 
                       cv2.FONT_HERSHEY_DUPLEX, 
                       0.7, (0, 0, 255))

        row_counter = dict(sorted(row_counter.items(), key=lambda x: x[0]))
        return img, row_counter, people_row_cordinate
    
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
        
    def camera_calibration(self, cam0_detections, people_row_cordinate):
        """
        카메라 캘리브레이션 처리
        cam 0 와 cam 2 의 좌석 위치 정보를 통해 카메라 캘리브레이션 처리
        cam 0 형태 : {0: 1, 1: 1, 2: 1, 3: 1 , 4: 1}
        cam 2 형태 : {0: {'LEFT': 1, 'RIGHT': 0}, 1: {'LEFT': 1, 'RIGHT': 0}, 2: {'LEFT': 1, 'RIGHT': 0}, 3: {'LEFT': 1, 'RIGHT': 0}}
        people_row_cordinate : {f'row{i}': [[center_x, center_y], [center_x, center_y]]}
        """
        # key를 row key로 변환
        row_detections = {}
        for key in cam0_detections:
            row_key = f'row{key}'
            row_detections[row_key] = cam0_detections[key]
                    
        for i in range(1,5): # 카메라 cam0 cam2를 이용한 좌석 판단 작성.
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
                # cam0에서만 감지된 경우 (사각지대) - 좌우 판단 불가로 pass 그러나 row3번은 손잡이로인해 cam0이 하나 cam2가 0일떄 좌로가정
                if row_name == 'row3':
                    self.seat_positions[row_name]['right'] = 0
                    self.seat_positions[row_name]['left'] = 1
                else: # cam0  떔빵용 
                    _  , cy = people_row_cordinate[row_name][0]
                    if cy >= 270:
                        self.seat_positions[row_name]['right'] = 1
                    else:
                        self.seat_positions[row_name]['left'] = 1
                        
            elif cam0_row_count == 1 and cam2_row_count_left + cam2_row_count_right == 1:
                # 정상 케이스: 한 명이 정확히 감지됨
                self.seat_positions[row_name]['left'] = cam2_row_count_left
                self.seat_positions[row_name]['right'] = cam2_row_count_right
                
            elif cam0_row_count == 1 and cam2_row_count_left + cam2_row_count_right == 2:
                # cam2에서 2명 감지 - 좌우 모두 착석으로 처리
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
                
        print("row_detections")
        print(row_detections)
        print("self.seat_positions")
        print(self.seat_positions)    
        
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
        모든 좌석의 상태를 포맷팅하여 출력
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

if __name__ == "__main__":
    try:
        C = YoloDetector()
        C.set_run()
    except Exception as e:
        logging.error(f"메인 실행 중 오류 발생: {e}")
