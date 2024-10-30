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
        
    def _load_and_pair_images(self):
        """이미지 로딩 및 페어링"""
        try:
            cam0_list = natsorted(glob(f'collect/scne{self.SCEN}_0.5origin/cam0/*.jpg'))
            cam2_list = natsorted(glob(f'collect/scne{self.SCEN}_0.5origin/cam2/*.jpg'))
            return self.pair(cam0_list, cam2_list)
        
        except Exception as e:
            logging.error(f"이미지 로딩 실패: {e}")
            return []

    def pair(self, cam0_list, cam2_list, flag=True):
        """이미지 경로 페어링 메서드"""
        buffer = []
        cam0_dict = {os.path.basename(cam0_path): cam0_path for cam0_path in cam0_list}

        for path in cam2_list:
            name = os.path.basename(path)

            if str(self.SCEN) in ['2', '3', '3x']:
                index, _ = os.path.splitext(name)
                cam0_path = cam0_dict.get(f"{str(int(index) + 10)}.jpg")
            else:
                cam0_path = cam0_dict.get(name)

            if flag:
                if cam0_path is not None:
                    buffer.append((cam0_path, path))

        return buffer

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
                img_res , row_counter = self.LOCATION.cam0_run(img_res, nmx_boxes)
            elif flag == 'cam2':
                img_res , row_counter = self.LOCATION.cam2_run(img_res, nmx_boxes)
            
            
            return img_res , row_counter
        except Exception as e:
            logging.error(f"예측 중 오류 발생: {e}")
            return img , {}

    def set_run(self) -> None:
        """실행 메서드"""
        try:
            if not self.set_list:
                logging.error("이미지 리스트가 비어있습니다.")
                return

            self.set_list = self.set_list[450:]
            
            for door_path, under_path in tqdm(self.set_list):
                door = cv2.imread(door_path, cv2.IMREAD_COLOR)
                under = cv2.imread(under_path, cv2.IMREAD_COLOR)
                
                if door is None or under is None:
                    logging.error(f"이미지 로드 실패: {door_path} or {under_path}")
                    continue

                door_plane = fisheye2plane.run(door, -40)
                under_plane = fisheye2plane.run(under, -40)
                
                under_prediction, position_details = self.prediction(under_plane, flag='cam2')
                door_prediction, door_row_counter = self.prediction(door_plane, flag='cam0')
                
                # 디버깅을 위한 출력
                print("\nposition_details:", position_details)
                print("door_row_counter:", door_row_counter)
 
                cam0_detections = door_row_counter.copy()
                cam2_detections = position_details.copy()
                for idx, (key, value) in enumerate(cam2_detections.items()):
                    print(f"Row {idx+1}: {key} {value}")
                    
                print("\n=== cam0_detections 상태 ===")
                print(cam0_detections)

                self.seat_detector.determine_seat_positions(cam0_detections, cam2_detections)
                self.seat_detector.print_seat_status()
                
                visualization = self.visualizer.visualize_seats(self.seat_detector.get_seat_status())
                cv2.namedWindow("under", cv2.WINDOW_NORMAL)
                cv2.namedWindow('door', cv2.WINDOW_NORMAL)
                cv2.imshow('under', under_prediction)
                cv2.imshow('door', door_prediction)
                cv2.imshow('visualization', visualization)
                
                if cv2.waitKey(0) & 0xFF == ord('q'):
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
    
    side_seat_limit_x , side_seat_limit_y = [ 565 , 200 ] 
    
    def cam2_run(self, img, boxes):
        h, w, _ = img.shape
        
        # for i in range(len(Location.right_points) - 1):
        #     cv2.line(img, Location.right_points[i], Location.right_points[i + 1], (0, 255, 0), 2)
        # for i in range(len(Location.left_points) - 1):
        #     cv2.line(img, Location.left_points[i], Location.left_points[i + 1], (0, 255, 0), 2)      
        # for i in Location.y_limit:
        #     cv2.line(img, (0, i), (w, i - 1), (255, 0, 0), 2)
        
        # cv2.line(img, (300, 0), (300, h - 1), (0, 0, 255), 2)
        
        def find_intersection_x(y, points):
            """주어진 y좌표에서 선분들과 만나는 x좌표 찾기"""
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
            
            cv2.line(img, (0, Location.side_seat_limit_y), (w, Location.side_seat_limit_y), (0, 255, 255), 2)
            cv2.line(img, (Location.side_seat_limit_x, 0), (Location.side_seat_limit_x, h-1), (200, 200, 200), 2)
            cv2.line(img , (0 , 325), (w , 325), (0, 0 , 255), 2)
            
            if Location.side_seat_limit_x <= x2 and y2 >= Location.side_seat_limit_y:
                if y2 >= 325:
                    side_seats_occupied['side_seats']['seat9'] = True
                else:
                    side_seats_occupied['side_seats']['seat10'] = True
            
            if not (center_x >= 300):
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
        print("\n=== 열별 전체 인원 ===")
        print("Row counts:", row_counter)

        position_details = {}
        for idx, pos in self.position_info:
            if idx not in position_details:
                position_details[idx] = {"LEFT": 0, "RIGHT": 0}
            position_details[idx][pos] += 1

        print("\n=== 열별 좌우 위치 상세 ===")
        for row in sorted(position_details.keys()):
            left_count = position_details[row]["LEFT"]
            right_count = position_details[row]["RIGHT"]
            print(f"Row {row}: LEFT={left_count}, RIGHT={right_count}")

        for key in row_counter.keys():
            loc = Location.y_limit[key]
            left_count = position_details.get(key, {"LEFT": 0})["LEFT"]
            right_count = position_details.get(key, {"RIGHT": 0})["RIGHT"]
            total_count = row_counter[key]
            
            cv2.putText(img, f"L:{left_count} R:{right_count} (T:{total_count})", 
                       (100, loc + 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255))
        position_details = dict(sorted(position_details.items(), key=lambda x: x[0]))
        position_details['side_seats'] = side_seats_occupied['side_seats']
        return img, position_details
    
    def cam0_run(self, img, boxes):
        h, w, _ = img.shape
        # 수직선 그리기
        for i in Location.cam0_x_limit:
            cv2.line(img, (i, 0), (i, h - 1), (255, 0, 0), 2)
        
        people_where = []
        # 각 사람의 위치 확인
        for x1, _, _, _ in boxes:
            for idx in range(len(Location.cam0_x_limit) - 1):
                lower_limit = Location.cam0_x_limit[idx]
                upper_limit = Location.cam0_x_limit[idx + 1]

                if lower_limit <= x1 <= upper_limit:
                    if abs(lower_limit - x1) <= abs(upper_limit - x1):
                        people_where.append(idx)
                    else:
                        people_where.append(idx + 1)
                    break  

        # 각 열의 인원 수 계산
        row_counter = Counter(people_where)
        row_counter = {4 - key: value for key, value in sorted(row_counter.items())}
        # print("\n=== 열별 전체 인원 ===")
        # print("Row counts:", row_counter)
        
        # 각 열에 인원 수 표시
        for key, value in row_counter.items():
            loc = Location.cam0_x_limit[4 - key]
            cv2.putText(img, f"T:{value}", 
                       (loc + 30, 30), 
                       cv2.FONT_HERSHEY_DUPLEX, 
                       0.7, (0, 0, 255))
            
        return img , row_counter
    
    def get_position_info(self):
        """position_info 반환"""
        return self.position_info
    
class SeatPositionDetector:
    def __init__(self):
        self.seat_positions = {
            'row1': {'left': False, 'right': False},
            'row2': {'left': False, 'right': False},
            'row3': {'left': False, 'right': False},
            'row4': {'left': False, 'right': False},
            'side_seats': {'seat9': False, 'seat10': False}
        }
        
    def determine_seat_positions(self, cam0_detections, cam2_detections):
        """
        cam0_detections: {4: 2, 3: 1, 2: 1, 1: 2} 형태의 딕셔너리
        cam2_detections: {
            1: {'LEFT': 1, 'RIGHT': 0},
            2: {'LEFT': 1, 'RIGHT': 0},
            3: {'LEFT': 1, 'RIGHT': 0},
            4: {'LEFT': 1, 'RIGHT': 0},
            'side_seats': {'seat9': 0, 'seat10': 0}
        } 형태의 딕셔너리
        """
        # 4개의 행 처리
        for row_idx in range(1, 5):
            row_name = f'row{row_idx}'
            cam2_info = cam2_detections.get(row_idx, {'LEFT': 0, 'RIGHT': 0})
            
            # 초기화 및 상태 설정
            self.seat_positions[row_name]['left'] = self._check_position(cam2_info, 'LEFT')
            self.seat_positions[row_name]['right'] = self._check_position(cam2_info, 'RIGHT')
        
        # 측면좌석 판별
        side_seats = cam2_detections.get('side_seats', {'seat9': 0, 'seat10': 0})
        self.seat_positions['side_seats']['seat9'] = bool(side_seats.get('seat9', 0))
        self.seat_positions['side_seats']['seat10'] = bool(side_seats.get('seat10', 0))

    def _check_position(self, detection_info, position):
        """
        특정 위치(LEFT/RIGHT)의 착석 여부 확인
        
        Args:
            detection_info (dict): {'LEFT': count, 'RIGHT': count} 형태의 딕셔너리
            position (str): 'LEFT' 또는 'RIGHT'
        """
        return detection_info.get(position, 0) > 0
    
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
        side_seats = self.seat_positions['side_seats']
        print(f"  9번 좌석: {'착석' if side_seats['seat9'] else '비어있음'}")
        print(f"  10번 좌석: {'착석' if side_seats['seat10'] else '비어있음'}")

if __name__ == "__main__":
    try:
        C = YoloDetector()
        C.set_run()
    except Exception as e:
        logging.error(f"메인 실행 중 오류 발생: {e}")
