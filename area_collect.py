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
from pathlib import Path

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
    def __init__( self , cam0_list , cam2_list):
        self.model = YOLO('yolo11x.pt').to('cuda')
        self.cam0_list = cam0_list
        self.cam2_list = cam2_list
        self.set_list = self.pair(self.cam0_list , self.cam2_list )
        self.circle_list = []
        self.circle_array = np.zeros((0, 2), dtype=int)  
        
        self.score_map = np.zeros(shape=(640, 640 ), dtype=np.float32)
        #self.score_map = cv2.imread('new/mask_2000.jpg').astype(np.float32)
        #self.score = (255 / len(self.set_list)) * 10
        self.score = 0.02 * 1000
        print("픽셀 부여점수 : " , self.score)
        
    @property
    @timing_decorator
    def set_run(self):
        self.set_list = self.set_list[ 2000  : ]
        for idx , (f ,d ) in tqdm(enumerate(self.set_list) , total=len(self.set_list)):
            door = cv2.imread(f , cv2.IMREAD_COLOR)
            under = cv2.imread(d , cv2.IMREAD_COLOR)

            #door_plane = fisheye2plane.run(door , -40)
            #under_plane = fisheye2plane.run(under , -40)
 
            under_plane , image_plot = self.prediction(under)
            cv2.namedWindow("image_plot" , cv2.WINDOW_NORMAL)
            cv2.imshow("image_plot" , image_plot)
            cv2.waitKey(0)
            
            
            # if idx % 1000 == 0 and idx != 0 or idx == len(self.set_list) - 1:
                # cv2.namedWindow("door" , cv2.WINDOW_NORMAL)
                # cv2.imshow('door' , door_plane)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                
                # os.makedirs('bottom' , exist_ok=True)
                # cv2.imwrite(os.path.join('bottom',f'mask_{idx}.jpg') , self.score_map)


    def prediction(self, img: np.ndarray | torch.Tensor) -> np.array:
        result = self.model(img, classes=[0], verbose=False , device='cuda' , half=False , augment=True , iou=0.4 , conf=0.4 , imgsz=640)
        result = result[0]
        if hasattr(result , 'plot'):
            image_plot = result.plot()
        boxes = []
        scores = []
      
        for box in result.boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = box.conf
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
                scores.append(float(score.detach().cpu().numpy()))

        nmx_boxes = self.apply_nms(boxes, scores)
        img_res = self.draw(nmx_boxes, img)
        return img_res , image_plot

    def draw(self, nmx_boxes: list[list[int]], img: np.ndarray) -> np.ndarray:
        for idx, i in enumerate(nmx_boxes):
            x1, y1, x2, y2 = i
            # 바운딩 박스의 하단 중앙점 계산
            bottom_center_x = min(max(0, (x1 + x2) // 2), 639)  # x축 중앙점
            bottom_y = min(max(0, y2), 639)  # 하단 y좌표 (바운딩 박스의 아래쪽)
            
            # float32 타입으로 계산 후 클리핑
            self.score_map[bottom_y, bottom_center_x] = np.clip(
                self.score_map[bottom_y, bottom_center_x] + self.score,
                0,
                255
            )
        
        # 최종 결과를 uint8로 변환
        return self.score_map.astype(np.uint8)

    def pair(self , cam0_list , cam2_list ):
        buffer = []
        print(cam0_list[-1])
        cam0_dict = {
            Path(cam0_path).parts[1].split('_')[0] + '_' + Path(cam0_path).name: cam0_path
            for cam0_path in cam0_list
        }
        
        for path in cam2_list:
            scen_full_name = os.path.split(path)[0]
            scen_name = scen_full_name.split('/')[1]
            slice_index = str(scen_full_name).split('/')[1].find('_')
            
            name = os.path.basename(path)
            name = scen_name[ : slice_index] + '_' + name

            if str(scen_name[ : slice_index]) in ['2', '3', '3x']:
                index , _ = os.path.splitext(name)
                cam0_path = cam0_dict.get(f"{str(int(index) + 10)}.jpg")
            else:
                cam0_path = cam0_dict.get(name)  

            if cam0_path is not None:
                buffer.append((cam0_path , path))                

        return buffer

    def apply_nms(self, boxes, scores, iou_threshold=0.4):
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.2, nms_threshold=iou_threshold)
        if isinstance(indices, list) and len(indices) > 0:
            return [boxes[i[0]] for i in indices]
        elif isinstance(indices, np.ndarray) and indices.size > 0:
            return [boxes[i] for i in indices.flatten()]
        else:
            return []

if __name__ == "__main__":
    from natsort import natsorted
    cam0_list = []
    cam2_list = []
    for root , dir , files in os.walk('collect'):
        if 'cam2' in root:
            for file in files:
                if file.endswith('jpg'):
                    cam2_list.append(os.path.join(root , file))
        elif 'cam0' in root:
            for file in files:
                if file.endswith("jpg"):
                    cam0_list.append(os.path.join(root , file))
    cam0_list = natsorted(cam0_list)
    cam2_list = natsorted(cam2_list)
    C = YoloDetector(cam0_list=cam0_list , cam2_list=cam2_list)
    C.set_run
