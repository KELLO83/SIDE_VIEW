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
import palne_remap
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
    def __init__( self , SCEN = '3x'):
        self.model = YOLO('yolo11x.pt').to('cuda')
        self.SCEN = SCEN
        self.cam0_list = natsorted(glob(os.path.join(f'collect/scne{str(SCEN)}_0.5origin/cam0','*.jpg')))
        self.cam2_list = natsorted(glob(os.path.join(f'collect/scne{str(SCEN)}_0.5origin/cam2','*.jpg')))
        self.cam4_list = natsorted(glob(os.path.join(f'collect/scne{str(SCEN)}_0.5origin/cam4','*.jpg')))
        self.set_list = self.pair(self.cam0_list , self.cam2_list , self.cam4_list)

    def pair(self , cam0_list , cam2_list , cam4_list , flag = True):
        buffer = []
        cam0_dict = {os.path.basename(cam0_path): cam0_path for cam0_path in cam0_list}
        cam4_dict = {os.path.basename(cam4_path) : cam4_path for cam4_path in cam4_list}
        
        for path in cam2_list:
            name = os.path.basename(path)
            
            if str(self.SCEN) in ['2', '3', '3x']:
                index , _ = os.path.splitext(name)
                cam0_path = cam0_dict.get(f"{str(int(index) + 10)}.jpg")
            else:
                cam0_path = cam0_dict.get(name)  
                
            if flag:
                if cam0_path is not None:
                    buffer.append((cam0_path, path , path))
            else:
                cam4_path = cam4_dict.get(name)
                if cam0_path is not None and cam4_path is not None:
                    buffer.append((cam0_path, path , cam4_path))
                    
                    
        #buffer = buffer[500 : ]
        return buffer
    def nmx_box_to_cv2_loc(self , boxes):
        
        
        x1 , y1 , w, h = boxes
        x2 = x1 + w
        y2 = y1 + h

        return [x1 , y1 , x2 , y2]
        
    def apply_nms(self, boxes, scores, iou_threshold=0.4):
            indices = cv2.dnn.NMSBoxes(boxes, scores, 0.2, iou_threshold)
            #print(indices)
            if isinstance(indices, list) and len(indices) > 0:
                return [boxes[i[0]] for i in indices]
            
            elif isinstance(indices, np.ndarray) and indices.size > 0:
                return [boxes[i] for i in indices.flatten()]
            
            else:
                return []


    @property
    @timing_decorator
    def set_run(self):
        self.set_list = self.set_list[ 300 : ]
        for idx , (f ,d ,b) in tqdm(enumerate(self.set_list) , total=len(self.set_list)):
            door = cv2.imread(f , cv2.IMREAD_COLOR)
            under = cv2.imread(d , cv2.IMREAD_COLOR)
            cam4 = cv2.imread(b , cv2.IMREAD_COLOR)


            
            door_plane = fisheye2plane.run(door , -40)
            under_plane = fisheye2plane.run(under , -40)
            #under2_plane = fisheye2plane.run(under , 40 , flag= False)
            #cam4_plane = fisheye2plane.run(cam4 , - 40 )
            
            under_prediction = self.prediction(under_plane ,flag = 1)
            #under2_prediction = self.prediction(under2_plane)
            door_predicition = self.prediction(door_plane , flag = 2)
            #cam4_prediction = self.prediction(cam4_plane)
            
            

            

            cv2.namedWindow("under" , cv2.WINDOW_NORMAL)
            #cv2.namedWindow('under2',cv2.WINDOW_NORMAL)
            cv2.namedWindow('door' , cv2.WINDOW_NORMAL)
            #cv2.namedWindow("cam4" , cv2.WINDOW_NORMAL)
                      

            cv2.imshow('under' , under_prediction)
            #cv2.imshow('under2',under2_prediction)
            cv2.imshow('door' , door_predicition)
            #cv2.imshow("cam4",cam4_prediction)
            
            cv2.waitKey(0)
    

    
    @property
    @timing_decorator
    def run(self):
        test_list = natsorted(glob(os.path.join(f'plane/cam2','*.jpg')))
        test_list = test_list[ 100 : ]
        for idx , f in tqdm(enumerate(test_list ), total=len(test_list)):
            frame = cv2.imread(f , cv2.IMREAD_COLOR)
            result = self.prediction(frame)
            
            cv2.namedWindow('res',cv2.WINDOW_NORMAL)
            cv2.imshow('res',result)
            cv2.waitKey(0)
            

    def prediction(self ,img: np.ndarray | torch.Tensor , flag = False) -> np.array:
        
 
        result = self.model(img , classes = 0)
        
        if isinstance(img , torch.Tensor) :
            if img.is_cuda:
                img = img.cpu()
                print("GPU -> CPU")
                img = img.squeeze(0).permute(1,2,0).numpy()
                img = np.clip(img * 255.0 , 0 , 255).astype(np.uint8)
            

        boxes = []
        scores = []
        for res in result:
            for box in res.boxes:
                if int(box.cls) == 0:
                    x1 , y1 , w , h = box.xyxy[0].tolist()
                    score = box.conf
                    boxes.append([int(x1), int(y1), int(w - x1), int(h - y1)])
                    scores.append(float(score.detach().cpu().numpy()))

        nmx_boxes = self.apply_nms(boxes , scores)
        nmx_boxes = list(map(self.nmx_box_to_cv2_loc , boxes))
        img_res = self.draw(nmx_boxes , img , flag)
        return img_res
    
    def draw(self , nmx_boxes : list[int] , img  , flag = False) -> np.array:
        h , w, _ = img.shape
        for idx , i in enumerate(nmx_boxes):
            x1 , y1 , x2 , y2 = i
            center_x , center_y = (x1 + x2) // 2 , (y1 + y2) // 2
            cv2.rectangle(img , (x1 , y1) ,(x2 , y2) , (255,255,0),thickness=2)
            cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)  
            cv2.putText(img, f'({center_x}, {center_y})', 
                        (center_x, center_y), 
                        cv2.FONT_HERSHEY_DUPLEX, 
                        0.3, 
                        (255, 255, 255), 
                        1)
        
        
        
        if flag == 1 : 
            x_line = [250 , 175]
            y_line = [500 , 260 , 150 , 90]

            for i in y_line:
                cv2.line(img , (0,i) , (w-1 , i) , (0,255,0), 2)
            
            for i in x_line:
                cv2.line(img , (i , 0) , (i , h-1) , (255,0,0,) , 2)
                
        if flag == 2 :
            y_line = [265]
            x_line = [93, 240 , 450 ]
            for i in y_line:
                cv2.line(img , (0,i) , (w-1 , i) , (0,255,0), 2)
            for i in x_line:
                cv2.line(img , (i , 0) , (i , h-1) , (255,0,0,) , 2)
        return img
    

    



if __name__ == "__main__":

    C = YoloDetector()
    C.set_run