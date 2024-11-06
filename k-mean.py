import cv2
import numpy as np

def visualize_coordinates():
    # 640x640 검은색 마스크 생성
    mask = np.zeros((640, 640), dtype=np.uint8)
    
    # buffer.txt 파일 읽기
    try:
        with open('buffer.txt', 'r') as f:
            lines = f.readlines()
            
        # 좌표 리스트 생성
        coordinates = []
        for line in lines:
            x, y = map(int, line.strip().split())
            
            # 원의 반경을 설정
            radius = 2
            
            # (x, y) 위치에 반경 5의 원을 그려서 주변 픽셀을 255로 설정
            cv2.circle(mask, (x, y), radius, 255, thickness=-1)  # thickness=-1로 채우기
            
        # 결과 시각화
        cv2.namedWindow('Coordinates Visualization', cv2.WINDOW_NORMAL)
        cv2.imshow('Coordinates Visualization', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return mask
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    visualize_coordinates()