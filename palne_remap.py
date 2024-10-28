import cv2
import numpy as np


# 피쉬아이 왜곡 보정 매개변수 (초기값 설정)

def remap(img):
    DIM = img.shape[:2][::-1]  # 이미지 크기
    K = np.array([[DIM[0], 0, DIM[0] / 2], 
                [0, DIM[1], DIM[1] / 2], 
                [0, 0, 1]])  # 카메라 행렬 초기 추정치
    D = np.zeros((4, 1))  # 왜곡 계수(초기 값은 0)

    # 왜곡 보정 매핑 생성
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return undistorted_img
    
#undistorted_img = cv2.rotate(undistorted_img , cv2.ROTATE_90_COUNTERCLOCKWISE)
# 결과 이미지 저장 및 출력

# 이미지 로드

if __name__ == '__main__':
    img = cv2.imread("cam2/600.jpg")
    undistorted_img = remap(img)
    cv2.imwrite("undistorted.jpg", undistorted_img)
    cv2.namedWindow("Undistorted Image",cv2.WINDOW_NORMAL)
    cv2.imshow("Undistorted Image", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
