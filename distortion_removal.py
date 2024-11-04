import cv2
import numpy as np

# 이미지 로드
img = cv2.imread('cam2/396.jpg')
h , w = img.shape[:2]

# 왜곡 보정 파라미터 추정 (이 값은 임의 값이며 조정이 필요함)
# OpenCV의 일반적인 보정 예제처럼 간단한 값을 시도하여 보정
K = np.array([[800, 0, img.shape[1] / 2],
              [0, 800, img.shape[0] / 2],
              [0, 0, 1]])  # 추정된 내적 행렬
D = np.array([-0.2, 0.1, 0, 0])  # 왜곡 계수 (임의 값, 조정 가능)

# 보정
new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (img.shape[1], img.shape[0]), 1, (img.shape[1], img.shape[0]))
undistorted_img = cv2.undistort(img, K, D, None, new_K)

# 결과 출력
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.imshow("Original Image", img)
cv2.namedWindow("Undistorted Image", cv2.WINDOW_NORMAL)
cv2.imshow("Undistorted Image", undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
