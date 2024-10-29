import cv2
import numpy as np
import fisheye2plane
# 빈 이미지 생성 (512x512 크기, 검정 배경)
image = cv2.imread('cam2/0.jpg')
image = fisheye2plane.run(image , -40)
# 점 좌표 리스트 (각각 (x, y) 형식)
points = [(241, 57), (231, 121), (213, 185), (194, 262), (181, 280), (185, 431), (130, 566), (126, 639)]

# 점들을 순차적으로 연결하며 선 그리기
for i in range(len(points) - 1):
    cv2.line(image, points[i], points[i + 1], (0, 255, 0), 2)  # 초록색 선, 두께 2


# 이미지 표시
cv2.imshow('Connected Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
