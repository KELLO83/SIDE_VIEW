import cv2
import numpy as np

# 이미지의 크기 설정 (예: 640x480)
w, h = 640, 480

# 빈 검은색 이미지 생성
mask = np.zeros((h, w), dtype=np.uint8)

# 주어진 좌표 리스트 (외곽과 내부)
under2_out_points = [(227, 254), (225, 269), (220, 290), (214, 318), (208, 347), (204, 382), (197, 409), (192, 442), (181, 488), (176, 513), (163, 573), (154, 614), (150, 639)]
under2_in_points = [(306, 195), (309, 217), (315, 250), (326, 257), (332, 271), (338, 289), (341, 312), (346, 337), (351, 363), (351, 376), (349, 399), (345, 420), (335, 439), (325, 457), (320, 474), (320, 599)]

# 1. 외곽 마스크 생성
cv2.fillPoly(mask, [np.array(under2_out_points, dtype=np.int32)], 255)

# 2. 내부 영역을 검은색으로 덮어 중간 영역을 만듦
cv2.fillPoly(mask, [np.array(under2_in_points, dtype=np.int32)], 0)

# 결과 확인
cv2.imshow("Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
