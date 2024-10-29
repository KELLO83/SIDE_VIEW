import cv2
import fisheye2plane
# 이미지 로드
image = cv2.imread('cam2/0.jpg')
image = fisheye2plane.run(image , - 40)
buffer = []  # 좌표를 저장할 리스트

# 마우스 이벤트 콜백 함수 정의
def mouse_callback(event, x, y, flags, param):
    global buffer, image

    # 마우스 우클릭 시 좌표 저장
    if event == cv2.EVENT_RBUTTONDOWN:
        print(f"좌표: ({x}, {y})")  # 좌표 출력
        buffer.append((x, y))  # 좌표를 buffer에 저장

        # 새로 저장된 좌표까지 직선을 그림
        if len(buffer) > 1:
            cv2.line(image, buffer[-2], buffer[-1], (0, 255, 0), 2)  # 초록색 직선
            print(buffer)
        # 새로 그려진 이미지를 업데이트
        cv2.imshow('t', image)
        

# 윈도우 생성 및 콜백 함수 설정
cv2.namedWindow('t', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('t', mouse_callback)

# 이미지 표시
cv2.imshow('t', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
