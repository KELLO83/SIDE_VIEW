import cv2
import mediapipe as mp

# MediaPipe 솔루션 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.1, min_tracking_confidence=0.1 , model_complexity=2)

mp_drawing = mp.solutions.drawing_utils

# 웹캠 캡처 시작

image = cv2.imread('under_plane/under_513.jpg')

image = cv2.cvtColor(image  , cv2.COLOR_BGR2RGB)

# 포즈 추정 수행
results = pose.process(image)

# RGB 이미지를 다시 BGR로 변환 (OpenCV에서 사용하기 위함)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# 포즈 랜드마크를 그리기
if results.pose_landmarks:
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# 화면에 출력
cv2.imshow('MediaPipe Pose', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
