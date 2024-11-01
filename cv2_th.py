import cv2

image = cv2.imread('bottom/mask_10657.jpg' , cv2.IMREAD_GRAYSCALE)
def on_event(pos):
    """
    트랙바 조작에 따른 이벤트 처리 함수
    """
    global image
    img_copy = image.copy()
    
    # _, th = cv2.threshold(img_copy, pos, 255, cv2.THRESH_BINARY)
    # #th[ 285 :, : ] = 0
    # th[ : , 315 : ] = 0
    # # k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # th = cv2.morphologyEx(th , cv2.MORPH_OPEN ,(3,3)  )
    cv2.imshow('t', img_copy)

cv2.namedWindow('t', cv2.WINDOW_NORMAL)

cv2.createTrackbar('th', 't', 0, 255, on_event)

on_event(0)

cv2.waitKey(0)
cv2.destroyAllWindows()
