import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
matplotlib.use('TkAgg')  # Tkinter 기반 백엔드 사용
plt.figure(figsize=(12,12))
a = np.array([1,2,3])
print(a)
plt.show()

image = np.ones(shape=(640,640),dtype=np.uint8)
cv2.namedWindow("test",cv2.WINDOW_NORMAL)
cv2.imshow('test',image)
cv2.waitKey(0)