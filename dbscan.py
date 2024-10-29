import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import matplotlib
from collections import Counter

matplotlib.use("Tkagg")
image = cv2.imread("t2.jpg", cv2.IMREAD_GRAYSCALE)
height, width = image.shape

X = []
for y in range(height):
    for x in range(width):
        pixel_value = image[y, x]
        if x > 315:
            pixel_value = 0
        X.append([x, y, pixel_value])  

sample_image = np.zeros_like(image , dtype=np.uint8)
for x,y,value in X:
    sample_image[y][x] = value

        

db = DBSCAN(eps=10, min_samples=40).fit(X)
labels = db.labels_
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"군집의 갯수: {num_clusters}")
label_counts = Counter(labels)

most_common_label = max(label_counts, key=lambda label: label_counts[label] if label != -1 else -1)
most_common_count = label_counts[most_common_label]

print(f"가장 많은 군집 라벨: {most_common_label}, 데이터 포인트 수: {most_common_count}")

colors = [
    (255, 0, 0),    # 빨강 (Red)
    (255, 165, 0),  # 주황 (Orange)
    (0, 255, 0),    # 초록 (Green)
    (0, 0, 255),    # 파랑 (Blue)
    (0, 0, 128),    # 남색 (Indigo)
    (75, 0, 130) ,   # 보라 (Violet)
    (255, 255, 0),  # 노랑 (Yellow)
]
for idx , i in enumerate(colors):
    r , g , b = i
    colors[idx] = (b , g, r )
    
output_image = np.zeros((height, width, 3), dtype=np.uint8)

for (x, y, pixel_value), label in zip(X, labels):
    if label == -1:
        color = (255,255,255)
        
    if label == 0 :
        color = (0, 0, 0)

    else:
        color = colors[ label % len(colors)]
    output_image[y][x] = color
    
    
mask = np.zeros_like(labels, dtype=bool)  

for idx, label in enumerate(labels):
    if label != -1 and label != 0:
        mask[idx] = True
    else:
        mask[idx] = False
X = np.array(X)



x_train = np.zeros_like(labels , dtype=np.uint8)
y_train = np.zeros_like(labels , dtype=np.uint8)
for idx , (x , y , value ) in enumerate(X):
    if mask[idx]:
        x_train[idx] = value
    else:
        x_train[idx] = 0


for idx , value in enumerate(labels):
    if mask[idx]:
        labels[idx] = value
    else:
        labels[idx] = -1


cv2.imshow("DBSCAN Clustering Result", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


