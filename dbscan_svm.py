import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.svm import SVC
from collections import Counter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter

matplotlib.use('Tkagg')
image = cv2.imread("t2.jpg", cv2.IMREAD_GRAYSCALE)
height, width = image.shape

X = []
for y in range(height):
    for x in range(width):
        pixel_value = image[y, x]
        if x > 315:
            pixel_value = 0
        X.append([x, y, pixel_value])  # (x, y, 밝기 값)

X = np.array(X)

db = DBSCAN(eps=10, min_samples=40).fit(X)
labels = db.labels_
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print("군집의 갯수 : ",num_clusters)
label_counts = Counter(labels)
print(label_counts)
most_common_label = max(label_counts , key= lambda label : label_counts[label] if label != -1 else -1)
common_label_count = label_counts[most_common_label]
print(" 가장많은 군집 {} 데이터수 {} ".format(most_common_label , common_label_count))

valid_indices = np.where((labels != -1) & (labels != 0))[0]
x_train = X[valid_indices].astype(np.float32)
y_train = labels[valid_indices].astype(np.int32)

pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train)
svm_pca = SVC(kernel='rbf', gamma='auto' , class_weight='balanced')
svm_pca.fit(x_train_pca, y_train)

x_min, x_max = x_train_pca[:, 0].min() - 1, x_train_pca[:, 0].max() + 1
y_min, y_max = x_train_pca[:, 1].min() - 1, x_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))


Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(x_train_pca[:, 0], x_train_pca[:, 1], c=y_train, edgecolor='k')
plt.title('SVM 결정 경계 (PCA 적용)')
plt.savefig('res.png')
plt.show()