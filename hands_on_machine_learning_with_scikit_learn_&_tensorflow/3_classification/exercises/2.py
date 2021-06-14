import numpy as np
from scipy.ndimage.interpolation import shift
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'], mnist['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])
    
# X_train_augmented = [image for image in X_train]
# y_train_augmented = [label for label in y_train]
X_train_augmented = []
y_train_augmented = []

for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    for image, label in zip(X_train, y_train):
        X_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(label)

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = X_train_augmented[shuffle_idx]
y_train_augmented = y_train_augmented[shuffle_idx]

knn = KNeighborsClassifier(n_neighbors = 5, weights = 'distance')
knn.fit(X_train_augmented, y_train_augmented)
print(knn.score(X_test, y_test))
'''
0.9749714285714286
'''
