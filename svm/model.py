import math
import numpy as np
# from sklearn.svm import SVR as Model
from sklearn.svm import SVC as Model

from import_data import *


def augment_features(features, window_size=1):
    assert window_size & 1, "Window size must be odd"
    num_sites = features.shape[1]

    return (features[:, i - window_size//2 : i + 1 + window_size//2] for i in range(window_size//2, num_sites - window_size//2))


def augment_labels(labels, window_size=1):
    assert window_size & 1, "Window size must be odd"
    num_sites = features.shape[1]

    return (labels[:, i] for i in range(window_size//2, num_sites - window_size//2))


w = 11
features = get_genotypes()
labels = get_ancestry()

labels[:,0:1000].shape

print(features.shape)
print(labels.shape)

models = []
for features, labels in zip(augment_features(features, window_size=w), augment_labels(labels, window_size=w)):
    print("Features shape: %s \tLabels shape: %s" % (str(features.shape), str(labels.shape)))
    model = Model(kernel="linear")
    model.fit(features, labels)
    predictions = model.predict(features)
    print(model.score(features, labels))
    models.append(model)

