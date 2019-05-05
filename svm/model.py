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

total_features = get_genotypes()
total_labels = get_ancestry()

print(total_features.shape, total_labels.shape)

#train test split
features, test_features = total_features[10:100], total_features[0:10]
labels, test_labels = total_labels[10:100], total_labels[0:10]

# labels[:,0:1000].shape


count = 0

# models = []
model = Model(kernel="linear")
for features, labels in zip(augment_features(features, window_size=w), augment_labels(labels, window_size=w)):
    #print("Features shape: %s \tLabels shape: %s" % (str(features.shape), str(labels.shape)))
    model.fit(features, labels)
    predictions = model.predict(features)
    if count%1000 == 0:
        print("===================================")
        for t_features, t_labels in zip(augment_features(test_features, window_size=w), augment_labels(test_labels, window_size=w)):
            print(model.score(t_features, t_labels))
    #models.append(model)
    count += 1
