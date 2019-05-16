import math
import numpy as np
# from sklearn.svm import SVR as Model
from sklearn.svm import SVC as Model

from import_data import *


def augment_features(features, window_size=1):
    assert window_size & 1, "Window size must be odd"
    # print(features.shape)
    num_sites = features.shape[1]

    # return np.hstack( a[i:1+i-window_size or None:1] for i in range(0,window_size))

    aug = (features[:, i - window_size//2 : i + 1 + window_size//2] for i in range(window_size//2, num_sites - window_size//2))

    return aug


def augment_labels(labels, window_size=1):
    assert window_size & 1, "Window size must be odd"
    num_sites = labels.shape[1]

    return (labels[:, i] for i in range(window_size//2, num_sites - window_size//2))


w = 11
# features = get_genotypes()
# labels = get_ancestry()

train_features, test_features = get_genotypes()
train_labels, test_labels = get_ancestry()

# labels[:,0:1000].shape
#
# train_labels[:,0:1000].shape
# test_labels[:,0:1000].shape

# print(features.shape)
# print(labels.shape)

print("FEATURES: TRAIN :" , train_features.shape,  "TEST :", test_features.shape )
print("LABELS: TRAIN :" , train_labels.shape,  "TEST :", test_labels.shape )

models = []

for features, labels in zip(augment_features(train_features, window_size=w), augment_labels(train_labels, window_size=w)):
    # print("Features shape: %s \tLabels shape: %s" % (str(features.shape), str(labels.shape)))
    # print(labels)
    if labels.shape[0] <= 1:
        pass
    model = Model(kernel="linear")
    model.fit(features, labels)
    models.append(model)

print("Done training")
scores = []

i = 0
for features, labels in zip(augment_features(test_features, window_size=w), augment_labels(test_labels, window_size=w)):
    # print("Features shape: %s \tLabels shape: %s" % (str(features.shape), str(labels.shape)))
    model = models[i]
    predictions = model.predict(features)
    # print("Score:", model.score(features, labels))
    scores.append(model.score(features,labels))
    i+=1

print("Average score:", np.average(scores))
