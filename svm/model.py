import math
import numpy as np
from sklearn.svm import SVC

from import_data import *


def augment_features(features, windows=1):
    N = len(features[0])
    target = math.ceil(N / windows)

    print("Target", target)

    return (features[:, i * target : (i+1) * target] for i in range(windows))


def augment_labels(labels, windows=1):
    N = len(labels[0])
    target = math.ceil(N / windows)

    return (
        np.round(
            np.mean(
                labels[:, i * target : (i+1) * target]
                , axis=1
            )
        ) for i in range(windows)
    )

w = 5
features = get_genotypes()
labels = get_ancestry()

labels[:,0:1000].shape

print(labels.shape)

svcs = [SVC(kernel='linear') for _ in range(w)]

for i, features, labels in zip(range(w), augment_features(features, windows=w), augment_labels(labels, windows=w)):
    print("%i: Features shape: %s \tLabels shape: %s" % (i, str(features.shape), str(labels.shape)))
    svcs[i].fit(features, labels)
    predictions = svcs[i].predict(features)
    print(svcs[i].score(features, labels))
