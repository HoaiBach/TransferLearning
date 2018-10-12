import numpy as np


def buildNewFeatures(feature, funcs):
    feature_new = np.empty((len(feature), 0))

    for func in funcs:
        value_new = np.array([func(*row) for row in feature])
        value_new = np.reshape(value_new, (len(feature), 1))
        feature_new = np.concatenate((feature_new, value_new), axis=1)

    return feature_new
