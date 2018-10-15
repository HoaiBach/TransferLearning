import mmd
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import Utility
import proxy_a_distance as ADistance

srcWeight = 0.1
margWeight= 0.0
condWeight = 0.9

condVersion = 1


def domainDifferece(src_feature, src_label, classifier, tarU_feature, tarU_soft_label = None, tarL_feature=None, tarL_label=None):

    if margWeight==0:
        diff_marg=0
    else:
        diff_marg = distributionDifference(source=src_feature, target=tarU_feature)

    if condWeight == 0:
        tar_err = 0
    else:
        if tarL_label is None:
            if condVersion == 1:
                tar_err = pseudoErrorGecco(src_feature, src_label, tarU_feature)
            elif condVersion == 2:
                tar_err = pseudoError(training_feature=src_feature, training_label=src_label,
                                      classifier=classifier,
                                      testing_feature=tarU_feature)
            else:
                tar_err = conditionalDistributionDifference(src_feature=src_feature, src_label=src_label,
                                                            classifier=classifier,
                                                            tar_feature=tarU_feature, tar_label=tarU_soft_label)
        else:
            tar_err = classificationError(training_feature=src_feature, training_label=src_label,
                                          classifier=classifier,
                                          testing_feature=tarL_feature, testing_label=tarL_label)

    if srcWeight != 0:
        src_err = nFoldClassificationError(features=src_feature, labels=src_label,
                                           classifier=classifier, n_fold=3)
    else:
        src_err = 0

    return src_err, diff_marg, tar_err


# fitness function with 3 components: marginal, source error, classification error
def fitnessFunction(src_feature, src_label, tarU_feature, tarU_soft_label, classifier, tarL_feature=None, tarL_label=None):
    src_err, diff_marg, tar_err = domainDifferece(src_feature=src_feature, src_label=src_label,
                                                  classifier=classifier,
                                                  tarU_feature=tarU_feature, tarU_soft_label=tarU_soft_label,
                                                  tarL_feature=tarL_feature, tarL_label=tarL_label)
    return srcWeight * src_err + condWeight * tar_err + margWeight * diff_marg


# Use training dataset to classify the testing dataset -> pseudo-testing
# use pseudo-testing to classify the training -> pseudo-training
# get the accuracy
def pseudoError(training_feature, training_label, classifier, testing_feature):

    classifier.fit(training_feature, training_label)
    testing_pseudo = classifier.predict(testing_feature)

    # now treat testing as training and training as test
    return classificationError(training_feature=testing_feature, training_label=testing_pseudo,
                               classifier=classifier,
                               testing_feature=training_feature, testing_label=training_label)


# Find the closest insance of an instance on the target domain
# Fnd the closest source instance of the two instances on the source domain
# It is expected that the two source instance should have the same class
def pseudoErrorGecco(training_feature, training_label, testing_feature):
    classifier = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
    classifier.fit(training_feature, training_label)
    # if classifier is 1-KNN then this process is similar to
    # finding the closest src insatnce for each target instance
    # then assign the class label, it is expected that the two closest
    # target instance should be in the same class
    testing_pseudo = classifier.predict(testing_feature)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='brute').fit(testing_feature)
    _ , indices = nbrs.kneighbors(testing_feature)
    count = 0.0
    for pair in indices:
        current, neighbor = pair
        if testing_pseudo[current] == testing_pseudo[neighbor]:
            count += 1.0
    return count/len(testing_feature)


# calculate the  distribution difference
def distributionDifference(source, target):
    union = np.concatenate((source, target))
    sigma = Utility.medianDistance(union)
    return mmd.my_rbf_mmd2(source, target, sigma)


# calculate the conditional distribution :
def conditionalDistributionDifference(src_feature, src_label, classifier, tar_feature, tar_label):
    # if label is none, we have to estimate the label using the current src label
    if tar_label is None:
        classifier.fit(src_feature, src_label)
        tar_label = classifier.predict(tar_feature)

    unique_class = np.unique(tar_label)
    diff = 0
    for uc in unique_class:
        src_feature_uc = src_feature[src_label == uc]
        tar_feature_uc = tar_feature[tar_label == uc]
        diff += distributionDifference(src_feature_uc, tar_feature_uc)
    return diff/len(unique_class)


def classificationError(training_feature, training_label, classifier, testing_feature, testing_label):
    classifier.fit(training_feature, training_label)
    error = 1.0 - classifier.score(testing_feature, testing_label)
    return error


def nFoldClassificationError(features, labels, classifier, n_fold):
    error = 0
    skf = StratifiedKFold(n_splits=n_fold, random_state=1617)
    for trainIndex, testIndex in skf.split(features, labels):
        train_feature, test_feature = features[trainIndex], features[testIndex]
        train_label, test_label = labels[trainIndex], labels[testIndex]
        classifier.fit(train_feature, train_label)
        fold_error = 1.0-classifier.score(test_feature, test_label)
        error += fold_error
    error = error/n_fold
    return error


def setWeight(src_feature, src_label, tarU_feature, tarU_label):
    global condWeight, margWeight
    margDiff = ADistance.proxy_a_distance(src_feature, tarU_feature)

    condDiff = 0
    uniqueClass = np.unique(tarU_label)
    for uc in uniqueClass:
        src_feature_c = src_feature[src_label==uc]
        tarU_feature_c = tarU_feature[tarU_label==uc]
        if len(src_feature_c) > 1 and len(tarU_feature_c) > 1:
            condDiff += ADistance.proxy_a_distance(src_feature_c, tarU_feature_c)

    condDiff = condDiff/len(uniqueClass)
    condWeight = condDiff/(condDiff+margDiff)
    margWeight = margDiff/(condDiff+margDiff)

