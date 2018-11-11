import mmd
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import Utility
import ADistance
import Core
from sklearn.metrics import silhouette_score
import math

srcWeight = 1
margWeight = 1
tarWeight = 1

# margVersion: 1-domain classification, 2-MMD
# tarVersion: 1-Gecco, 2-pseudo with classification, 3-pseudo with silhouette, 4-pseudo with MMDs
tarVersion = 1
margVersion = 2


def domain_differece(src_feature, src_label, classifier, tar_feature):
    if margWeight == 0:
        diff_marg = 0
    else:
        if margVersion == 1:
            diff_marg = domain_classification(src_feature, tar_feature, classifier)
        elif margVersion == 2:
            diff_marg = dis_difference(source=src_feature, target=tar_feature)
        else:
            diff_marg = 0

    if tarWeight == 0:
        tar_err = 0
    else:
        if tarVersion == 1:
            tar_err = pseudo_error_gecco(training_feature=src_feature, training_label=src_label,
                                         testing_feature=tar_feature)
        elif tarVersion == 2:
            tar_err = pseudo_error_classification(training_feature=src_feature, training_label=src_label,
                                                  classifier=classifier, testing_feature=tar_feature)
        elif tarVersion == 3:
            tar_err = pseudo_error_silhouette(training_feature=src_feature, training_label=src_label,
                                              classifier=classifier, testing_feature=tar_feature)
        elif tarVersion == 4:
            tar_err = cond_dis_difference(src_feature=src_feature, src_label=src_label,
                                          classifier=classifier, tar_feature=tar_feature)
        else:
            tar_err = 0

    if srcWeight != 0:
        src_err = nfold_classification_error(features=src_feature, labels=src_label,
                                             classifier=classifier, n_fold=3)
    else:
        src_err = 0

    return src_err, diff_marg, tar_err


# fitness function with 3 components: marginal, source error, classification error
def fitness_function(src_feature, src_label, tar_feature, classifier):
    src_err, diff_marg, tar_err = domain_differece(src_feature=src_feature, src_label=src_label,
                                                   classifier=classifier, tar_feature=tar_feature)

    return srcWeight * src_err + tarWeight * tar_err + margWeight * diff_marg, src_err, diff_marg, tar_err


# Use training dataset to classify the testing dataset -> pseudo-testing
# use pseudo-testing to classify the training -> pseudo-training
# get the accuracy
def pseudo_error_classification(training_feature, training_label, classifier, testing_feature):
    classifier.fit(training_feature, training_label)
    testing_pseudo = classifier.predict(testing_feature)

    return classification_error(training_feature=testing_feature, training_label=testing_pseudo,
                                testing_feature=training_feature, testing_label=training_label,
                                classifier=classifier)
    # return nfold_classification_error(testing_feature, testing_pseudo, classifier, n_fold=3)


# Use training dataset to classify the testing dataset -> pseudo-testing
# use silhouette measure on the pseudo-testing to
# the larger the silhouette, the better -> return -silhoutte
def pseudo_error_silhouette(training_feature, training_label, classifier, testing_feature):
    classifier.fit(training_feature, training_label)
    testing_pseudo = classifier.predict(testing_feature)
    return -silhouette_score(testing_feature, testing_pseudo, random_state=1617)


# Find the closest insance of an instance on the target domain
# Fnd the closest source instance of the two instances on the source domain
# It is expected that the two source instance should have the same class
def pseudo_error_gecco(training_feature, training_label, testing_feature):
    classifier = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
    classifier.fit(training_feature, training_label)
    # if classifier is 1-KNN then this process is similar to
    # finding the closest src instance for each target instance
    # then assign the class label, it is expected that the two closest
    # target instance should be in the same class
    testing_pseudo = classifier.predict(testing_feature)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='brute').fit(testing_feature)
    _, indices = nbrs.kneighbors(testing_feature)
    count = 0.0
    for pair in indices:
        current, neighbor = pair
        if testing_pseudo[current] == testing_pseudo[neighbor]:
            count += 1.0
    return 1-count/len(testing_feature)


# calculate the  distribution difference
def dis_difference(source, target):
    union = np.concatenate((source, target))
    sigma = Utility.medianDistance(union)
    return mmd.my_rbf_mmd2(source, target, sigma)


# calculate the domain difference(need to maximize its error or minimize
# the classification accuracy), since the overall fitness function is to minimize
# so this method will return the accuracy
def domain_classification(src_features, tar_features, classifier):
    domain_features = np.concatenate((src_features, tar_features))
    domain_labels = np.ones(len(src_features) + len(tar_features), dtype=int)
    domain_labels[len(src_features):len(src_features) + len(tar_features)] = 0
    return 1 - nfold_classification_error(domain_features, domain_labels, classifier, n_fold=3)


# calculate the conditional distribution difference:
def cond_dis_difference(src_feature, src_label, classifier, tar_feature):
    # if label is none, we have to estimate the label using the current src label
    classifier.fit(src_feature, src_label)
    tar_label = classifier.predict(tar_feature)
    unique_class = np.unique(tar_label)
    diff = 0
    for uc in unique_class:
        src_feature_uc = src_feature[src_label == uc]
        tar_feature_uc = tar_feature[tar_label == uc]
        diff += dis_difference(src_feature_uc, tar_feature_uc)
    return diff/len(unique_class)


def classification_error(training_feature, training_label, classifier, testing_feature, testing_label):
    classifier.fit(training_feature, training_label)
    error = 1.0 - classifier.score(testing_feature, testing_label)
    return error


def nfold_classification_error(features, labels, classifier, n_fold):
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


def set_weight(src_feature, src_label, tar_feature):
    global tarWeight, margWeight, srcWeight
    Core.classifier.fit(src_feature, src_label)
    tar_label = Core.classifier.predict(tar_feature)

    marg_diff = ADistance.proxy_a_distance(src_feature, tar_feature)

    cond_diff = 0
    unique_class = np.unique(tar_label)
    for uc in unique_class:
        src_feature_c = src_feature[src_label == uc]
        tar_feature_c = tar_feature[tar_label == uc]
        if len(src_feature_c) > 1 and len(tar_feature_c) > 1:
            cond_diff += ADistance.proxy_a_distance(src_feature_c, tar_feature_c)

    cond_diff = cond_diff/len(unique_class)
    tarWeight = cond_diff / (cond_diff + marg_diff)
    margWeight = marg_diff/(cond_diff+marg_diff)
    srcWeight = 0.0
