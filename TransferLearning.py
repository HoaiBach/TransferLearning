'''
Created on 21/09/2018

@author: nguyenhoai2
'''

import numpy as np
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics.pairwise import pairwise_distances as PW
import Utility as utility
import math
import mmd

#classifier = GaussianNB()
classifier = KNeighborsClassifier(n_neighbors=1,algorithm = 'brute')
threshold = 0.6

src_data = np.genfromtxt("data/Source",delimiter=",")
tar_data = np.genfromtxt("data/TargetU",delimiter=",")
no_features = len(src_data[0])-1

src_feature = src_data[:,0:no_features]
src_label = src_data[:,no_features:no_features+1]
src_label = np.reshape(src_label,len(src_label))
no_classes = len(np.unique(src_label))

tar_feature = tar_data[:,0:no_features]
tar_label = tar_data[:,no_features:no_features+1]
tar_label = np.reshape(tar_label,len(tar_label))

src_feature,tar_feature = utility.standarizeSrcTar(src_feature, tar_feature)


def getReducedSet(features,labels):
    nc = len(np.unique(labels))
    ni = len(features)
    step = ni/nc
    indices = []
    for i in range(nc):
        indices.append(i*step)
        indices.append(i*step+1)
        indices.append(i*step+2)
    return np.asarray([features[i] for i in indices]),np.asarray([labels[i] for i in indices])

#src_feature,src_label = getReducedSet(src_feature, src_label)
#tar_feature,tar_label = getReducedSet(tar_feature, tar_label)

def fitnessFunctionTest(trees, toolbox):
    src_fea_new,tar_fea_new = buildNewFeatures(trees,toolbox)
    classifier.fit(src_fea_new,src_label)
    error = 1.0-classifier.score(tar_fea_new,tar_label)
    fitness = error
    return fitness

def geccoFitnessFunction(trees, toolbox):
    src_fea_new,tar_fea_new = buildNewFeatures(trees,toolbox)
    src_fea_new,tar_fea_new = utility.standarizeSrcTar(src_fea_new, tar_fea_new)
    
    #marginal distance
    margin = distanceDistribution(src_fea_new, tar_fea_new)
    
    #conditional distribution
    classifier.fit(src_fea_new, src_label)
    tar_pseudo = classifier.predict(tar_fea_new)
    tar_pseudo = np.asarray(tar_pseudo)
    nbrs = NearestNeighbors(n_neighbors=2,algorithm="brute").fit(tar_fea_new)   
    _,indices = nbrs.kneighbors(tar_fea_new)
    correct=0
    for pair in indices:
        #the two closest instances are expected to have the same 
        #pseudo labels.
        if tar_pseudo[pair[0]] == tar_pseudo[pair[1]]:
            correct = correct+1.0
    cond = 1- correct/len(tar_pseudo)
    
    error = aveErrorFolds(src_fea_new, src_label, 3) 
    
    return margin+error+cond,

def fitnessFunctionTestSelection(position):
    indices = [index for index,entry in enumerate(position) if entry>threshold]
    src_fea_new = src_feature[:,indices]
    tar_fea_new = tar_feature[:,indices]
    
    classifier.fit(src_fea_new,src_label)
    error = 1.0-classifier.score(tar_fea_new,tar_label)
    fitness = error
    return fitness,

def geccoFitnessFunctionSelection(position):
    indices = [index for index,entry in enumerate(position) if entry>threshold]
    src_fea_new = src_feature[:,indices]
    tar_fea_new = tar_feature[:,indices]
    
    #marginal distance
    margin = distanceDistribution(src_fea_new, tar_fea_new)
    
    #conditional distribution
    classifier.fit(src_fea_new, src_label)
    tar_pseudo = classifier.predict(tar_fea_new)
    tar_pseudo = np.asarray(tar_pseudo)
    nbrs = NearestNeighbors(n_neighbors=2,algorithm="brute").fit(tar_fea_new)   
    _,indices = nbrs.kneighbors(tar_fea_new)
    correct=0
    for pair in indices:
        #the two closest instances are expected to have the same 
        #pseudo labels.
        if tar_pseudo[pair[0]] == tar_pseudo[pair[1]]:
            correct = correct+1.0
    cond = 1- correct/len(tar_pseudo)
    
    error = aveErrorFolds(src_fea_new, src_label, 3) 
    
    return 0.9*margin+0.0*error+0.1*cond,


def fitnessFunction(trees, toolbox):
    src_fea_new,tar_fea_new = buildNewFeatures(trees,toolbox)
    src_fea_new,tar_fea_new = utility.standarizeSrcTar(src_fea_new, tar_fea_new)
    
    # measure the difference in terms of marginal distributions
    diff_marg = mmd.normal_mmd2(src_fea_new,tar_fea_new)

    classifier.fit(src_fea_new, src_label)
    tar_pseudo = classifier.predict(tar_fea_new)
    tar_pseudo = np.asarray(tar_pseudo)
    uniq_class = np.unique(tar_label)
    diff_con = 0
    for uc in uniq_class:
        src_fea_uc = src_fea_new[src_label == uc]
        tar_fea_uc = tar_fea_new[tar_label == uc]
        diff_con += mmd.normal_mmd2(src_fea_uc,tar_fea_uc)
    diff_con = diff_con/len(uniq_class)

    error = aveErrorFolds(src_fea_new,src_label,3)

    fitness = 0.0*diff_marg + 0.0*diff_con + error
    return fitness

def aveErrorFolds(features,labels,nfolds):
    error = 0
    skf = StratifiedKFold(n_splits=nfolds,random_state=1617)
    for trainIndex,testIndex in skf.split(features, labels):
        train_feature,test_feature = features[trainIndex],features[testIndex]
        train_label,test_label     = labels[trainIndex], labels[testIndex]
        classifier.fit(train_feature,train_label)
        fold_error = 1.0-classifier.score(test_feature, test_label)
        error += fold_error
    error = error/nfolds
    return error

def distanceDistribution(src,tar):
    s2s = np.asarray(PW(src,src,metric="euclidean"))
    t2t = np.asarray(PW(tar,tar,metric="euclidean"))
    s2t = np.asarray(PW(src,tar,metric="euclidean"))
    
    #find the gamma for kernel
    median = np.median(s2t)
    sigma = math.sqrt(median/2)
    
    s2t = np.vectorize(math.exp)(-s2t/(2*sigma*sigma))
    s2s = np.vectorize(math.exp)(-s2s/(2*sigma*sigma))
    t2t = np.vectorize(math.exp)(-t2t/(2*sigma*sigma))
    
    return math.sqrt(math.fabs(np.mean(s2s)+np.mean(t2t)-2*np.mean(s2t)))

def accuracyTarget(trees, toolbox):
    src_fea_new,tar_fea_new = buildNewFeatures(trees,toolbox)
    classifier.fit(src_fea_new,src_label)
    return classifier.score(tar_fea_new,tar_label)

def accuracyTargetSelection(position):
    indices = [index for index,entry in enumerate(position) if entry>threshold]
    src_fea_new = src_feature[:,indices]
    tar_fea_new = tar_feature[:,indices]
    classifier.fit(src_fea_new,src_label)
    return classifier.score(tar_fea_new,tar_label)

def accuracyTargetNo():
    classifier.fit(src_feature,src_label)
    return classifier.score(tar_feature,tar_label)

def accuracySource(trees, toolbox):
    src_fea_new,_ = buildNewFeatures(trees,toolbox)
    classifier.fit(src_fea_new,src_label)
    return classifier.score(src_fea_new,src_label)

def accuracySourceNo():
    classifier.fit(src_feature,src_label)
    return classifier.score(src_feature,src_label)

def buildNewFeatures(trees,toolbox):
    src_feature_new = np.empty((len(src_feature),0))
    tar_feature_new = np.empty((len(tar_feature),0))

    for tree in trees:
        func = toolbox.compile(expr=tree);
        src_tree_new = np.array([func(*row) for row in src_feature])
        src_tree_new = np.reshape(src_tree_new, (len(src_feature),1))
        tar_tree_new = np.array([func(*row) for row in tar_feature])
        tar_tree_new = np.reshape(tar_tree_new, (len(tar_feature),1))
        src_feature_new = np.append(src_feature_new, src_tree_new, axis=1)
        tar_feature_new = np.append(tar_feature_new,tar_tree_new, axis=1)

    return src_feature_new,tar_feature_new

def sigmoid(x):
    return 1 / (1 + math.exp(-x))
