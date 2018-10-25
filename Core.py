#Contain all data and classification algorithms

import numpy as np
import Utility
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import FitnessFunction
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics.pairwise import euclidean_distances as ecd
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


src_data = np.genfromtxt("data/Source", delimiter=",")
no_features = len(src_data[0])-1
src_feature = src_data[:, 0:no_features]
src_label = np.ravel(src_data[:, no_features:no_features+1])
no_classes = len(np.unique(src_label))

tarU_data = np.genfromtxt("data/TargetU", delimiter=",")
tarU_feature = tarU_data[:, 0:no_features]
tarU_label = np.ravel(tarU_data[:, no_features:no_features+1])

tarL_data = np.genfromtxt("data/TargetL", delimiter=",")
tarL_feature = tarL_data[:, 0:no_features]
tarL_label = np.ravel(tarL_data[:, no_features:no_features+1])

features = np.concatenate((src_feature, tarU_feature, tarL_feature), axis=0)
features_new = Utility.standarizeData(features)

src_feature = features_new[0:len(src_feature), ]
tarU_feature = features_new[len(src_feature): len(src_feature)+len(tarU_feature), ]
tarL_feature = features_new[len(src_feature)+len(tarU_feature): len(src_feature)+len(tarU_feature)+len(tarL_feature), ]


#src_feature, src_label = Utility.balanced_sample_maker(src_feature, src_label, sample_size=10)
#tarL_feature, tarL_label = Utility.balanced_sample_maker(tarL_feature, tarL_label, sample_size=1)
#tarU_feature, tarU_label = Utility.balanced_sample_maker(tarU_feature, tarU_label, sample_size=10)


classifier = \
KNeighborsClassifier(n_neighbors=1, algorithm='brute')
# LinearSVC(random_state=1617)

# create a feature selection
#sfs = SFS(classifier, k_features=no_features/10, forward=True, floating=False, scoring='accuracy')
#sfs = sfs.fit(src_feature, src_label)
#f_indices = sfs.k_feature_idx_
#src_feature = src_feature[:, f_indices]
#tarL_feature = tarL_feature[:, f_indices]
#tarU_feature = tarU_feature[:, f_indices]
#no_features = len(f_indices)
#print(np.shape(src_feature))
