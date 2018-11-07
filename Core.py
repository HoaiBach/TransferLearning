#Contain all data and classification algorithms

import numpy as np
import Utility
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


src_data = np.genfromtxt("data/Source", delimiter=",")
no_features = len(src_data[0])-1
src_feature = src_data[:, 0:no_features]
src_label = np.ravel(src_data[:, no_features:no_features+1])
no_classes = len(np.unique(src_label))

tar_data = np.genfromtxt("data/Target", delimiter=",")
tar_feature = tar_data[:, 0:no_features]
tar_label = np.ravel(tar_data[:, no_features:no_features+1])

features = np.concatenate((src_feature, tar_feature), axis=0)
features_new = Utility.standarizeData(features)

src_feature = features_new[0:len(src_feature), ]
tar_feature = features_new[len(src_feature): len(src_feature)+len(tar_feature), ]

# src_feature, src_label = Utility.balanced_sample_maker(src_feature, src_label, sample_size=10)
# tarL_feature, tarL_label = Utility.balanced_sample_maker(tarL_feature, tarL_label, sample_size=1)
# tarU_feature, tarU_label = Utility.balanced_sample_maker(tarU_feature, tarU_label, sample_size=10)

classifier = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
#KNeighborsClassifier(n_neighbors=1, algorithm='brute')
#LinearSVC(random_state=1617)
