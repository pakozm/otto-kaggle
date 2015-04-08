from scripts import common
import csv
import sklearn as sk
import numpy as np
import math
import random

from sklearn.cross_validation import train_test_split
from sklearn import ensemble

NUM_TREES=1000
MAX_FEATS=40

def fit(train_feats, train_labels):
    model = ensemble.GradientBoostingClassifier(n_estimators=NUM_TREES,
                                                max_features=MAX_FEATS,
                                                random_state=123,
                                                verbose=100,
                                                loss='deviance')
    model.fit(train_feats, train_labels)
    return model

train_feats  = np.loadtxt("DATA/train_feats.noname.split.mat.gz")
train_labels = np.loadtxt("DATA/train_labels.noname.split.mat.gz")
val_feats    = np.loadtxt("DATA/val_feats.noname.split.mat.gz")
val_labels   = np.loadtxt("DATA/val_labels.noname.split.mat.gz")

model = fit(train_feats, train_labels)
val_p = model.predict_proba(val_feats)
print sk.metrics.log_loss(val_labels, val_p)
common.save_csv("validation.bt.csv", val_p)

train_feats = np.concatenate( (train_feats, val_feats), axis=1 )
train_labels = np.concatenate( (train_labels, val_labels), axis=1 )

model = fit(train_feats, train_labels)
test_feats = np.loadtxt("DATA/test_feats.noname.split.mat.gz")
test_p = model.predict_proba(test_feats)
common.save_csv("result.bt.csv", test_p)
