from scripts import common
import csv
import sklearn as sk
import numpy as np
import math
import random
import sys

from sklearn.cross_validation import train_test_split
from sklearn import ensemble

ID=int(sys.argv[1])

NUM_TREES=1000
MAX_FEATS=40

def fit(train_feats, train_labels):
    model = ensemble.RandomForestClassifier(n_estimators=NUM_TREES,
                                            max_features=MAX_FEATS,
                                            random_state=123,
                                            verbose=100)
    model.fit(train_feats, train_labels)
    return model

train_feats  = np.loadtxt("DATA/train_feats.raw.split.mat.gz")
train_labels = np.loadtxt("DATA/train_labels.split.mat.gz")
val_feats    = np.loadtxt("DATA/val_feats.raw.split.mat.gz")
val_labels   = np.loadtxt("DATA/val_labels.split.mat.gz")

model = fit(train_feats, train_labels)
train_p = model.predict_proba(train_feats)
val_p = model.predict_proba(val_feats)
print "# TR LOSS",sk.metrics.log_loss(train_labels, train_p)
print "# VA LOSS",sk.metrics.log_loss(val_labels, val_p)
val_cls = model.predict(val_feats)
print "# VA ACC ",sk.metrics.accuracy_score(val_labels, val_cls)
common.save_csv("ID_%03d.validation.rf.csv"%(ID), val_p)

train_feats = np.concatenate( (train_feats, val_feats), axis=0 )
train_labels = np.concatenate( (train_labels, val_labels), axis=0 )

model = fit(train_feats, train_labels)
test_feats = np.loadtxt("DATA/test_feats.raw.split.mat.gz")
test_p = model.predict_proba(test_feats)
common.save_csv("ID_%03d.test.rf.csv"%(ID), test_p)
