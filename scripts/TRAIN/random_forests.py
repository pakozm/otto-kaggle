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
    model = ensemble.RandomForestClassifier(n_estimators=NUM_TREES,
                                            max_features=MAX_FEATS,
                                            random_state=123,
                                            verbose=100)
    model.fit(train_feats, train_labels)
    return model

data = common.load_csv("DATA/train.csv")

train_data,val_data = train_test_split(data, test_size=0.20, random_state=42)

train_feats = train_data[:,0:93]
train_labels = train_data[:,93]
val_feats = val_data[:,0:93]
val_labels = val_data[:,93]

model = fit(train_feats, train_labels)
val_p = model.predict_proba(val_feats)
print sk.metrics.log_loss(val_labels, val_p)

train_feats = data[:,0:93]
train_labels = data[:,93]
model = fit(train_feats, train_labels)
test_feats = load_csv("DATA/test.csv")
test_p = model.predict_proba(test_feats)
common.save_csv("result.rf.csv", test_p)
