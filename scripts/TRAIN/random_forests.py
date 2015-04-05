import csv
import sklearn as sk
import numpy as np
import math
import random

from sklearn.cross_validation import train_test_split
from sklearn import ensemble

NUM_TREES=1000
MAX_FEATS=40

cls = {
    "Class_1":0,
    "Class_2":1,
    "Class_3":2,
    "Class_4":3,
    "Class_5":4,
    "Class_6":5,
    "Class_7":6,
    "Class_8":7,
    "Class_9":8
}

def load_csv(filename):
    f = open(filename)
    f.readline()
    data = []
    for line in f:
        line = line.rstrip()
        d = line.split(',')
        data.append(map(float,d[1:94]))
        if len(d) == 95:
            data[-1].append(cls[d[94]])
    return np.array(data)

def save_csv(filename, data):
    header = "id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9"
    ids = np.array( [ [i+1] for i in range(data.shape[0]) ] )
    data = np.concatenate( (ids,data), axis=1)
    fmt = "%.10f"
    np.savetxt(filename, data, delimiter=",", header=header,
               fmt=[ "%.0f", fmt, fmt, fmt, fmt, fmt, fmt, fmt, fmt, fmt] )
    

def fit(train_feats, train_labels):
    model = ensemble.RandomForestClassifier(n_estimators=NUM_TREES,
                                            max_features=MAX_FEATS,
                                            random_state=123,
                                            verbose=100)
    model.fit(train_feats, train_labels)
    return model

data = load_csv("DATA/train.csv")

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
save_csv("result.rf.csv", test_p)
