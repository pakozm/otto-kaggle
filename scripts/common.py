import numpy as np

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
    np.savetxt(filename, data, delimiter=",", header=header, comments="",
               fmt=[ "%.0f", fmt, fmt, fmt, fmt, fmt, fmt, fmt, fmt, fmt] )
