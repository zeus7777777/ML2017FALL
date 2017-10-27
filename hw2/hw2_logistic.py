import numpy as np
import pandas as pd

import sys

NORMALIZATION = False

def read_data(filename):
    data = pd.read_csv(filename).values.tolist()
    return data

def load(filename):
    with open(filename, 'r') as f:
        arr = f.read().split()
        w = np.array(arr[:-1], dtype=np.float)
        b = float(arr[-1])
    return w, b

if __name__=='__main__':
    test_x = read_data(sys.argv[1])
    w, b = load('model/hw2_logistic.txt')

    test_x = np.array(test_x)
    test_x = np.delete(test_x, 1, axis=1)

    pred = 1 / (1 + np.exp(-(np.dot(test_x, w)+b)))
    print(pred[-1000:])
    pred = pred >= 0.5
    pred = pred.astype(np.int)
    
    with open(sys.argv[2], 'w') as f:
        f.write('id,label\n')
        for i in range(len(pred)):
            f.write(str(i+1)+','+str(pred[i])+'\n')
