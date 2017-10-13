import csv
import sys

import numpy as np

def read_data(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        input_x = []
        for i in range(240):
            f = []
            tmp = []
            for j in range(18):
                row = next(reader)
                f.append([float(row[k]) if row[k]!='NR' else 0.0 for k in range(2, len(row))])
            for j in range(9):
                for k in range(18):
                    tmp.append(f[k][j])
            input_x.append(tmp)
        return input_x

def load():
    w = []
    b = 0
    with open('model/model1.txt', 'r') as f:
        arr = f.read().split()
        for i in range(len(arr)-1):
            w.append(float(arr[i]))
        b = float(arr[-1])
    return w, b

def load_std():
    m = []
    s = []
    with open('model/model1std.txt', 'r') as f:
        arr = f.read().split()
        m = arr[:len(arr)//2]
        s = arr[len(arr)//2:]
        return m, s

if __name__=='__main__':
    w, b = load()
    input_x = np.array(read_data(sys.argv[1]))
    x_ext = np.square(input_x)
    input_x = np.concatenate([input_x, x_ext], axis=1)

    mean, std = load_std()
    mean = np.array(mean, dtype=np.float)
    std = np.array(std, dtype=np.float)
    input_x = (input_x-mean)/std

    answer = []
    for i in range(len(input_x)):
        answer.append(sum([w[j]*input_x[i][j] for j in range(len(w))]) + b)
    with open(sys.argv[2], 'w') as f:
        f.write('id,value\n')
        for i in range(len(answer)):
            f.write('id_'+str(i)+','+str(answer[i])+'\n')
