import csv
import numpy as np

def read_data(filename):
    with open(filename, 'r', encoding='big5') as f:
        reader = csv.reader(f)
        print(next(reader))
        f = []
        feature = []

        x = []
        y = []
        for i in range(4320):
            row = next(reader)
            row = [float(row[i]) if row[i]!='NR' else 0.0 for i in range(3, len(row))]
            f.append(row)
        for i in range(12):
            for j in range(20):
                for k in range(24):
                    t = []
                    for l in range(18):
                        t.append(f[i*360 + j*18 + l][k])
                    feature.append(t)

        for i in range(12):
            for j in range(480-9):
                t = []
                for k in range(9):
                    t.extend(feature[i*480 + j + k])
                x.append(t)
                y.append(feature[i*480 + j + k + 1][9])
        return x, y

def train(input_x, input_y):
    load_model = False
    lr = 1
    epoch = 30000
    beta = 0

    input_x = np.array(input_x)
    input_y = np.array(input_y)

    input_x_t = input_x.transpose()
    beta /= len(input_x)

    n_dim = len(input_x[0])

    w = np.array([0.0]*n_dim)
    b = 0.0

    ada_w = np.array([0.0]*n_dim)
    ada_b = 0.0

    if load_model:
        w, b = load()

    for _ in range(epoch):
        err = input_y - (np.dot(input_x, w)+b)
        if (_+1)%1000==0: 
            orig_loss = np.sum(np.square(err))/len(input_x)
            loss = orig_loss + beta*np.sum(np.square(w))
            print(_+1, np.sqrt(orig_loss))
        g = -np.dot(input_x_t, err)/len(input_x) + beta*w
        bg = -np.sum(err)/len(input_x)

        ada_w += np.square(g)
        ada_b += np.square(bg)

        w += -lr * (g/np.sqrt(ada_w))
        b += -lr * (bg/np.sqrt(ada_b))
    store(w, b)

def load():
    w = []
    b = 0
    with open('simple_model_reg.txt', 'r') as f:
        arr = f.read().split()
        for i in range(len(arr)-1):
            w.append(float(arr[i]))
        b = float(arr[-1])
    return np.array(w), b

def store(w, b):
    with open('simple_model_reg.txt', 'w') as f:
        for i in range(len(w)):
            f.write(str(w[i])+'\n')
        f.write(str(b)+'\n')

def load_std():
    m = []
    s = []
    with open('simple_model_regstd.txt', 'r') as f:
        arr = f.read.split()
        m = arr[:162]
        s = arr[162:]
        return m, s

def store_std(m, s):
    with open('simple_model_regstd.txt', 'w') as f:
        for i in range(len(m)):
            f.write(str(m[i])+'\n')
        for i in range(len(s)):
            f.write(str(s[i])+'\n')

if __name__=='__main__':
    input_x, input_y = read_data('train.csv')

    input_x = np.array(input_x)

    x_ext = np.square(input_x)
    input_x = np.concatenate([input_x, x_ext], axis=1)
    print('x shape:', input_x.shape)

    mean = np.mean(input_x, axis=0)
    std = np.std(input_x, axis=0)
    store_std(mean, std)

    input_x = (input_x - mean)/std

    train(input_x, input_y)