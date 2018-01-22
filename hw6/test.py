import numpy as np

def read_test_data(filename):
    test = []
    with open(filename, 'r') as f:
        lines = f.read().strip().split('\n')[1:]
        for line in lines:
            arr = line.split(',')
            test.append([int(arr[1]), int(arr[2])])
    return test

if __name__=='__main__':
    labels = np.load('labels.npy')
    test = read_test_data('test_case.csv')

    with open('output.txt', 'w') as f:
        f.write('ID,Ans\n')
        for i in range(len(test)):
            a = test[i][0]
            b = test[i][1]
            if labels[a]==labels[b]:
                f.write(str(i)+',1\n')
            else:
                f.write(str(i)+',0\n')
