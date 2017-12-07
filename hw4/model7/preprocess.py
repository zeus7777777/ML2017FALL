import collections
import numpy as np
import pickle
import collections

import config

def read_train_data(label_filename, unlabel_filename):
    with open(label_filename, 'r') as f, open(unlabel_filename, 'r') as g:
        lines = [line.split() for line in f.read().strip().split('\n')]
        lines2 = [line.split() for line in g.read().strip().split('\n')]

        label = [int(line[0]) for line in lines]
        np.save('train_label.npy', label)
        print(len(label), label[:100])
        
        #
        words = []
        for i in range(len(lines)):
            lines[i] = lines[i][2:]
            words += lines[i]
        for line in lines2:
            words += line
    
    counter = collections.Counter(words)
    with open('counter.pkl', 'wb') as f:
        pickle.dump(counter, f)

    for i in range(len(lines)):
        for j in range(len(lines[i])):
            if counter[lines[i][j]]<config.MIN_COUNT:
                lines[i][j] = '<unk>'
    for i in range(len(lines2)):
        for j in range(len(lines2[i])):
            if counter[lines2[i][j]]<config.MIN_COUNT:
                lines2[i][j] = '<unk>'
    
    print(lines[:5])
    print('')
    print(lines2[:5])
    print('')
    
    np.save('train_data_label.npy', lines)
    np.save('train_data_unlabel.npy', lines2)

    mmax = 0
    for i in range(len(lines)):
        mmax = max(len(lines[i]), mmax)
    print(mmax)

    mmax = 0
    cnt = 0
    for i in range(len(lines2)):
        mmax = max(len(lines2[i]), mmax)
        if len(lines2[i])>39:
            cnt += 1
    print(mmax, cnt)

    print(sum([1 for i in range(len(label)) if label[i]==1])/len(label))    

    return lines, label, lines2
    
def read_test_data(filename):
    test_x = []
    with open(filename, 'r') as f:
        data = f.read().strip().split('\n')[1:]
        for i in range(len(data)):
            test_x.append([j for j in data[i][data[i].index(',')+1:].split()])
    
    with open('counter.pkl', 'rb') as f:
        counter = pickle.load(f)
    for i in range(len(test_x)):
        for j in range(len(test_x[i])):
            if counter[test_x[i][j]]<config.MIN_COUNT:
                test_x[i][j] = '<unk>'
    print(test_x[:10])
    print(len(test_x))
    return test_x

if __name__=='__main__':
    #read_train_data('../training_label.txt', '../training_nolabel.txt')
    read_test_data('../testing_data.txt')