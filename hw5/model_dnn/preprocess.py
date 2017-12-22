import csv
import collections
import pickle

import numpy as np

import config

def read_train_data(filename, filename2, filename3):
    userid = []
    movieid = []
    rating = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            i += 1
            if i==1: continue
            uid = int(row[1])
            mid = int(row[2])
            r = int(row[3])
            userid.append(uid)
            movieid.append(mid)
            rating.append(r)

    with open(filename2, 'r', encoding='latin-1') as f:
        lines = f.read().strip().split('\n')[1:]
    if config.BUILD_DICT:
        all_cat = [x for line in lines for x in line.split('::')[-1].split('|')]
        counter = collections.Counter(all_cat)
        int2movie = [x[0] for x in counter.most_common()]
        movie2int = {x:i for i, x in enumerate(int2movie)}
        with open('int2movie.pkl', 'wb') as g: 
            pickle.dump(int2movie, g)
        with open('movie2int.pkl', 'wb') as g: 
            pickle.dump(movie2int, g)
    else:
        with open('int2movie.pkl', 'rb') as g: 
            int2movie = pickle.load(g)
        with open('movie2int.pkl', 'rb') as g: 
            movie2int = pickle.load(g)
    print(len(int2movie))
    print(int2movie[:10])

    category = np.zeros([config.N_MOVIE+1, config.N_CATEGORY])
    for i in range(len(lines)):
        cat = lines[i].split('::')[-1].split('|')
        for c in cat:
            category[i][movie2int[c]] = 1
    print(category[:10])

    cat_ = []
    for i in range(len(userid)):
        cat_.append(category[movieid[i]])
    category = cat_
    print('cat len', len(category))

    user = np.zeros([config.N_USER+1, 3])
    with open(filename3, 'r') as f:
        lines = f.read().strip().split('\n')[1:]
    for i in range(len(lines)):
        attr = lines[i].split('::')
        uid, gender, age, occu = int(attr[0]), attr[1], int(attr[2]), int(attr[3])
        user[uid][0] = 0 if gender=='M' else 1
        user[uid][1] = age
        user[uid][2] = occu
    print(user[:10])

    user_ = []
    for i in range(len(userid)):
        user_.append(user[userid[i]])
    user = user_

    return userid, movieid, rating, category, user

def read_test_data(filename, filename2, filename3):
    userid = []
    movieid = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            i += 1
            if i==1: continue
            uid = int(row[1])
            mid = int(row[2])
            userid.append(uid)
            movieid.append(mid)
    
    with open(filename2, 'r', encoding='latin-1') as f:
        lines = f.read().strip().split('\n')[1:]
    with open('int2movie.pkl', 'rb') as g: int2movie = pickle.load(g)
    with open('movie2int.pkl', 'rb') as g: movie2int = pickle.load(g)

    category = np.zeros([config.N_MOVIE+1, config.N_CATEGORY])
    for i in range(len(lines)):
        cat = lines[i].split('::')[-1].split('|')
        for c in cat:
            category[i][movie2int[c]] = 1
    print(category[:10])

    cat_ = [category[movieid[i]] for i in range(len(userid))]
    category = cat_
    print('cat len', len(category))

    user = np.zeros([config.N_USER+1, 3])
    with open(filename3, 'r') as f:
        lines = f.read().strip().split('\n')[1:]
    for i in range(len(lines)):
        attr = lines[i].split('::')
        uid, gender, age, occu = int(attr[0]), attr[1], int(attr[2]), int(attr[3])
        user[uid][0] = 0 if gender=='M' else 1
        user[uid][1] = age
        user[uid][2] = occu
    print(user[:10])

    user_ = [user[userid[i]] for i in range(len(userid))]
    user = user_

    return userid, movieid, category, user

if __name__=='__main__':
    #read_train_data('../train.csv', '../movies.csv', '../users.csv')
    read_test_data('../test.csv', '../movies.csv', '../users.csv')
