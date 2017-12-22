import csv

import numpy as np

import config

def read_train_data(filename):
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
    return userid, movieid, rating

def read_test_data(filename):
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
    return userid, movieid

if __name__=='__main__':
    rating, mask = read_train_data('train.csv')
    print(rating)
    print(mask)
