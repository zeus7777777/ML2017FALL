import tensorflow as tf
import numpy as np

import model_mf
import preprocess
import config

if __name__=='__main__':
    userid, movieid = preprocess.read_test_data('../test.csv')

    model = model_mf.ModelMatrixFactorization()
    
    with tf.Session() as sess:
        model.load(sess)
        rating = model.predict(sess)
    
    if config.NORMALIZE_RATING:
        rating = (np.array(rating)*4.0) + 1.0
    
    with open('output.txt', 'w') as f:
        f.write('TestDataID,Rating\n')
        for i in range(len(userid)):
            f.write(str(i+1)+','+str(rating[userid[i]][movieid[i]])+'\n')
    