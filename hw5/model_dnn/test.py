import tensorflow as tf
import numpy as np

import preprocess
import model_dnn
import config

if __name__=='__main__':
    userid, movieid, category, userattr = preprocess.read_test_data('../test.csv', '../movies.csv', '../users.csv')

    model = model_dnn.Model()
    mean = np.load('mean.npy')
    std = np.load('std.npy')
    userattr = (userattr-mean)/std
    
    with tf.Session() as sess:
        model.load(sess)

        batches = len(userid)//config.BATCH_SIZE
        pred_all = []
        for i in range(batches):
            a = userid[i*config.BATCH_SIZE:(i+1)*config.BATCH_SIZE]
            b = movieid[i*config.BATCH_SIZE:(i+1)*config.BATCH_SIZE]
            c = category[i*config.BATCH_SIZE:(i+1)*config.BATCH_SIZE]
            d = userattr[i*config.BATCH_SIZE:(i+1)*config.BATCH_SIZE]
            pred = model.predict(sess, a, b, c, d)
            pred_all.extend(pred)
        if len(userid)-config.BATCH_SIZE*batches>0:
            remain = len(userid)-config.BATCH_SIZE*batches
            a = userid[-remain:]
            b = movieid[-remain:]
            c = category[-remain:]
            d = userattr[-remain:]
            pred = model.predict(sess, a, b, c, d)
            pred_all.extend(pred)
        
        with open('output.txt', 'w') as f:
            f.write('TestDataID,Rating\n')
            for i in range(len(pred_all)):
                if pred_all[i]>5.0: pred_all[i] = 5.0
                if pred_all[i]<1.0: pred_all[i] = 1.0
                f.write(str(i+1)+','+str(pred_all[i])+'\n')