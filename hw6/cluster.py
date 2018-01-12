import numpy as np
import scipy.misc
import sklearn.cluster
import tensorflow as tf

import model as Model
import config as cfg

if __name__=='__main__':
    images = np.load('image.npy')
    model = Model.FeatureExtractor()
    features = np.zeros([len(images), 128])

    with tf.Session() as sess:
        model.load(sess)
        batches = len(images)//cfg.batch_size
        for i in range(batches):
            features[i*cfg.batch_size:(i+1)*cfg.batch_size, :] = \
                model.get_feature(sess, images[i*cfg.batch_size:(i+1)*cfg.batch_size]/255.0)
        remain = len(images)-cfg.batch_size*batches
        if remain>0:
            features[len(images)-remain:, :] = model.get_feature(sess, images[len(images)-remain:]/255.0)

    km = sklearn.cluster.KMeans(n_clusters=2, n_jobs=4).fit(features)
    label = km.labels_

    np.save('labels.npy', label)

    # debug

    cnt = 0
    for i in range(len(label)):
        if label[i]==0:
            cnt += 1
    print(cnt, len(label)-cnt)
    
