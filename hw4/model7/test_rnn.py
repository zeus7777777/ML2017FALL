import tensorflow as tf
import numpy as np
import gensim 

import preprocess
import config
import model as Model

def word2embed(embedding, sentences):
    res = [[[0]*config.WORD_DIM if word=='<pad>' else embedding.wv[str(word)] for word in sentence] for sentence in sentences]
    return res

def test(test_x):
    model = Model.Model()
    embedding = gensim.models.word2vec.Word2Vec.load(config.EMBED_FILE)
    with tf.Session() as sess:
        model.load(sess)

        pred_ = []
        v = len(test_x)//config.BATCH_SIZE
        for i in range(v):
            feed_x = test_x[i*config.BATCH_SIZE : (i+1)*config.BATCH_SIZE]
            feed_x = word2embed(embedding, feed_x)
            p = model.predict(sess, feed_x)
            pred_.extend(p)
        remain = len(test_x)-v*config.BATCH_SIZE
        if remain>0:
            feed_x = test_x[v*config.BATCH_SIZE:]
            for i in range(config.BATCH_SIZE-remain):
                feed_x.append([0]*config.MAX_SEQ_LEN)
            feed_x = word2embed(embedding, feed_x)
            pred_.extend(model.predict(sess, feed_x))
    return pred_[:len(test_x)]
            
    
if __name__=='__main__':
    test_x = preprocess.read_test_data('../testing_data.txt')
    seq_len = [min(len(test_x[i]), config.MAX_SEQ_LEN) for i in range(len(test_x))]
    test_x = [test_x[i][:config.MAX_SEQ_LEN] if seq_len[i]>=config.MAX_SEQ_LEN else test_x[i][:config.MAX_SEQ_LEN]+['<pad>']*(config.MAX_SEQ_LEN-seq_len[i]) for i in range(len(test_x))]
    print(test_x[:10])

    ans = test(test_x)
    with open('output.txt', 'w') as f:
        f.write('id,label\n')
        for i in range(len(ans)):
            f.write(str(i)+','+str(ans[i])+'\n')
