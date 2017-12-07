import tensorflow as tf
import gensim
import numpy as np

import preprocess
import config
import model as Model

def word2embed(embedding, sentences):
    res = [[[0]*config.WORD_DIM if word=='<pad>' else embedding.wv[word] for word in sentence] for sentence in sentences]
    return res

if __name__=='__main__':
    train_x, train_y, unlabel_x = preprocess.read_train_data('../training_label.txt', '../training_nolabel.txt')
    seq_len = [min(len(train_x[i]), config.MAX_SEQ_LEN) for i in range(len(train_x))]
    train_x = [train_x[i][:config.MAX_SEQ_LEN] if seq_len[i]>=config.MAX_SEQ_LEN else train_x[i][:config.MAX_SEQ_LEN]+['<pad>']*(config.MAX_SEQ_LEN-seq_len[i]) for i in range(len(train_x))]
    print(train_x[:10])

    if config.VALID_MODEL:
        valid_x = train_x[180000:]
        valid_y = train_y[180000:]
        train_x = train_x[:180000]
        train_y = train_y[:180000]
    else:
        valid_x = []
        valid_y = []
    
    model = Model.Model()
    embedding = gensim.models.word2vec.Word2Vec.load(config.EMBED_FILE)

    cfg = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333))
    with tf.Session(config=cfg) as sess:
        if config.LOAD_MODEL:
            model.load(sess)
        else:
            model.init(sess)
        
        for epoch in range(config.EPOCH):
            tmp = list(zip(train_x, train_y))
            np.random.shuffle(tmp)
            train_x, train_y = list(zip(*tmp))

            batches = len(train_x)//config.BATCH_SIZE
            for i in range(batches):
                feed_x = train_x[i*config.BATCH_SIZE : (i+1)*config.BATCH_SIZE]
                feed_y = train_y[i*config.BATCH_SIZE : (i+1)*config.BATCH_SIZE]
                feed_x = word2embed(embedding, feed_x)
                loss = model.train(sess, feed_x, feed_y)
                if (i+1)%config.DISPLAY_LOSS_PERIOD==0:
                    print('epoch:', epoch+1, '/', config.EPOCH, 'batch:', i+1, '/', batches, 'loss:', loss)
                
            model.save(sess)
        
            if (epoch+1)%config.VALID_PERIOD==0:
                accuracy = []
                for i in range(batches):
                    feed_x = train_x[i*config.BATCH_SIZE : (i+1)*config.BATCH_SIZE]
                    feed_y = train_y[i*config.BATCH_SIZE : (i+1)*config.BATCH_SIZE]
                    feed_x = word2embed(embedding, feed_x)
                    accuracy.append(model.accuracy(sess, feed_x, feed_y))
                print('Training Accuracy =', sum(accuracy)/len(accuracy))

                if config.VALID_MODEL:
                    accuracy = []
                    v = len(valid_x)//config.BATCH_SIZE
                    for i in range(v):
                        feed_x = valid_x[i*config.BATCH_SIZE : (i+1)*config.BATCH_SIZE]
                        feed_y = valid_y[i*config.BATCH_SIZE : (i+1)*config.BATCH_SIZE]
                        feed_x = word2embed(embedding, feed_x)
                        accuracy.append(model.accuracy(sess, feed_x, feed_y))
                    print('Validation Accuracy =', sum(accuracy)/len(accuracy))