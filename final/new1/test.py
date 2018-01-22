import tensorflow as tf
import numpy as np

import preprocess
import model as Model
import config as cfg

if __name__=='__main__':
    mfcc, mfcc_len, caption, int2str, str2int = preprocess.read_test_data()

    model = Model.RetrievalModel()

    with tf.Session() as sess:
        model.load(sess, 'models/model_novalid.ckpt')
        output = []
        v = len(mfcc)//(cfg.BATCH_SIZE//4)
        for i in range(v):
            a = [] # mfcc
            b = [] # mfcc_len
            c = [] # caption
            d = [] # caption_len
            for j in range(cfg.BATCH_SIZE//4):
                for k in range(4):
                    a.append(mfcc[i*(cfg.BATCH_SIZE//4) + j])
                    b.append(mfcc_len[i*(cfg.BATCH_SIZE//4) + j])
                    sentence = list(caption[i*(cfg.BATCH_SIZE//4) + j][k])
                    orig_len = len(sentence)
                    if len(sentence)>cfg.DECODER_MAX_TIME_STEP:
                        sentence = sentence[:cfg.DECODER_MAX_TIME_STEP]
                    else:
                        sentence += [str2int['<pad>']]*(cfg.DECODER_MAX_TIME_STEP-len(sentence))
                    c.append(sentence)
                    d.append(orig_len)
            loss = model.predict(sess, a, b, c, d)
            for j in range(cfg.BATCH_SIZE//4):
                score = [loss[j*4], loss[j*4+1], loss[j*4+2], loss[j*4+3]]
                output.append(np.argmax(score))
        remain = len(mfcc) - v*(cfg.BATCH_SIZE//4)
        if remain>0:
            a = [] # mfcc
            b = [] # mfcc_len
            c = [] # caption
            d = [] # caption_len
            for idx in range(len(mfcc)-remain, len(mfcc)):
                for k in range(4):
                    a.append(mfcc[idx])
                    b.append(mfcc_len[idx])
                    sentence = list(caption[idx][k])
                    orig_len = len(sentence)
                    if len(sentence)>cfg.DECODER_MAX_TIME_STEP:
                        sentence = sentence[:cfg.DECODER_MAX_TIME_STEP]
                    else:
                        sentence += [str2int['<pad>']]*(cfg.DECODER_MAX_TIME_STEP-len(sentence))
                    c.append(sentence)
                    d.append(orig_len)
            for i in range(cfg.BATCH_SIZE-len(a)):
                a.append(mfcc[0])
                b.append(mfcc_len[0])
                c.append(sentence)
                d.append(orig_len)
            loss = model.predict(sess, a, b, c, d)
            for i in range(remain):
                score = [loss[i*4], loss[i*4+1], loss[i*4+2], loss[i*4+3]]
                output.append(np.argmax(score))

        ans = output
        print(ans)
        with open('output.txt', 'w') as f:
            f.write('id,answer\n')
            for i in range(len(ans)):
                f.write(str(i+1)+','+str(ans[i])+'\n')
