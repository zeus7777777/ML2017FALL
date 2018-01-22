import pickle
import collections
import os
import numpy as np
import tensorflow  as tf

import config as cfg

def read_train_data():
    caption = np.load('caption.npy')
    caption_len = np.load('caption_len.npy')

    with open('int2str.pkl', 'rb') as f:
        int2str = pickle.load(f)
    with open('str2int.pkl', 'rb') as f:
        str2int = pickle.load(f)

    print('dict len:', len(int2str))

    mfcc = np.load('../mfcc.npy')
    mfcc_len = np.load('../mfcc_len.npy')
    
    return mfcc, mfcc_len, caption, caption_len, int2str, str2int

def read_test_data():
    mfcc = np.load('../mfcc.test.npy')
    mfcc_len = np.load('../mfcc_len.test.npy')
    with open('caption.test.pkl', 'rb') as f:
        caption = pickle.load(f)
    with open('int2str.pkl', 'rb') as f:
        int2str = pickle.load(f)
    with open('str2int.pkl', 'rb') as f:
        str2int = pickle.load(f)
    return mfcc, mfcc_len, caption, int2str, str2int

def preprocess_train_data(file_data, file_caption):
    '''
    can only run once dut to dictionary construction
    '''
    
    print('loading mfcc feature')
    with open(file_data, 'rb') as f:
        data = pickle.load(f)
    data_output = []
    data_len = []
    for i in range(len(data)):
        data_output.append([])
        data_len.append(len(data[i]))
        for j in range(cfg.ENCODER_MAX_TIME_STEP):
            if j<len(data[i]):
                data_output[i].append(data[i][j])
            else:
                data_output[i].append([0.0]*39)
    print('mfcc feature processed')
    data_output = np.array(data_output)
    data_len = np.array(data_len)
    np.save('../mfcc.npy', data_output)
    np.save('../mfcc_len.npy', data_len)
    print('mfcc feature saved')
    

    print('loading caption')
    with open(file_caption, 'r') as f:
        caption = f.read().strip().split('\n')
        caption = [line.split() for line in caption]
    print('total captions:', len(caption))
    print('first 10 captions', caption[:10])
    print('max caption len', max([len(caption[i]) for i in range(len(caption))]))

    all_words = []
    for line in caption:
        all_words += line
    counter = collections.Counter(all_words)

    #for x in counter.most_common():
    #    print(x[0], x[1])

    int2str = ['<unk>', '<pad>'] + [x[0] for x in counter.most_common()]
    str2int = {x:i for i, x in enumerate(int2str)}
    print('dict size:', len(int2str))
    print('first 100 int2str:', int2str[:100])
    print('check:', [str2int.get(int2str[i], 0) for i in range(10)])

    with open('int2str.pkl', 'wb') as f:
        pickle.dump(int2str, f)
    with open('str2int.pkl', 'wb') as f:
        pickle.dump(str2int, f)

    for i in range(len(caption)):
        for j in range(len(caption[i])):
            caption[i][j] = str2int.get(caption[i][j], 0)
    print('first 10 captions (converted)', caption[:10])
    caption_len = [len(x) for x in caption]
    print('first 10 captions\' len:', caption_len[:10],'max caption len:', max(caption_len))

    cap = []
    for i in range(len(caption)):
        cap.append([])
        for j in range(cfg.DECODER_MAX_TIME_STEP):
            if j<caption_len[i]:
                cap[i].append(caption[i][j])
            else:
                cap[i].append(1)
    cap = np.array(cap)
    caption_len = np.array(caption_len)
    print('first 10 captions (padded)', cap[:10])
    np.save('caption.npy', cap)
    np.save('caption_len.npy', caption_len)

def preprocess_test_data(file_data, file_caption):
    with open(file_data, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    data_len = [len(x) for x in data]
    data_out = []
    for i in range(len(data)):
        data_out.append([])
        for j in range(cfg.ENCODER_MAX_TIME_STEP):
            if j<len(data[i]):
                data_out[i].append(data[i][j])
            else:
                data_out[i].append([0.0]*39)
    data_out = np.array(data_out)
    data_len = np.array(data_len)
    np.save('../mfcc.test.npy', data_out)
    np.save('../mfcc_len.test.npy', data_len)

    with open(file_caption, 'r') as f:
        caption = f.read().strip().split('\n')
    for i in range(len(caption)):
        caption[i] = [x.split() for x in caption[i].split(',')]
    print(caption[:10])

    with open('str2int.pkl', 'rb') as f:
        str2int = pickle.load(f)
    with open('int2str.pkl', 'rb') as f:
        int2str = pickle.load(f)
    for i in range(len(caption)):
        for j in range(len(caption[i])):
            for k in range(len(caption[i][j])):
                caption[i][j][k] = str2int.get(caption[i][j][k], 0)
    with open('caption.test.pkl', 'wb') as f:
        pickle.dump(caption, f)
    print(caption[:10])

def create_tfrecord(filename, mfcc, mfcc_len, caption, caption_out, mask):
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(len(mfcc)):
        assert len(caption_out[i])==14
        example = tf.train.Example(features=tf.train.Features(feature={
            'mfcc': tf.train.Feature(float_list=tf.train.FloatList(value=np.reshape(mfcc[i], [cfg.ENCODER_MAX_TIME_STEP*39]))),
            'mfcc_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[mfcc_len[i]])),
            'caption': tf.train.Feature(int64_list=tf.train.Int64List(value=caption[i])),
            #'caption_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[caption_len[i]])),
            'caption_out': tf.train.Feature(int64_list=tf.train.Int64List(value=caption_out[i])),
            'mask': tf.train.Feature(float_list=tf.train.FloatList(value=mask[i]))
        }))
        writer.write(example.SerializeToString())
    writer.close()

if __name__=='__main__':
    #read_train_data()
    #preprocess_train_data('../data/train.data', '../data/train.caption')

    preprocess_test_data('../data/test.data', '../data/test.csv')