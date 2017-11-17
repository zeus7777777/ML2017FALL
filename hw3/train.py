import csv
import sys

import numpy as np
import tensorflow as tf

def read_data(filename):
    data = []
    label = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            i += 1
            if i==1:
                continue
            data.append(row[1].split())
            label.append(row[0])
    data = np.array(data, dtype=np.float)
    label = np.array(label, dtype=np.intc)
    print(len(data), data[0], len(data[0]))
    print(label[:10])
    cat = [0]*7
    for i in range(len(label)):
        cat[label[i]] += 1
    print(cat)

    return data, label

def conv(input, shape_):
    w = tf.Variable(tf.truncated_normal(shape_, stddev=0.1))
    b = tf.Variable(tf.constant(0.0, shape=[shape_[3]]))
    return tf.nn.relu(tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='SAME') + b)

def pool(input):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def fc(input, shape_, act=None):
    w = tf.Variable(tf.truncated_normal(shape_, stddev=0.1))
    b = tf.Variable(tf.constant(0.0, shape=[shape_[1]]))
    if act is not None:
        return act(tf.matmul(input, w) + b)
    else:
        return tf.matmul(input, w) + b

def train_valid(train_x, train_y, weight, valid_x, valid_y):
    batch_size = 128
    epoch = 200
    learning_rate = 1e-5
    dropout_rate = 0
    train_size = len(train_x)

    global_epoch = tf.Variable(0, trainable=False, dtype=tf.int32)
    input_x = tf.placeholder(tf.float32, [None, 48, 48])
    input_y = tf.placeholder(tf.int32, [None])
    input_w = tf.placeholder(tf.float32, [None])
    distort_input = tf.placeholder(tf.int32, [])
    is_testing = tf.placeholder(tf.int32, [])

    image = tf.reshape(input_x, [-1, 48, 48, 1])

    # augment data
    aug_image = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), image)
    aug_image = tf.contrib.image.rotate(aug_image, tf.random_uniform([1], minval=-np.pi/18, maxval=np.pi/18))
    aug_image = tf.map_fn(lambda img: tf.random_crop(img, [40, 40, 1]), aug_image)
    #aug_image = tf.image.crop_to_bounding_box(aug_image, 4, 4, 40, 40)
    aug_image = tf.map_fn(lambda img: tf.image.random_brightness(img, 63/255.0), aug_image)
    aug_image = tf.map_fn(lambda img: tf.image.random_contrast(img, 0.2, 1.8), aug_image)

    # valid data
    image_ = tf.image.crop_to_bounding_box(image, 4, 4, 40, 40)
    
    # test
    test_image = tf.contrib.image.rotate(image, tf.random_uniform([1], minval=-np.pi/18, maxval=np.pi/18))
    test_image = tf.map_fn(lambda img: tf.random_crop(img, [40, 40, 1]), test_image)

    image = tf.cond(tf.equal(distort_input, 1), lambda: aug_image, lambda: image_)
    image = tf.cond(tf.equal(is_testing, 1), lambda: test_image, lambda: image)

    conv1 = conv(image, [3, 3, 1, 64])
    conv1_2 = conv(conv1, [3, 3, 64, 64])
    pool1 = pool(conv1_2)

    conv2 = conv(pool1, [3, 3, 64, 128])
    conv2_2 = conv(conv2, [3, 3, 128, 128])
    pool2 = pool(conv2_2)

    conv3 = conv(pool2, [3, 3, 128, 256])
    conv3_2 = conv(conv3, [3, 3, 256, 256])
    pool3 = pool(conv3_2)

    conv4 = conv(pool3, [3, 3, 256, 512])
    conv4_2 = conv(conv4, [3, 3, 512, 512])
    pool4 = pool(conv4_2)

    conv5 = conv(pool4, [3, 3, 512, 512])
    conv5_2 = conv(conv5, [3, 3, 512, 512])
    pool5 = pool(conv5_2)

    flat = tf.reshape(pool5, [-1, 2*2*512])

    keep_prob = tf.placeholder(tf.float32)
    fc1 = fc(flat, [2*2*512, 4096], act=tf.nn.relu)
    fc1 = tf.nn.dropout(fc1, keep_prob)
    fc2 = fc(fc1, [4096, 4096], act=tf.nn.relu)
    fc2 = tf.nn.dropout(fc2, keep_prob)
    fc3 = fc(fc2, [4096, 7])

    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input_y, logits=fc3)*input_w)   
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input_y, logits=fc3)) 
    
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    pred = tf.argmax(fc3, axis=1)
    prob = tf.nn.softmax(fc3)

    summary_writer = tf.summary.FileWriter('log/')

    with tf.Session() as sess:
        summary_writer.add_graph(sess.graph)

        '''saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('.')
        saver.restore(sess, ckpt.model_checkpoint_path)'''

        sess.run(tf.global_variables_initializer())
        for _ in range(epoch):

            sess.run(global_epoch.assign_add(1))
            ep = sess.run(global_epoch)
            print(ep)
            tmp_ = list(zip(train_x, train_y, weight))
            np.random.shuffle(tmp_)
            train_x, train_y, weight = list(zip(*tmp_))

            for i in range(int(np.ceil(train_size/batch_size))):
                feed_x = np.copy(train_x[i*batch_size:(i+1)*batch_size])
                feed_y = train_y[i*batch_size:(i+1)*batch_size]
                feed_w = weight[i*batch_size:(i+1)*batch_size]

                loss_, __ = sess.run([loss, train_step], feed_dict={input_x:feed_x, input_y:feed_y, input_w:feed_w, distort_input:1, is_testing:0, keep_prob:0.5})
                if (i+1)%49==0:
                    print('batch:', (i+1), '/', int(np.ceil(train_size/batch_size)), 'loss:', loss_)

            if (_+1)%5==0:
                summary = tf.Summary()
                summary.value.add(tag='Loss/Training Loss', simple_value=float(loss_))

                pred_ = []
                for i in range(int(np.ceil(train_size/batch_size))):
                    feed_x = list(train_x[i*batch_size:(i+1)*batch_size])
                    feed_y = train_y[i*batch_size:(i+1)*batch_size]
                    pred_.extend(sess.run(pred, feed_dict={input_x: feed_x, input_y: feed_y, distort_input:0,   is_testing:0, keep_prob:1.0}))
                corr = 0.0
                for i in range(train_size):
                    if pred_[i]==train_y[i]:
                        corr += 1
                print('Epoch:', ep, 'Accuracy:', corr/train_size)
                summary.value.add(tag='Accuracy/Training Accuracy', simple_value=float(corr/train_size))
            
                '''valid_size = len(valid_x)
                pred_ = []
                for i in range(int(np.ceil(valid_size/batch_size))):
                    feed_x = list(valid_x[i*batch_size:(i+1)*batch_size])
                    feed_y = valid_y[i*batch_size:(i+1)*batch_size]
                    pred_.extend(sess.run(pred, feed_dict={input_x: feed_x, input_y:feed_y, distort_input:0,   is_testing:0, keep_prob:1.0}))
                corr = [1 if pred_[i]==valid_y[i] else 0 for i in range(valid_size)]
                print('Validation accuracy:', sum(corr)/valid_size)
                summary.value.add(tag='Accuracy/Validation Accuracy', simple_value=sum(corr)/valid_size)

                # new test
                prob_ = np.zeros([valid_size, 7])
                for __ in range(10):
                    pred_ = []
                    for i in range(int(np.ceil(valid_size/batch_size))):
                        feed_x = list(valid_x[i*batch_size:(i+1)*batch_size])
                        feed_y = valid_y[i*batch_size:(i+1)*batch_size]
                        pred_.extend(sess.run(prob, feed_dict={input_x: feed_x, input_y:feed_y, distort_input:0, is_testing:1, keep_prob:1.0}))
                    prob_ += pred_
                ans = np.argmax(prob_, axis=1)
                corr = [1 if ans[i]==valid_y[i] else 0 for i in range(valid_size)]
                print('Multicrop valid accuracy:', sum(corr)/valid_size)
                summary.value.add(tag='Accuracy/Validation Accuracy (MultiCrop)', simple_value=sum(corr)/valid_size)

                summary_writer.add_summary(summary, ep)
                summary_writer.flush()'''

                saver = tf.train.Saver()
                saver.save(sess, './model.ckpt')
        
        saver = tf.train.Saver()
        saver.save(sess, './model.ckpt')

if __name__=='__main__':
    print('reading data')
    data, label = read_data(sys.argv[1])

    data = np.array(data)
    label = np.array(label)

    '''train_x = data[:25000] / 255.0
    train_y = label[:25000]
    valid_x = data[25000:] / 255.0
    valid_y = label[25000:]'''
    train_x = data[:] / 255.0
    train_y = label[:]
    valid_x = []
    valid_y = []

    train_x = np.reshape(train_x, [len(train_x), 48, 48])
    valid_x = np.reshape(valid_x, [len(valid_x), 48, 48])

    cat = [0]*7
    w = []
    for i in range(len(train_x)):
        cat[train_y[i]] += 1
    for i in range(len(train_x)):
        w.append(cat[train_y[i]]/len(train_x))
    w = 1.0 / np.array(w)
    print(w[:10])

    print('start training')
    train_valid(train_x, train_y, w, valid_x, valid_y)