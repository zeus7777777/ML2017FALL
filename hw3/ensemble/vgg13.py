import csv 
import sys

import tensorflow as tf
import numpy as np

def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            i += 1
            if i==1:
                continue
            data.append(row[1].split())
        return np.array(data, dtype=np.float)

def conv(input, shape_):
    w = tf.Variable(tf.truncated_normal(shape_))
    b = tf.Variable(tf.constant(0.1, shape=[shape_[3]]))
    return tf.nn.relu(tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='SAME') + b)

def pool(input):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def fc(input, shape_, act=None):
    w = tf.Variable(tf.truncated_normal(shape_))
    b = tf.Variable(tf.constant(0.1, shape=[shape_[1]]))
    if act is not None:
        return act(tf.matmul(input, w) + b)
    else:
        return tf.matmul(input, w) + b

def test(test_x):
    global_epoch = tf.Variable(0, trainable=False, dtype=tf.int32)
    input_x = tf.placeholder(tf.float32, [None, 48, 48])
    input_y = tf.placeholder(tf.int32, [None])
    distort_input = tf.placeholder(tf.int32, [])
    is_testing = tf.placeholder(tf.int32, [])

    image = tf.reshape(input_x, [-1, 48, 48, 1])

    # augment data
    aug_image = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), image)
    #aug_image = tf.contrib.image.rotate(aug_image, tf.random_uniform([1], minval=-np.pi/18, maxval=np.pi/18))
    aug_image = tf.map_fn(lambda img: tf.random_crop(img, [40, 40, 1]), aug_image)
    #aug_image = tf.image.crop_to_bounding_box(aug_image, 4, 4, 40, 40)
    #aug_image = tf.map_fn(lambda img: tf.image.random_brightness(img, 63/255.0), aug_image)
    #aug_image = tf.map_fn(lambda img: tf.image.random_contrast(img, 0.2, 1.8), aug_image)

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

    pred = tf.argmax(fc3, axis=1)
    prob = tf.nn.softmax(fc3)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('./model/vgg13/')
        saver.restore(sess, ckpt.model_checkpoint_path)

        test_size = len(test_x)
        prob_ = np.zeros([test_size, 7])
        for __ in range(10):
            pred_ = []
            for i in range(int(np.ceil(test_size/256))):
                feed_x = list(test_x[i*256:(i+1)*256])
                pred_.extend(sess.run(prob, feed_dict={input_x: feed_x, distort_input:0, is_testing:1, keep_prob:1.0}))
            prob_ += pred_
        ans = np.argmax(prob_, axis=1)
        return ans

if __name__=='__main__':
    test_x = read_data(sys.argv[1]) / 255.0
    test_x = np.reshape(test_x, [len(test_x), 48, 48])
    ans = test(test_x)
    print(ans)
    print(len(ans))

    with open(sys.argv[2], 'w') as f:
        f.write('id,label\n')
        for i in range(len(ans)):
            f.write(str(i)+','+str(ans[i])+'\n')
