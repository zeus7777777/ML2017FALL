import numpy as np
import pandas as pd
import tensorflow as tf
import sys

def read_data(filename):
    data = pd.read_csv(filename).values.tolist()
    return data

def load_ms(filename):
    with open(filename, 'r') as f:
        arr = f.read().split()
        m = np.array(arr[:len(arr)//2], dtype=np.float)
        s = np.array(arr[len(arr)//2:], dtype=np.float)
        return m, s

def test(test_x):
    test_x = np.array(test_x)

    m, s = load_ms('tf_ms.txt')
    test_x = (test_x-m)/s

    input_x = tf.placeholder(tf.float32, [None, len(test_x[0])])
    input_y = tf.placeholder(tf.int32, [None])
    y_ext = tf.reshape(input_y, [1, -1])

    fc1_w = tf.Variable(tf.truncated_normal([len(test_x[0]), 64]))
    fc1_b = tf.Variable(tf.constant(0.1, shape=[64]))
    fc1 = tf.nn.relu(tf.matmul(input_x, fc1_w) + fc1_b)

    fc2_w = tf.Variable(tf.truncated_normal([64, 16]))
    fc2_b = tf.Variable(tf.constant(0.1, shape=[16]))
    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_w) + fc2_b)

    fc3_w = tf.Variable(tf.truncated_normal([16, 2]))
    fc3_b = tf.Variable(tf.constant(0.1, shape=[2]))
    prob = tf.matmul(fc2, fc3_w) + fc3_b

    l2 = tf.nn.l2_loss(fc1_w) + tf.nn.l2_loss(fc1_b) + \
         tf.nn.l2_loss(fc2_w) + tf.nn.l2_loss(fc2_b) + \
         tf.nn.l2_loss(fc3_w) + tf.nn.l2_loss(fc3_b)

    loss = l2*0.005 + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prob, labels=input_y))
    train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)

    pred = tf.argmax(tf.nn.softmax(prob), output_type=tf.int32, axis=1)
    eq = tf.equal(pred, y_ext)
    c = tf.cast(eq, tf.int32)
    acc = tf.reduce_sum(c)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, 'model/tfmodel.ckpt')
        pro = sess.run(prob, feed_dict={input_x:test_x})
    
    c = []
    for i in range(len(pro)):
        if pro[i][0]>pro[i][1]:
            c.append(0)
        else:
            c.append(1)
    return c

if __name__=='__main__':
    test_x = read_data(sys.argv[1])
    c = test(test_x)
        
    with open(sys.argv[2], 'w') as f:
        f.write('id,label\n')
        for i in range(len(c)):
            f.write(str(i+1)+','+str(c[i])+'\n')