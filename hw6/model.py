import tensorflow as tf
import config as cfg

def fc(input_, shape_, act=None):
    w = tf.Variable(tf.truncated_normal(shape_, stddev=0.1))
    b = tf.Variable(tf.constant(0.0, shape=[shape_[1]]))
    if act is not None:
        return act(tf.matmul(input_, w) + b)
    else:
        return tf.matmul(input_, w) + b

def conv(input_, shape_, act=None):
    w = tf.Variable(tf.truncated_normal(shape_, stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[shape_[3]]))
    if act is not None:
        return act(tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='SAME') + b)
    else:
        return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='SAME') + b

def pool(input_):
    return tf.nn.max_pool(input_, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

class FeatureExtractor():
    def __init__(self):
        self.image = tf.placeholder(tf.float32, [None, 784])

        fc1 = fc(self.image, [784, 256], tf.nn.relu)
        #fc2 = fc(fc1, [512, 256], tf.nn.relu)
        fc3 = fc(fc1, [256, 128], tf.nn.relu)

        self.flat = fc3

        fc11 = fc(self.flat, [128, 256], tf.nn.relu)
        #fc22 = fc(fc11, [256, 512], tf.nn.relu)
        fc33 = fc(fc11, [256, 784], tf.nn.sigmoid)
        
        self.reconstruct = tf.reshape(fc33, [-1, 28, 28])

        self.loss = tf.reduce_mean(tf.square(self.image - fc33))
        trainer = tf.train.AdamOptimizer(cfg.learning_rate)
        self.train_step = trainer.minimize(self.loss)

        self.saver = tf.train.Saver()
    
    def init(self, sess):
        sess.run(tf.global_variables_initializer())
        print('Variables initialized')
    
    def load(self, sess):
        ckpt = tf.train.get_checkpoint_state('.')
        self.saver.restore(sess, ckpt.model_checkpoint_path)
        print('Model loaded from', ckpt.model_checkpoint_path, '.')
    
    def save(self, sess):
        self.saver.save(sess, './model.ckpt')
        print('Model saved.')
    
    def train(self, sess, images):
        loss, _ = sess.run([self.loss, self.train_step], feed_dict={self.image:images})
        return loss
    
    def test(self, sess, images):
        images_ = sess.run(self.reconstruct, feed_dict={self.image:images})
        return images_

    def get_feature(self, sess, images):
        feature = sess.run(self.flat, feed_dict={self.image:images})
        return feature
        