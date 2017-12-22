import tensorflow as tf

import config

class ModelMatrixFactorization():
    def __init__(self):
        self.target = tf.placeholder(tf.float32, [config.N_USER+1, config.N_MOVIE+1])
        self.mask = tf.placeholder(tf.float32, [config.N_USER+1, config.N_MOVIE+1])

        self.user_embed = tf.Variable(tf.truncated_normal([config.N_USER+1, config.EMBED_DIM], stddev=0.1))
        self.movie_embed = tf.Variable(tf.truncated_normal([config.N_MOVIE+1, config.EMBED_DIM], stddev=0.1))
        if config.USING_BIAS:
            self.user_bias = tf.Variable(tf.constant(0.0, shape=[config.N_USER+1, 1]))
            self.movie_bias = tf.Variable(tf.constant(0.0, shape=[1, config.N_MOVIE+1]))

        self.rating = tf.matmul(self.user_embed, self.movie_embed, transpose_b=True)
        if config.USING_BIAS:
            self.rating = tf.add(self.rating, self.user_bias)
            self.rating = tf.add(self.rating, self.movie_bias)

        self.loss = tf.reduce_sum(tf.square(self.target-self.rating)*self.mask)

        trainer = tf.train.AdamOptimizer(config.LEARNING_RATE)
        self.train_step = trainer.minimize(self.loss)

        self.saver = tf.train.Saver()
    
    def train(self, sess, target, mask):
        loss, _ = sess.run([self.loss, self.train_step], feed_dict={self.target:target, self.mask:mask})
        return loss

    def predict(self, sess):
        rating = sess.run(self.rating)
        return rating

    def error(self, sess, target, mask):
        loss = sess.run(self.loss, feed_dict={self.target:target, self.mask:mask})
        return loss
    
    def init(self, sess):
        sess.run(tf.global_variables_initializer())
        print('Variables initialized')

    def load(self, sess):
        ckpt = tf.train.get_checkpoint_state('.')
        self.saver.restore(sess, ckpt.model_checkpoint_path)
        print('Model loaded from', ckpt.model_checkpoint_path)

    def save(self, sess):
        self.saver.save(sess, './model.ckpt')
        print('Model saved')