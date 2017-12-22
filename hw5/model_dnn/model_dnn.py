import tensorflow as tf
import numpy as np

import config

def fc(input, shape_, act=None):
    w = tf.Variable(tf.truncated_normal(shape_, stddev=0.1))
    b = tf.Variable(tf.constant(0.0, shape=[shape_[1]]))
    if act is not None:
        return act(tf.matmul(input, w) + b)
    else:
        return tf.matmul(input, w) + b

class Model():
    def __init__(self):
        self.input_userid = tf.placeholder(tf.int32, [None])
        self.input_movieid = tf.placeholder(tf.int32, [None])
        self.input_rating = tf.placeholder(tf.float32, [None])
        self.input_category = tf.placeholder(tf.float32, [None, config.N_CATEGORY])
        self.input_userattr = tf.placeholder(tf.float32, [None, 3])

        self.keep_prob = tf.placeholder(tf.float32)

        movie_embedding = tf.Variable(tf.truncated_normal([config.N_MOVIE+1, config.EMBED_DIM], stddev=0.1))
        user_embedding = tf.Variable(tf.truncated_normal([config.N_USER+1, config.EMBED_DIM], stddev=0.1))

        user_input = tf.nn.embedding_lookup(user_embedding, self.input_userid)
        movie_input = tf.nn.embedding_lookup(movie_embedding, self.input_movieid)

        user_input = tf.concat([user_input, self.input_category, self.input_userattr], axis=-1)
        movie_input = tf.concat([movie_input, self.input_category, self.input_userattr], axis=-1)

        user_input = fc(user_input, [config.EMBED_DIM+config.N_CATEGORY+3, 64], tf.nn.relu)
        movie_input = fc(movie_input, [config.EMBED_DIM+config.N_CATEGORY+3, 64], tf.nn.relu)

        user_input = tf.nn.dropout(user_input, keep_prob=self.keep_prob)
        movie_input = tf.nn.dropout(movie_input, keep_prob=self.keep_prob)

        user_input = fc(user_input, [64, 16])
        movie_input = fc(movie_input, [64, 16])
        print(user_input)
        print(movie_input)

        user_bias_embed = tf.Variable(tf.constant(0.0, shape=[config.N_USER+1, 1]))
        movie_bias_embed = tf.Variable(tf.constant(0.0, shape=[config.N_MOVIE+1, 1]))

        user_bias = tf.nn.embedding_lookup(user_bias_embed, self.input_userid)
        movie_bias = tf.nn.embedding_lookup(movie_bias_embed, self.input_movieid)
        print(user_bias)
        print(movie_bias)
        
        score = tf.reduce_sum(user_input*movie_input, axis=1, keep_dims=True) + user_bias + movie_bias
        score = tf.reshape(score, [-1])
        print(score)

        self.pred = score

        self.loss = tf.reduce_mean(tf.square(self.pred-self.input_rating))
        trainer = tf.train.AdamOptimizer(config.LEARNING_RATE)
        self.train_step = trainer.minimize(self.loss)

        self.saver = tf.train.Saver()
    
    def train(self, sess, userid, movieid, category, userattr, rating):
        loss, _ = sess.run([self.loss, self.train_step], feed_dict={self.input_userid:userid, self.input_movieid:movieid, self.input_category:category, self.input_userattr:userattr, self.input_rating:rating, self.keep_prob:config.KEEP_PROB})
        return np.sqrt(loss)

    def valid(self, sess, userid, movieid, category, userattr, rating):
        loss = sess.run(self.loss, feed_dict={self.input_userid:userid, self.input_movieid:movieid, self.input_category:category, self.input_userattr:userattr, self.input_rating:rating, self.keep_prob:1.0})
        return np.sqrt(loss)

    def predict(self, sess, userid, movieid, category, userattr):
        pred = sess.run(self.pred, feed_dict={self.input_userid:userid, self.input_movieid:movieid, self.input_category:category, self.input_userattr:userattr, self.keep_prob:1.0})
        return pred

    def load(self, sess):
        ckpt = tf.train.get_checkpoint_state('.')
        self.saver.restore(sess, ckpt.model_checkpoint_path)
        print('Model loaded from', ckpt.model_checkpoint_path, '.')
    
    def save(self, sess):
        self.saver.save(sess, './model.ckpt')
        print('Model saved.')
    
    def init(self, sess):
        sess.run(tf.global_variables_initializer())
        print('Variables initialized.')