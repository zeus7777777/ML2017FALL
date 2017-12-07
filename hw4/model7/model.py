import tensorflow as tf

import config

class Model():
    def fc(input, shape_, act=None):
        w = tf.Variable(tf.truncated_normal(shape_, stddev=0.1))
        b = tf.Variable(tf.constant(0.0, shape=[shape_[1]]))
        if act is not None:
            return act(tf.matmul(input, w) + b)
        else:
            return tf.matmul(input, w) + b
    
    def __init__(self):
        self.input_x = tf.placeholder(tf.float32, [None, config.MAX_SEQ_LEN, config.WORD_DIM])
        self.input_y = tf.placeholder(tf.int64, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(config.RNN_SIZE)
        lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(lstm_cell_fw, output_keep_prob=self.keep_prob)
        lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell_fw]*config.RNN_LAYER)
        state_fw = lstm_cell_fw.zero_state(config.BATCH_SIZE, tf.float32)

        lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(config.RNN_SIZE)
        lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(lstm_cell_bw, output_keep_prob=self.keep_prob)
        lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell_bw]*config.RNN_LAYER)
        state_bw = lstm_cell_bw.zero_state(config.BATCH_SIZE, tf.float32)

        rnn_input = tf.nn.dropout(self.input_x, self.keep_prob)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, rnn_input, dtype=tf.float32)
        outputs = tf.concat(outputs, axis=2)
        rnn_output1 = outputs[:, 0, :]
        rnn_output2 = outputs[:, config.MAX_SEQ_LEN-1, :]
        rnn_output = tf.concat([rnn_output1, rnn_output2], axis=1)
        rnn_output = tf.reshape(rnn_output, [config.BATCH_SIZE, -1])
        
        fc1 = Model.fc(rnn_output, [config.RNN_SIZE*4, 1024], tf.nn.relu)
        fc2 = Model.fc(fc1, [1024, 2])

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=fc2))
        trainer = tf.train.AdamOptimizer(config.LEARNING_RATE)
        self.train_step = trainer.minimize(self.loss)

        self.score = tf.nn.softmax(fc2)[:, 1]
        self.pred = tf.reshape(tf.argmax(fc2, axis=1), [-1])
        self.accu = tf.reduce_mean(tf.cast(tf.cast(tf.equal(self.pred, self.input_y), tf.int32), tf.float32))

        self.saver = tf.train.Saver()
    
    def train(self, sess, feed_x, feed_y):
        loss_, __ = sess.run([self.loss, self.train_step], feed_dict={self.input_x:feed_x, self.input_y:feed_y, self.keep_prob:config.KEEP_PROB})

        return loss_

    def accuracy(self, sess, feed_x, feed_y):
        accu_ = sess.run(self.accu, feed_dict={self.input_x:feed_x, self.input_y:feed_y, self.keep_prob:1.0})

        return accu_
    
    def predict(self, sess, feed_x):
        pred_ = sess.run(self.pred, feed_dict={self.input_x:feed_x, self.keep_prob:1.0})

        return pred_

    def confidence(self, sess, feed_x):
        score_ = sess.run(self.score, feed_dict={self.input_x: feed_x, self.keep_prob:1.0})

        return score_

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