import tensorflow as tf

import config as cfg

def dbg_show_all_variable():
    print('[Trainable Variables]=========================')
    variables = [v for v in tf.trainable_variables()]
    for k in variables:
        print("Variable: ", k)
    print('[Trainable Variables]=========================')

def lstm_cell(kp, size):
    #cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(size)
    cell = tf.contrib.rnn.GRUCell(size)
    #cell = tf.contrib.rnn.BasicLSTMCell(size)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=kp)
    return cell

class RetrievalModel():
    def __init__(self):
        self.input_mfcc = tf.placeholder(tf.float32, [None, cfg.ENCODER_MAX_TIME_STEP, 39])
        self.input_mfcc_len = tf.placeholder(tf.int32, [None])
        self.input_caption = tf.placeholder(tf.int32, [None, cfg.DECODER_MAX_TIME_STEP])
        self.input_caption_len = tf.placeholder(tf.int32, [None])
        self.input_label = tf.placeholder(tf.float32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        # encode mfcc:
        with tf.variable_scope('mfcc_encoder'):
            mfcc_cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell(self.keep_prob, cfg.RNN_SIZE) for _ in range(cfg.RNN_LAYER)])
            mfcc_cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell(self.keep_prob, cfg.RNN_SIZE) for _ in range(cfg.RNN_LAYER)])
            mfcc_output, mfcc_state = tf.nn.bidirectional_dynamic_rnn(
                mfcc_cell_fw, 
                mfcc_cell_bw, 
                self.input_mfcc,
                sequence_length=self.input_mfcc_len, 
                dtype=tf.float32
            )
            mfcc_output = tf.concat(mfcc_output, 2)

            batch_range = tf.range(tf.shape(mfcc_output)[0])
            indices = tf.stack([batch_range, self.input_mfcc_len-1], axis=1)
            mfcc_output = tf.gather_nd(mfcc_output, indices)

        # embedding:
        embedding = tf.Variable(tf.truncated_normal([cfg.DICT_SIZE, cfg.EMBED_DIM], stddev=0.01))
        caption_input = tf.nn.embedding_lookup(embedding, self.input_caption)

        # encode caption:
        with tf.variable_scope('caption_encoder'):
            caption_cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell(self.keep_prob, cfg.RNN_SIZE) for _ in range(cfg.RNN_LAYER)])
            caption_cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell(self.keep_prob, cfg.RNN_SIZE) for _ in range(cfg.RNN_LAYER)])
            caption_output, caption_state = tf.nn.bidirectional_dynamic_rnn(
                caption_cell_fw, 
                caption_cell_bw, 
                caption_input,
                sequence_length=self.input_caption_len, 
                dtype=tf.float32
            )
            caption_output = tf.concat(caption_output, 2)

            batch_range = tf.range(tf.shape(caption_output)[0])
            indices = tf.stack([batch_range, self.input_caption_len-1], axis=1)
            caption_output = tf.gather_nd(caption_output, indices)

        dot = mfcc_output * caption_output
        dot = tf.reduce_sum(dot, axis=-1)
        self.pred = tf.nn.sigmoid(dot)
        
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_label, logits=dot)
        self.loss = tf.reduce_mean(self.loss)

        trainer = tf.train.AdamOptimizer(cfg.LEARNING_RATE)
        self.train_step = trainer.minimize(self.loss)
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter('log/')
        dbg_show_all_variable()
    
    def train(self, sess, mfcc, mfcc_len, caption, caption_len, label):
        loss, _ = sess.run([self.loss, self.train_step], feed_dict={
            self.input_mfcc:mfcc,
            self.input_mfcc_len:mfcc_len,
            self.input_caption:caption,
            self.input_caption_len:caption_len,
            self.input_label:label,
            self.keep_prob:cfg.KEEP_PROB
        })
        return loss

    def valid(self, sess, mfcc, mfcc_len, caption, caption_len, label):
        loss = sess.run(self.loss, feed_dict={
            self.input_mfcc:mfcc,
            self.input_mfcc_len:mfcc_len,
            self.input_caption:caption,
            self.input_caption_len:caption_len,
            self.input_label:label,
            self.keep_prob:1.0
        })
        return loss

    def predict(self, sess, mfcc, mfcc_len, caption, caption_len):
        pred = sess.run(self.pred, feed_dict={
            self.input_mfcc:mfcc,
            self.input_mfcc_len:mfcc_len,
            self.input_caption:caption,
            self.input_caption_len:caption_len,
            self.keep_prob:1.0
        }) 
        return pred

    def init(self, sess):
        sess.run(tf.global_variables_initializer())
        print('Variables initialized')
    
    def load(self, sess, name):
        self.saver.restore(sess, './'+name)
        print('Model loaded')
    
    def save(self, sess, name):
        self.saver.save(sess, './'+name+'.ckpt')
        print('Model saved')
    
    def add_global_step(self, sess):
        sess.run(self.global_step.assign_add(1))
    
    def add_summary(self, sess, train_loss, valid_accu):
        summary = tf.Summary()
        summary.value.add(tag='Loss/Training Loss', simple_value=train_loss)
        summary.value.add(tag='Accuracy/Validation Accuracy', simple_value=valid_accu)
        self.summary_writer.add_summary(summary, sess.run(self.global_step))
        self.summary_writer.flush()
        print('summary written')

if __name__=='__main__':
    model = RetrievalModel()