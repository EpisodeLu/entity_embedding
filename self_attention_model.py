import tensorflow as tf
import numpy as np


class Config(object):
    embedding_dim = 128
    seq_length = 25
    num_classes = 3
    vocab_size = 6282
    trainable = True

    num_layers = 2
    hidden_dim = 128
    rnn = 'lstm'

    dropout_keep_prob = 0.8
    lr = 1e-3
    batch_size = 128
    num_epochs = 10

    print_per_batch = 100
    save_per_batch = 10

    da = 64
    r = 1


class SelfAttentionModel(object):
    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(tf.int32, shape=[None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.att_rnn()

    def att_rnn(self):
        def lstm_cell():
            return tf.contrib.rnn.LSTMCell(self.config.hidden_dim, state_is_tuple=True)

        def gru_cell():
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout():
            if self.config.rnn == 'lstm':
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim],
                                        trainable=self.config.trainable)  # [vocab_size, dim]
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)  # [batch_size, maxlen, dim]

        with tf.name_scope("rnn"):
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            self._outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs,
                                                 dtype=tf.float32)  # _outputs.shape=[batch_size, maxlen, dim]

        with tf.name_scope("attention_embedding"):
            # s = self._outputs.get_shape().as_list()
            s = tf.shape(self._outputs)
            # W_s1 = tf.get_variable('W_s1', shape=[[2], self.config.da],
            #                        initializer=tf.truncated_normal_initializer(stddev=0.5),
            #                        dtype=tf.float32)
            # W_s2 = tf.get_variable('W_s2', shape=[self.config.da, self.config.r],
            #                        initializer=tf.truncated_normal_initializer(stddev=0.5),
            #                        dtype=tf.float32)
            output_re = tf.reshape(self._outputs, shape=[-1, self.config.hidden_dim])
            fc = tf.layers.dense(output_re, self.config.da, name='fc1')
            # fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc2 = tf.layers.dense(fc, self.config.r, name='fc2')
            fc2_re = tf.reshape(fc2, shape=[s[0], -1, self.config.r])
            fc2_trans = tf.map_fn(lambda inp: tf.transpose(inp), fc2_re)
            # self.att = tf.map_fn(lambda inp: tf.nn.softmax(inp, axis=-1), fc2_trans)
            # entity_embs = tf.matmul(self.att, self._outputs)
            self.att = tf.map_fn(lambda inp: tf.nn.softmax(inp, axis=-1), fc2_trans)
            entity_embs = tf.matmul(self.att, self._outputs)
            self.entity_embs = tf.reshape(entity_embs, shape=[s[0], self.config.hidden_dim])

        with tf.name_scope("score"):
            full_c = tf.layers.dense(self.entity_embs, self.config.hidden_dim, name='fc3')  # [batch_size, hidden_dim]
            full_c = tf.contrib.layers.dropout(full_c, self.keep_prob)
            full_c = tf.nn.relu(full_c)

            self.logits = tf.layers.dense(full_c, self.config.num_classes, name='fc4')  # [batch_size, num_classes]
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope("optimize"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)

            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(self.loss)

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))