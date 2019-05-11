import tensorflow as tf
import numpy as np

BIDIR = True
LAYERS = 2
HIDDEN = 512
LR = 1e-3
class Graph():
    def last_relevant(self, output, length):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size)*max_length+(length-1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant
    def length(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    def __init__(self, max_len, is_train=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs = tf.placeholder(
                    tf.float32, (None, max_len, 768), 'inputs')
            self.inputs_utt = tf.placeholder(
                    tf.float32, (None,), 'inputs_utt')
            self.outputs = tf.placeholder( tf.float32, (None, ), 'outputs')
            self.inputs_len = self.length(self.inputs)
                    
            with tf.name_scope('rnn'):
                cell = tf.nn.rnn_cell.BasicLSTMCell
                if BIDIR:
                    cells_fw = []
                    cells_bw = []
                    for _ in range(LAYERS):
                        cell_fw = cell(HIDDEN)
                        cell_bw = cell(HIDDEN)
                        if is_train:
                            cell_fw =tf.nn.rnn_cell.DropoutWrapper(cell_fw,
                                    0.5,0.5,True)
                            cell_bw =tf.nn.rnn_cell.DropoutWrapper(cell_bw,
                                    0.5,0.5,True)
                        cells_fw.append(cell_fw)
                        cells_bw.append(cell_bw)
                    cells_fw = tf.nn.rnn_cell.MultiRNNCell(cells_fw)
                    cells_bw = tf.nn.rnn_cell.MultiRNNCell(cells_bw)
                    rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                                        cells_fw,
                                        cells_bw,
                                        inputs = self.inputs,
                                        sequence_length = self.inputs_len,
                                        dtype = tf.float32)
                    rnn_outputs = tf.concat(rnn_outputs, 2)

            with tf.name_scope('project'):
                last = self.last_relevant(rnn_outputs, self.inputs_len)
                last = tf.concat([last, self.inputs_utt[:,None]],1)
                logits = tf.layers.dense(last, 256, tf.nn.relu)
                logits = tf.layers.dropout(logits, 0.5, training=is_train)
                logits = tf.layers.dense(last, 256, tf.nn.relu)
                logits = tf.layers.dropout(logits, 0.5, training=is_train)
                self.preds = tf.layers.dense(logits, 1)
            self.loss = tf.reduce_mean(tf.square(self.outputs-self.preds))
            self.optimizer = tf.train.RMSPropOptimizer(LR)
            self.train_op =  self.optimizer.minimize(self.loss)
            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()
                            


