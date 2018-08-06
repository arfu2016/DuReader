"""
@Project   : DuReader
@Module    : match_layer.py
@Author    : Deco [deco@cubee.com]
@Created   : 7/26/18 10:15 AM
@Desc      : This module implements the core layer of Match-LSTM and BiDAF
"""

import tensorflow as tf
import tensorflow.contrib as tc


class MatchLSTMAttnCell(tc.rnn.LSTMCell):
    """
    Implements the Match-LSTM attention cell
    继承LSTMCell就表明是单层的cell
    """
    def __init__(self, num_units, context_to_attend):
        super(MatchLSTMAttnCell, self).__init__(num_units, state_is_tuple=True)
        self.context_to_attend = context_to_attend
        self.fc_context = tc.layers.fully_connected(
            self.context_to_attend,
            num_outputs=self._num_units,
            activation_fn=None)

    def __call__(self, inputs, state, scope=None):
        (c_prev, h_prev) = state
        with tf.variable_scope(scope or type(self).__name__):
            ref_vector = tf.concat([inputs, h_prev], -1)
            G = tf.tanh(
                self.fc_context + tf.expand_dims(
                    tc.layers.fully_connected(
                        ref_vector,
                        num_outputs=self._num_units,
                        activation_fn=None), 1))
            logits = tc.layers.fully_connected(
                G, num_outputs=1, activation_fn=None)
            scores = tf.nn.softmax(logits, 1)
            attended_context = tf.reduce_sum(
                self.context_to_attend * scores, axis=1)
            new_inputs = tf.concat(
                [inputs, attended_context, inputs - attended_context,
                 inputs * attended_context], -1)
            return super(
                MatchLSTMAttnCell, self).__call__(new_inputs, state, scope)
        # 输入的passage词向量矩阵，经过了question词向量矩阵的变换，得到新的词向量矩阵


class MatchLSTMLayer(object):
    """
    Implements the Match-LSTM layer, which attend to the question
    dynamically in a LSTM fashion.
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def match(self, passage_encodes, question_encodes, p_length, q_length):
        """
        Match the passage_encodes with question_encodes using Match-LSTM
        algorithm
        """
        with tf.variable_scope('match_lstm'):
            cell_fw = MatchLSTMAttnCell(self.hidden_size, question_encodes)
            cell_bw = MatchLSTMAttnCell(self.hidden_size, question_encodes)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, inputs=passage_encodes,
                sequence_length=p_length, dtype=tf.float32)
            match_outputs = tf.concat(outputs, 2)
            state_fw, state_bw = state
            c_fw, h_fw = state_fw
            c_bw, h_bw = state_bw
            match_state = tf.concat([h_fw, h_bw], 1)
        return match_outputs, match_state


class AttentionFlowMatchLayer(object):
    """
    Implements the Attention Flow layer,
    which computes Context-to-question Attention and question-to-context
    Attention
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def match(self, passage_encodes, question_encodes, p_length, q_length):
        """
        Match the passage_encodes with question_encodes using Attention Flow
        Match algorithm
        """
        with tf.variable_scope('bidaf'):
            #  自己用矩阵乘法和softmax来实现神经网络模型
            sim_matrix = tf.matmul(passage_encodes, question_encodes,
                                   transpose_b=True)
            context2question_attn = tf.matmul(tf.nn.softmax(sim_matrix, -1),
                                              question_encodes)
            b = tf.nn.softmax(tf.expand_dims(tf.reduce_max(sim_matrix, 2), 1),
                              -1)
            question2context_attn = tf.tile(tf.matmul(b, passage_encodes),
                                         [1, tf.shape(passage_encodes)[1], 1])
            concat_outputs = tf.concat(
                [passage_encodes, context2question_attn,
                 passage_encodes * context2question_attn,
                 passage_encodes * question2context_attn],
                -1)
            return concat_outputs, None
        # 输出None是为了和MLSTM接口的统一
        # 整体上做的就是把passage和question两个矩阵合并成一个矩阵