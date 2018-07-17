"""
@Project   : DuReader
@Module    : qa_model.py
@Author    : Deco [deco@cubee.com]
@Created   : 7/16/18 11:54 AM
@Desc      : 
"""

import tensorflow as tf

from embed2.basic_rnn import rnn


class QaModel:
    def __init__(self, vocab):
        self.vocab = vocab
        self.p = tf.placeholder(tf.int32, [None, None])
        self.q = tf.placeholder(tf.int32, [None, None])
        self.p_length = tf.placeholder(tf.int32, [None])
        self.q_length = tf.placeholder(tf.int32, [None])
        self.hidden_size = 16

    def _embed(self):
        """
        The embedding layer, question and passage share embeddings
        只要词是一样的，embedding就是一样的
        """
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            # 此处指定使用cpu
            # 第一次建立这个variable_scope, 而不是reuse
            # reuse中最重要的是模型中的trainable variable的复用
            self.word_embeddings = tf.get_variable(
                'word_embeddings',
                shape=(self.vocab.size(), self.vocab.embed_dim),
                initializer=tf.constant_initializer(self.vocab.embeddings),
                # 把已经初始化好的embedding传过来
                trainable=True
            )
            # 生成variable，一般是可训练的；如果不可训练，就是始终使用pretrained
            # embedding
            self.p_emb = tf.nn.embedding_lookup(self.word_embeddings, self.p)
            # paragraph
            self.q_emb = tf.nn.embedding_lookup(self.word_embeddings, self.q)
            # question
            # 把p和q中代表词的int转换为embedding

    def _encode(self):
        """
        句子矩阵或者句向量，此处用bi-lstm实现
        Employs two Bi-LSTMs to encode passage and question separately
        p和q都做了encode，之后又合并了，本质上不是encode，是预处理
        """
        with tf.variable_scope('passage_encoding'):
            self.sep_p_encodes, _ = rnn('bi-lstm', self.p_emb, self.p_length,
                                        self.hidden_size)
            # self.p_emb是二维的，一个维度是batch size，另一个维度是多个sample中最长的
            # p的长度；self.p_length是一维的，长度是batch size，具体内容是各个sample的
            # p的长度；hidden_size是这个rnn中lstm的hidden unit数目，是可以调参的
            # rnn的返回值既有output，也有hidden state，此处只记录output
            # 其实是从一个矩阵变换到了另一个矩阵
        with tf.variable_scope('question_encoding'):
            self.sep_q_encodes, _ = rnn('bi-lstm', self.q_emb, self.q_length,
                                        self.hidden_size)
            # self.sep_p_encodes, self.sep_q_encodes都是句子矩阵
