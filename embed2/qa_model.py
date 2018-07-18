"""
@Project   : DuReader
@Module    : qa_model.py
@Author    : Deco [deco@cubee.com]
@Created   : 7/16/18 11:54 AM
@Desc      :
"""

import tensorflow as tf

from embed2.basic_rnn import rnn
from embed2.match_layer import MatchLSTMLayer
from embed2.match_layer import AttentionFlowMatchLayer


class QaModel:
    def __init__(self, vocab):
        self.vocab = vocab
        self.p = tf.placeholder(tf.int32, [None, None])
        self.q = tf.placeholder(tf.int32, [None, None])
        self.p_length = tf.placeholder(tf.int32, [None])
        self.q_length = tf.placeholder(tf.int32, [None])
        self.hidden_size = 16
        self.algo = 'MLSTM'

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
            # p的长度，提供这个参数可以加快程序运行的速度；
            # hidden_size是这个rnn中lstm的hidden unit数目，是可以调参的，默认
            # cell unit数目等于hidden unit数目
            # rnn的返回值既有output，也有hidden state，此处只记录output
            # 其实是从一个矩阵变换到了另一个矩阵
        with tf.variable_scope('question_encoding'):
            self.sep_q_encodes, _ = rnn('bi-lstm', self.q_emb, self.q_length,
                                        self.hidden_size)
            # self.sep_p_encodes, self.sep_q_encodes都是句子矩阵

    def _match(self):
        """
        The core of RC model, get the question-aware passage
        encoding with either BIDAF or MLSTM
        The attention process
        文档的加权句向量，权重由问句决定
        """
        if self.algo == 'MLSTM':
            match_layer = MatchLSTMLayer(self.hidden_size)
        elif self.algo == 'BIDAF':
            match_layer = AttentionFlowMatchLayer(self.hidden_size)
        else:
            raise NotImplementedError(
                'The algorithm {} is not implemented.'.format(self.algo))
        self.match_p_encodes, _ = match_layer.match(self.sep_p_encodes,
                                                    self.sep_q_encodes,
                                                    self.p_length,
                                                    self.q_length)
        # 所谓的attention的过程，就是把两个矩阵用神经网络的方式合并成一个矩阵
        # 中间过程要拿到attention分布，这个分布与q有关，然后把这个分布施加在p上，得到新的
        # 矩阵; p后来要被pointer net来用，所以p是主要的，q是次要的，p和q是有关系的，
        # 但这种关系如何建模，一般是采取attention的方式来建模，不管MLSM还是DIDAF，
        # 用的都是这种方式，只不过细节不同
        # 只记录lstm的outputs，不记录hidden states
