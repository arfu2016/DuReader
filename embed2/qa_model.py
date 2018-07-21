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
from embed2.pointer_net import PointerNetDecoder


class QaModel:
    def __init__(self, vocab):
        self.vocab = vocab
        self.p = tf.placeholder(tf.int32, [None, None])
        self.q = tf.placeholder(tf.int32, [None, None])
        self.p_length = tf.placeholder(tf.int32, [None])
        self.q_length = tf.placeholder(tf.int32, [None])
        self.hidden_size = 16
        self.algo = 'MLSTM'
        self.start_label = tf.placeholder(tf.int32, [None])
        self.end_label = tf.placeholder(tf.int32, [None])
        self.weight_decay = 0.01

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
            # 词向量是300维，self.hidden_size是150，由于'bi-lstm'，最终还是300维
        with tf.variable_scope('question_encoding'):
            self.sep_q_encodes, _ = rnn('bi-lstm', self.q_emb, self.q_length,
                                        self.hidden_size)
            # self.sep_p_encodes, self.sep_q_encodes都是句子矩阵
            # 对于句子矩阵，因为维度还是词的个数*hidden_size，所以本质上还是词向量，是变换过的词向量

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
        # 在利用问句词向量加权之后，最终得到的还是文档词向量，还是一个矩阵

    def _fuse(self):
        """
        Employs Bi-LSTM again to fuse the context information after match layer
        得到新的rnn
        """
        with tf.variable_scope('fusion'):
            self.fuse_p_encodes, _ = rnn('bi-lstm', self.match_p_encodes,
                                         self.p_length,
                                         self.hidden_size, layer_num=1)
            # attention之后，再用bi-lstm做一次矩阵变换，是真正的encode
            # 此处lstm的layer_num是可调的（layer_num大于1的话，似乎代码有bug）
            # 同样只记录outputs，作为pointer net选择的对象
            # 再用一个新的bi-lstm做一次变换，得到的还是文档词向量，还是300维

    def _decode(self):
        decoder = PointerNetDecoder(self.hidden_size)
        self.start_probs, self.end_probs = decoder.decode(
            self.fuse_p_encodes, self.sep_q_encodes)
        # 在普通的seq2seq model中，decoder的输入是一个词的词向量，得到输出的unit维度向量，
        # 然后通过softmax转变成很多个分类。
        # 在seq2seq中加入attention以后，decoder的输入除了一个词的词向量之外，还多了
        # 一个词向量，然后得到输出的unit维度向量，通过softmax转变成很多个分类。
        # 在pointer net中，decoder的输入其实是一个文档词向量矩阵，与作为hidden state的
        # 问句词向量矩阵互作后，得到的输出是word维度向量，直接就反映了各个word分类的概率

    def _compute_loss(self):
        """
        The loss function
        计算损失函数就为了之后的参数优化，本质上是为了back propagation
        此处损失函数用的还是cross entropy，既考虑start loss，也考虑end loss，把
        二者结合起来。在做拟合和推测时，都用到start prob和end prob，但处理方法是不同的
        还有一种处理方法，更类似加强学习，就是拟合的结果不和answer相比，而是把目标
        函数变成start_prob*end_prob的最大化
        """

        def sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):
            """
            negative log likelyhood loss
            """
            with tf.name_scope(scope, "log_loss"):
                # 此处是name_scope
                labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
                # labels是具体位置的序号，此处要one hot encoding
                losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)
                # cross entropy的公式, + epsilon是为了处理probs为0的情况
            return losses

        self.start_loss = sparse_nll_loss(probs=self.start_probs,
                                          labels=self.start_label)
        self.end_loss = sparse_nll_loss(probs=self.end_probs,
                                        labels=self.end_label)
        # 要算两个cross entropy的loss，start和end各算一次
        self.all_params = tf.trainable_variables()
        self.loss = tf.reduce_mean(tf.add(self.start_loss, self.end_loss))
        # 优化的目标函数：两个loss的加和求最小值
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
                # 做l2 regularization，使得拟合的weight不至于太大
            self.loss += self.weight_decay * l2_loss
            # self.weight_decay就是做l2 regularization时前面的那个系数，
            # 用来控制正则化的程度
