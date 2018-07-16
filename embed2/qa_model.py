"""
@Project   : DuReader
@Module    : qa_model.py
@Author    : Deco [deco@cubee.com]
@Created   : 7/16/18 11:54 AM
@Desc      : 
"""

import tensorflow as tf


class QaModel:
    def __init__(self, vocab):
        self.vocab = vocab
        self.p = tf.placeholder(tf.int32, [None, None])
        self.q = tf.placeholder(tf.int32, [None, None])

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
