"""
@Project   : DuReader
@Module    : sentence_similar.py
@Author    : Deco [deco@cubee.com]
@Created   : 4/28/18 10:52 AM
@Desc      : 
"""
from collections import defaultdict
import logging
import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


tf.logging.set_verbosity(tf.logging.WARN)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
# 1压制info，只输出warn以上级别


class SentenceTarget:

    meaning2question = defaultdict(list)
    question2meaning = dict()
    meaning_embedding = defaultdict(float)
    meaning_weight = defaultdict(int)

    def __init__(self):
        self.logger = logging.getLogger("vector")
        self.set_logger()
        self.embed = None
        self.build_embed()
        self.std_embedding = None
        self.group_embedding = None
        self.meaning_list = None
        for st in self.question2meaning:
            self.append_embedding(st)
        # temp_embedding = list(self.meaning_embedding.values())
        # temp_embedding2 = [embedding.tolist()[0]
        #                    for embedding in temp_embedding]
        # self.logger.info(temp_embedding2)
        # self.group_embedding = np.array(temp_embedding2)

    def set_logger(self):
        # self.logger.setLevel(logging.INFO)
        self.logger.setLevel(logging.ERROR)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)

    def append_tpl_by_key(self, meaning, st):
        if st in self.question2meaning:
            self.logger.warning(
                '{} already exists in the templates'.format(st))
        else:
            self.meaning2question[meaning].append(st)
            self.question2meaning.update({st: meaning})
            self.append_embedding(st)

    def append_tpl_by_value(self, ref, st):
        meaning = self.question2meaning.get(ref, None)
        if meaning is not None:
            self.append_tpl_by_key(meaning, st)
        else:
            self.logger.warning(
                '{} does not exist in the templates'.format(ref))

    def target_select(self, st):
        def run_sts_benchmark(session):
            """Returns the similarity scores"""
            emba, embb, scores = session.run(
                [sts_encode1, sts_encode2, sim_scores],
                feed_dict={
                    sts_input1: group_question,
                    sts_std: self.group_embedding
                })
            return scores

        group_question = [st] * len(self.meaning2question)
        sts_input1 = tf.placeholder(tf.string, shape=(None))
        sts_std = tf.placeholder(tf.float32, shape=(None, None))

        # For evaluation we use exactly normalized rather than
        # approximately normalized.
        sts_encode1 = tf.nn.l2_normalize(self.embed(sts_input1))
        sts_encode2 = tf.nn.l2_normalize(sts_std)

        sim_scores = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2),
                                   axis=1)
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())
            scores = run_sts_benchmark(session)
        return scores

    def append_embedding(self, st, weight=1):
        embedding = self.embed([st])
        with tf.Session() as session:
            session.run(
                [tf.global_variables_initializer(), tf.tables_initializer()])
            embedding_here = session.run(embedding)
        self.logger.info(type(embedding_here))
        self.logger.info(embedding_here.shape)
        self.logger.info(self.question2meaning)
        meaning = self.question2meaning.get(st, None)
        weight_before = self.meaning_weight[meaning]
        self.logger.info(weight_before)
        embedding_before = self.meaning_embedding[meaning]
        self.logger.info(embedding_before)
        result = (weight_before*embedding_before +
                  embedding_here*weight)/(weight_before + weight)
        self.logger.info(result)
        self.meaning_embedding[meaning] = result

        # temp_embedding = list(self.meaning_embedding.values())
        temp_embedding = [(key, value) for key, value
                          in self.meaning_embedding.items()]
        self.meaning_list, value_embedding = zip(*temp_embedding)
        temp_embedding2 = [embedding.tolist()[0]
                           for embedding in value_embedding]
        self.logger.info(temp_embedding2)
        self.group_embedding = np.array(temp_embedding2)

    def build_embed(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.embed = hub.Module(os.path.join(file_dir,
                                'data/universal-sentence-encoder'))


if __name__ == '__main__':

    football = SentenceTarget()
    football.append_tpl_by_key('person_info', 'Who is Jim')
    football.append_tpl_by_key('person_info', 'Jim')
    football.append_tpl_by_key('team_info', 'Is Arsenal a good team')
    football.append_tpl_by_key('team_info', 'The history of Arsenal')
    football.logger.info(football.group_embedding)
    # it should be logger.debug()
    the_scores = football.target_select('Arsenal')
    # print(the_scores.shape)
    # print(sum(the_scores))
    # print(the_scores/sum(the_scores))
    print('Similarity scores:', list(the_scores/sum(the_scores)))
    # it should be logger.info()
    print('List of different meanings:', football.meaning_list)
    print('Predicted meaning:', football.meaning_list[np.argmax(the_scores)])
