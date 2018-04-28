"""
@Project   : DuReader
@Module    : sentence_similar.py
@Author    : Deco [deco@cubee.com]
@Created   : 4/28/18 10:52 AM
@Desc      : 句子相似度计算
"""
import logging
import os
from collections import defaultdict

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
# 1压制info，只输出warn以上级别


class SentenceTarget:

    def __init__(self, original_data=None):

        self._set_logger()
        self._build_embed()
        self._load_original_data(original_data)
        if len(self.question2meaning) == 0:
            self.group_embedding = None
        # else:
        #     for st in self.question2meaning:
        #         self._append_embedding(st)

    def _set_logger(self):

        self.logger = logging.getLogger("vector")
        # self.logger.setLevel(logging.ERROR)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)

    def _build_embed(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.embed = hub.Module(os.path.join(file_dir,
                                'data/universal-sentence-encoder'))

    def _load_original_data(self, q2m=None):
        self.meaning2question = defaultdict(list)
        self.question2meaning = dict()
        self.meaning_embedding = defaultdict(float)
        self.meaning_weight = defaultdict(int)
        if q2m is not None:
            for question, meaning in q2m.items():
                self.append_tpl_by_key(meaning, question)

    def append_tpl_by_key(self, meaning, st):
        if st in self.question2meaning:
            self.logger.warning(
                '{} already exists in the templates'.format(st))
        else:
            self.meaning2question[meaning].append(st)
            self.question2meaning.update({st: meaning})
            self._append_embedding(st)

    def append_tpl_by_value(self, ref, st):
        meaning = self.question2meaning.get(ref, None)
        if meaning is not None:
            self.append_tpl_by_key(meaning, st)
        else:
            self.logger.warning(
                '{} does not exist in the templates'.format(ref))

    def _append_embedding(self, st, weight=1):
        embedding = self.embed([st])
        with tf.Session() as session:
            session.run(
                [tf.global_variables_initializer(), tf.tables_initializer()])
            embedding_here = session.run(embedding)

        self.logger.debug(type(embedding_here))
        self.logger.debug(embedding_here.shape)
        self.logger.debug(self.question2meaning)

        # meaning = self.question2meaning.get(st, None)
        meaning = self.question2meaning[st]
        weight_before = self.meaning_weight[meaning]
        self.meaning_weight[meaning] = weight_before + weight

        embedding_before = self.meaning_embedding[meaning]
        result = (weight_before*embedding_before +
                  embedding_here*weight)/(weight_before + weight)
        self.meaning_embedding[meaning] = result

        temp_embedding = [(key, value) for key, value
                          in self.meaning_embedding.items()]
        self.meaning_list, value_embedding = zip(*temp_embedding)
        components_embedding = [embedding.tolist()[0]
                                for embedding in value_embedding]

        self.group_embedding = np.array(components_embedding)

    def target_select(self, st):

        if self.group_embedding is None:

            return None

        def run_cal_similarity(sess):
            """Returns the similarity scores"""
            emba, embb, scores_sim = sess.run(
                [sts_encode1, sts_encode2, sim_scores],
                feed_dict={
                    sts_input1: group_question,
                    sts_std: self.group_embedding
                })
            return scores_sim

        group_question = [st] * len(self.meaning2question)
        sts_input1 = tf.placeholder(tf.string, shape=(None, ))
        sts_std = tf.placeholder(tf.float32, shape=(None, None))

        # We use exactly normalized rather than
        # approximately normalized.
        sts_encode1 = tf.nn.l2_normalize(self.embed(sts_input1))
        sts_encode2 = tf.nn.l2_normalize(sts_std)

        sim_scores = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2),
                                   axis=1)
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())
            scores = run_cal_similarity(session)
        return scores


if __name__ == '__main__':

    football = SentenceTarget(
        {'Could you tell me something about Jim': 'person_info'})

    football.append_tpl_by_key('person_info', 'Who is Jim')
    football.append_tpl_by_key('person_info', 'Jim')
    football.append_tpl_by_key('team_info', "Team Arsenal's story")
    football.append_tpl_by_key('team_info', 'The history of Arsenal')

    football.logger.debug(football.group_embedding)

    arsenal_scores = football.target_select('Arsenal')

    if arsenal_scores is not None:
        score_list = list(arsenal_scores / sum(arsenal_scores))
        football.logger.info(
            'Similarity scores: {}'.format(score_list))
        football.logger.info(
            'List of different meanings: {}'.format(football.meaning_list))
        predict_meaning = football.meaning_list[np.argmax(arsenal_scores)]
        football.logger.info(
            'Predicted meaning: {}'.format(predict_meaning))

    else:
        football.logger.warning('No references to classify the question.')
