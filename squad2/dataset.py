"""
@Project   : DuReader
@Module    : dataset.py
@Author    : Deco [deco@cubee.com]
@Created   : 7/23/18 5:50 PM
@Desc      : 
"""

import json
import logging
from collections import Counter

import numpy as np
from nltk.tokenize import word_tokenize
from .logger_setup import define_logger
logger = define_logger('squad2.dataset')


class BRCDataset:
    """
    This module implements the APIs for loading and using baidu reading
    comprehension dataset
    """

    def __init__(self, max_p_num, max_p_len, max_q_len,
                 train_files=None, dev_files=None, test_files=None):
        # p: paragraph, q: question
        # maximal passage number for each question

        if train_files is None:
            train_files = []
        if dev_files is None:
            dev_files = []
        if test_files is None:
            test_files = []

        self.logger = logger

        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len

        self.train_set, self.dev_set, self.test_set = [], [], []

        if train_files:
            for train_file in train_files:
                self.train_set += self._load_dataset(train_file, train=True)
                # 准备训练数据：train_set中放的是读入的数据
            self.logger.info(
                'Train set size: {} questions.'.format(len(self.train_set)))

        if dev_files:
            for dev_file in dev_files:
                self.dev_set += self._load_dataset(dev_file)
            self.logger.info(
                'Dev set size: {} questions.'.format(len(self.dev_set)))

        if test_files:
            for test_file in test_files:
                self.test_set += self._load_dataset(test_file)
            for sample in self.test_set:
                logger.debug(sample['question'])
            self.logger.info(
                'Test set size: {} questions.'.format(len(self.test_set)))

    def _load_dataset(self, data_path, train=False):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        logger.debug(data_path)
        data_set, all_answers = [], []
        # 放在with模块外面，不要做成with的局部变量
        with open(data_path, encoding='utf-8') as fin:

            logger.debug('Begin to load data from json file...')
            squad = json.load(fin)['data']

            # for lidx, line in enumerate(fin):
            for document in squad:
                # sample = json.loads(line.strip())
                # 把json格式转换成python格式
                # if train:
                #     if len(sample['answer_spans']) == 0:
                #         continue
                #     # if sample['answer_spans'][0][1] >= self.max_p_len:
                #     if sample['answer_spans'][0][1] - \
                #             sample['answer_spans'][0][0] >= self.max_p_len:
                #         continue
                    # 对容错的考虑
                    # 如果没有'answer_spans'或者answer_spans太长，该行数据就被舍弃了，
                    # 不放在training data中

                # if 'answer_docs' in sample:
                #     sample['answer_passages'] = sample['answer_docs']
                    # 重新命名

                paragraphs = document['paragraphs']

                # if 'paragraphs' in document:
                #     document['answer_passages'] = document['paragraphs']

                for paragraph in paragraphs:
                    sample = dict()
                    # 每次把sample bound到一个新的字典上，不要去修改原来的对象

                    context = paragraph['context']
                    # segmented_paragraphs = context.split()
                    segmented_paragraphs = word_tokenize(context)
                    # todo_finished: revise
                    qas = paragraph['qas']

                    for qa in qas:
                        # sample['question_tokens'] = sample[
                        #     'segmented_question']
                        sample['question_tokens'] = word_tokenize(
                            qa['question'])
                        # logger.debug(qa['question'])
                        # todo_finished: revise
                        question = ' '.join(sample['question_tokens'])
                        answer = qa['answers'][0]['text']
                        squad_id = len(all_answers)
                        all_answers.append(
                            [a['text'] for a in qa['answers']])
                        answer_start = qa['answers'][0]['answer_start']
                        answer_end = answer_start + len(answer)

                        sample['passages'] = [
                            {'passage_tokens': segmented_paragraphs,
                             'is_selected': True}]
                        # logger.debug(sample['passages'][0]['passage_tokens'])

                        sample['question'] = question
                        # logger.debug(sample['question'])
                        sample['question_id'] = squad_id
                        sample['question_type'] = "DESCRIPTION"
                        sample['answers'] = [answer]
                        # logger.debug(sample['answers'])
                        sample['answer_passages'] = [0]

                        word_start = len(word_tokenize(context[:answer_start]))
                        word_end = word_start + len(
                            word_tokenize(context[answer_start: answer_end]))

                        word_position = dict()
                        p_idx = 0
                        for idx, word in enumerate(segmented_paragraphs):
                            word_position[idx] = p_idx
                            p_idx += len(word)

                        sample['answer_spans'] = [[word_start, word_end]]
                        # logger.debug(sample['answer_spans'])
                        fake_answer = sample[
                            'passages'][0]['passage_tokens'][
                                word_start: word_end]
                        # fake_answer2 = context[answer_start: answer_end]
                        sample['fake_answers'] = [' '.join(fake_answer)]
                        # logger.debug(sample['fake_answers'])
                        # logger.debug(fake_answer2)
                        # todo_finished: revise
                        sample['match_scores'] = [1.00]
                        sample['segmented_answers'] = answer.split()

                        data_set.append(sample)
                        # 把该行数据加入到data_set中

        print('line 166')
        for sample in data_set:
            logger.debug(sample['question'])

        return data_set

    def _one_mini_batch(self, data, indices, pad_id):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        # logger.debug(indices)
        batch_data = {'raw_data': [data[i] for i in indices],
                      'question_token_ids': [],
                      'question_length': [],
                      'passage_token_ids': [],
                      'passage_length': [],
                      'start_id': [],
                      'end_id': []}
        # 这里用来对语料做标记，主要是标记start_id和end_id

        # for sample in batch_data['raw_data']:
        #     logger.debug(sample['question'])
        max_passage_num = max(
            [len(sample['passages']) for sample in batch_data['raw_data']])
        # 这一批数据中段落数目最大的样本
        max_passage_num = min(
            self.max_p_num, max_passage_num)
        # 设定的最大的段落数
        for sidx, sample in enumerate(batch_data['raw_data']):
            for pidx in range(max_passage_num):
                # pidx表示sample中的第几个段落，比如第0个、第1个等
                if pidx < len(sample['passages']):
                    batch_data['question_token_ids'].append(
                        sample['question_token_ids'])
                    batch_data['question_length'].append(
                        len(sample['question_token_ids']))
                    passage_token_ids = sample[
                        'passages'][pidx]['passage_token_ids']
                    batch_data['passage_token_ids'].append(
                        passage_token_ids)
                    # 一般来说，max_passage_num是5，也就是每个问题对应着5段文档
                    # 所以batch_data['passage_token_ids']的维度是(16*5, 500)
                    batch_data['passage_length'].append(
                        min(len(passage_token_ids), self.max_p_len))
                    # 'passage_length'最大按self.max_p_len算
                    # 之后把这个参数传给tensorflow以后，长度之外的，tf就不管了
                else:
                    # 没有这么多的段落的话，补空值
                    batch_data['question_token_ids'].append([])
                    batch_data['question_length'].append(0)
                    batch_data['passage_token_ids'].append([])
                    batch_data['passage_length'].append(0)
        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(
            batch_data, pad_id)
        for sample in batch_data['raw_data']:
            if 'answer_passages' in sample and len(sample['answer_passages']):
                # answer docs给出来以后，还要知道是哪个段落
                # 给出了answer来源于的passage，可用来设定start_id和end_id
                # 对于train和validate数据，'answer_passages'这个字段是必需的
                gold_passage_offset = padded_p_len * sample[
                    'answer_passages'][0]
                # 如果是16个batch，由于每个问题给出了5个document，开始的时候数据的第一维
                # 是16*5=80，但后来会缩减为16，padded_p_len是用于训练的每个文档的长度，
                # sample['answer_passages'][0]中放的是答案在第几个文档中的信息，所以
                # 二者相乘就是答案所在文档的起始点。

                # 这里，只取了sample['answer_passages']中的第一个元素，也就是说我们假
                # 设只有一个答案，实际上，dureader数据集是给出了多个答案的.
                # 如果考虑多个答案的话，最终的评分有可能会更高。
                batch_data['start_id'].append(
                    gold_passage_offset + sample['answer_spans'][0][0])
                # answer span给出的list以0开始，以answer长度终止
                batch_data['end_id'].append(
                    gold_passage_offset + sample['answer_spans'][0][1])
            else:
                # fake span for some samples, only valid for testing
                batch_data['start_id'].append(0)
                batch_data['end_id'].append(0)
        return batch_data

    def _dynamic_padding(self, batch_data, pad_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = min(self.max_q_len, max(batch_data['question_length']))
        # pad后的长度考虑了self.max_p_len和self.max_q_len
        batch_data['passage_token_ids'] = [
            (ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
            for ids in batch_data['passage_token_ids']]
        batch_data['question_token_ids'] = [
            (ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
            for ids in batch_data['question_token_ids']]
        # 上面是padding batch_data的过程
        return batch_data, pad_p_len, pad_q_len

    def word_iter(self, set_name=None):
        """
        Actually a generator with yield statement
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError(
                'No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                # 每个sample代表原数据文件中的一行
                for token in sample['question_tokens']:
                    yield token
                for passage in sample['passages']:
                    # sample中的每一个文档都做处理
                    for token in passage['passage_tokens']:
                        # tokens in the most related paragraph
                        yield token

    def convert_to_ids(self, vocab):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
                # for training data, dev_set and test_set is None
            for sample in data_set:
                sample['question_token_ids'] = \
                    vocab.convert_to_ids(sample['question_tokens'])
                # convert tokens to ids
                for passage in sample['passages']:
                    passage['passage_token_ids'] = \
                        vocab.convert_to_ids(passage['passage_tokens'])

    def gen_mini_batches(self, set_name, batch_size, pad_id, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError(
                'No data set named as {}'.format(set_name))
        data_size = len(data)
        # for sample in data:
        #     logger.debug(sample['question'])
        logger.debug(data_size)
        indices = np.arange(data_size)
        # 是所有data的index
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            # logger.debug(batch_indices)
            yield self._one_mini_batch(data, batch_indices, pad_id)
