"""
@Project   : DuReader
@Module    : dataset.py
@Author    : Deco [deco@cubee.com]
@Created   : 7/23/18 5:50 PM
@Desc      : 
"""

import logging


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

        self.logger = logging.getLogger("brc")
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
            self.logger.info(
                'Test set size: {} questions.'.format(len(self.test_set)))
