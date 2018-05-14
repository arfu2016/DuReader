"""
@Project   : DuReader
@Module    : load_table.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/14/18 3:54 PM
@Desc      : 
"""
import os
import logging
import sys
from collections import Counter
import pickle
import pprint


def set_logger():
    # for logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    return logger


def count_list(list_for_count):
    cnt = Counter()
    for item in list_for_count:
        cnt[item] += 1
    return cnt


def load_data():
    logger = set_logger()
    file_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(file_dir, 'data/templates_intents.pkl')

    with open(file_name, 'rb') as f:
        ti = pickle.load(f)

    neg_label = {'no_idea', 'None', 'ask_back'}
    ti = [(tpl, intent) for tpl, intent in ti if intent not in neg_label]

    logger.info('There are {} effective samples'.format(len(ti)))

    # pprint.pprint(ti[0:100])

    _, ti_intent = zip(*ti)

    intent_cnt = count_list(ti_intent)

    # print('Number of different intents:')
    # pprint.pprint(intent_cnt.most_common())

    top10 = intent_cnt.most_common(10)
    print('Number of top 10 intents:')
    pprint.pprint(top10)

    _, top_number = zip(*top10)
    logger.info('There are {} top samples'.format(sum(top_number)))


if __name__ == '__main__':
    load_data()
