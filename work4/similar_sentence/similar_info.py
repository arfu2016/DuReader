"""
@Project   : DuReader
@Module    : similar_info.py
@Author    : Deco [deco@cubee.com]
@Created   : 8/15/18 1:33 PM
@Desc      : 
"""
import pprint
import time

import numpy as np

from work4.elasticsearch2.extract_person import search_data_match
from work4.logger_setup import define_logger
from work4.similar_sentence.one_vs_group import st_st_similarity, most_similar

logger = define_logger('work4.similar_sentence.similar_info')

label_dict = {
    '梅西 跑动和进球': '梅西 跑动数据, 梅西 跑动速度; 梅西 刷新 历史进球纪录',
    '梅西 暑假': '世界足坛 明星们 过 暑假;'
}
# labels = ['梅西 跑动和进球', '梅西 暑假']

label_st = {
    '梅西 跑动和进球': ['9球，梅西刷新自己的甘伯杯历史进球纪录; ',
                 '阿根廷跑动数据：梅西7.6公里; 对阵克罗地亚全场84%的时间梅西的跑动速度均在7km/h以下。'],
    '梅西 暑假': ['梅西逗狗，二弟玩鹰！来看看世界足坛明星们都是怎么过暑假的']
}


def cal_threshold(key):
    if len(label_st[key]) == 0:
        return 0.8
    else:
        sim_scores = st_st_similarity(tuple(label_st[key]), (label_dict[key],))
        mean = np.mean(sim_scores)
        std = np.std(sim_scores)
        # print(mean, std)
        return mean-5*std


def label_sentence():
    training_sentences = tuple(search_data_match())
    for key in label_dict:
        label = label_dict[key]
        test_sentence = (label,)
        threshold = cal_threshold(key)
        top_sentences = most_similar(training_sentences, test_sentence,
                                     threshold)
        print()  # 竟然是非阻塞式的调用
        time.sleep(0.5)
        logger.info('The similar sentences as {}'.format(label))
        time.sleep(0.5)
        pprint.pprint(top_sentences)


if __name__ == '__main__':

    label_sentence()
