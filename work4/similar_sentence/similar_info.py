"""
@Project   : DuReader
@Module    : similar_info.py
@Author    : Deco [deco@cubee.com]
@Created   : 8/15/18 1:33 PM
@Desc      : 
"""
import pprint
from work4.elasticsearch2.extract_person import search_data_match
from work4.similar_sentence.one_vs_group import most_similar


def t_most_similar():
    training_sentences = tuple(search_data_match())
    test_sentence = ('诺坎普的王 梅西 首次 奖杯 队史 第一人',)
    # 梅西 跑动数据 跑动速度
    # 诺坎普的王 梅西 首次 奖杯 队史 第一人

    # 梅西 跑动数据
    # 梅西 荣誉
    top_sentences = most_similar(training_sentences, test_sentence)
    pprint.pprint(top_sentences)


def label_sentence():
    label_dict = {
        '梅西 跑动和进球': ['梅西 跑动数据 跑动速度; 梅西 进球纪录'],
        '梅西 暑假': ['世界足坛 明星们 过 暑假;']
        # '梅西 跑动数据': ['梅西 跑动数据 跑动速度'],
        # '梅西 荣誉': ['梅西 荣誉 首次 奖杯 队史']
    }
    # labels = ['梅西 跑动数据 跑动速度', '梅西 荣誉 首次 奖杯 队史 第一人']
    value_list = list(label_dict.values())
    labels = []
    for value in value_list:
        labels.extend(value)
    training_sentences = tuple(search_data_match())
    for label in labels:
        test_sentence = (label,)
        top_sentences = most_similar(training_sentences, test_sentence)
        pprint.pprint(top_sentences)


if __name__ == '__main__':

    # t_most_similar()
    label_sentence()
