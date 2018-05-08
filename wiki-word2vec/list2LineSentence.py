"""
@Project   : DuReader
@Module    : list2LineSentence.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/8/18 4:03 PM
@Desc      : 
"""
from io import StringIO
import jieba
from gensim.models.word2vec import LineSentence


def list_generate(sts):
    seg_list = [list(jieba.cut(st)) for st in sts]
    return seg_list


def file_generate(sts):
    seg_list = [' '.join(jieba.cut(st)) for st in sts]
    handle = StringIO('\n'.join(seg_list))
    return handle


if __name__ == '__main__':
    messages = [
        # 不在顶级联赛
        "哪些球员球队如今不在顶级联赛",
        "现在哪些球员所在的球队不在顶级联赛了",
        "有没有什么球员现在不在顶级联赛踢球了",

        # 死敌
        "哪些球员的俱乐部是死敌",
        "现在哪些球员所在球队是死敌",
        "哪些球员所在的俱乐部势不两立",

        # 效力
        "梅西效力于哪家俱乐部",
        "梅西在哪个球队踢球",
        "梅西在哪个俱乐部踢球",

        # Asking about age
        "你多大了",
        "你的年龄是多少",
    ]

    print('messages:')
    for message in LineSentence(file_generate(messages)):
        print(message)

    print('messages2:')
    for message in LineSentence(file_generate(messages)):
        print(message)
