"""
@Project   : DuReader
@Module    : punctuation_sub.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/16/18 1:36 PM
@Desc      : 
"""
import string
import re


def clean_sentence(st):
    intab = string.punctuation + '。，“”‘’（）：；？·—《》、'
    outtab = ' '
    table = str.maketrans(dict.fromkeys(intab, outtab))
    st1 = st.translate(table)
    return st1


def clean_sentence2(st):
    """
    数据预处理
    :param st: string
    :return: string
    """
    in_tab = r'''[{}]'''
    out_tab = ' '
    # out_tab = 'p'
    clean = re.sub(in_tab, out_tab, st)
    return clean


def clean_sentence3(st):
    """
    数据预处理
    :param st: string
    :return: string
    """
    in_tab = '[' + string.punctuation + '。，“”‘’（）：；？·—《》、' + ']'
    out_tab = ''
    clean = re.sub(in_tab, out_tab, st)
    return clean


def clean_sentence4(st):
    """
    数据预处理
    :param st: string
    :return: string
    """
    in_tab = string.punctuation + '。，“”‘’（）：；？·—《》、'
    clean = ''.join([c for c in st if c not in in_tab])
    # string search, time complexity m*O(n)
    return clean


def clean_sentence5(st):
    """
    数据预处理
    :param st: string
    :return: string
    """
    in_tab = string.punctuation + '。，“”‘’（）：；？·—《》、'
    pt = set(p for p in in_tab)
    clean = ''.join([c for c in st if c not in pt])
    # hash search, time complexity m*O(1)
    return clean


if __name__ == "__main__":
    print(string.punctuation)
    print(clean_sentence4('The period will be removed.'))
    print(clean_sentence5('The period will be removed.'))
