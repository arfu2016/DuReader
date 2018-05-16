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
