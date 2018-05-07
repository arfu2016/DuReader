"""
@Project   : DuReader
@Module    : word_segment.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/4/18 5:55 PM
@Desc      :
"""
# python word_segment.py data/wiki.zh.text data/wiki.zh.text.seg

import logging
import os.path
import sys
import re
import jieba
from opencc import OpenCC


def clean_word(word):

    pattern = '[\u4e00-\u9fff]+'
    # 该词至少有一个汉字

    extract = re.search(pattern, word)
    if extract is not None:
        tag = True
    else:
        tag = False
    return tag


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) != 3:
        print("Using: python word_segment.py wiki.zh.text wiki.zh.text.seg")
        sys.exit(1)
    inp, outp = sys.argv[1:3]
    i = 0

    openCC = OpenCC(
        't2s')  # convert from Simplified Chinese to Traditional Chinese

    finput = open(inp, 'r', encoding='utf-8')
    output = open(outp, 'w', encoding='utf-8')
    for line in finput:
        # todo:  用正则去掉非中文的字符
        line_jian = openCC.convert(line.strip())  # 繁体转换为简体
        line_seg = jieba.cut(line_jian)
        zh_seg = [word for word in line_seg if clean_word(word)]
        # 把非中文词汇（包括标点符号都去掉）
        output.write(' '.join(zh_seg) + '\n')
        i = i + 1
        if i % 5000 == 0:
            logger.info("Saved " + str(i) + " articles_seg")

    finput.close()
    output.close()
    logger.info("Finished Saved " + str(i) + " articles")
