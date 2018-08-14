"""
@Project   : text-classification-cnn-rnn
@Module    : clear_items.py
@Author    : Deco [deco@cubee.com]
@Created   : 6/19/18 11:03 AM
@Desc      : 手动把存放爬虫内容的文件清空
"""

import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fn = os.path.join(base_dir, 'items.json')


if __name__ == '__main__':
    with open(fn, 'w', encoding='utf-8') as f:
        f.write('')
