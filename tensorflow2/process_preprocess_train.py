"""
@Project   : DuReader
@Module    : process_preprocess.py
@Author    : Deco [deco@cubee.com]
@Created   : 8/17/18 10:26 AM
@Desc      : 
"""
import os

base_dir = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

data_path_in = os.path.join(base_dir, 'data/preprocessed/trainset/search.train.json')
data_path_out = os.path.join(base_dir, 'data/preprocessed/trainset/search.train2.json')

with open(data_path_in, encoding='utf-8') as fin:
    with open(data_path_out, 'w', encoding='utf-8') as fout:
        for lidx, line in enumerate(fin):
            fout.write(line)
            if lidx == 86945:
                break
