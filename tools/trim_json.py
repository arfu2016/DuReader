"""
@Project   : DuReader
@Module    : trim_json.py
@Author    : Deco [deco@cubee.com]
@Created   : 8/7/18 4:39 PM
@Desc      : 
"""
import json

file_name = '/decaNLP/.data/squad/train-v1.1.json'
train_file = '/home/projects/DuReader/data/demo_squad/train_demo.json'
valid_file = '/home/projects/DuReader/data/demo_squad/valid_demo.json'

train_data = dict()
valid_data = dict()
with open(file_name, encoding='utf-8') as fin:
    squad = json.load(fin)['data']
    # print(type(squad))
    train_data['data'] = squad[0:2]
    valid_data['data'] = squad[2:4]

with open(train_file, 'w', encoding='utf-8') as fout:
    json.dump(train_data, fout)

with open(valid_file, 'w', encoding='utf-8') as fout:
    json.dump(valid_data, fout)
