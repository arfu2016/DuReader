"""
@Project   : DuReader
@Module    : trim_json.py
@Author    : Deco [deco@cubee.com]
@Created   : 8/7/18 4:39 PM
@Desc      : 
"""
import json

file_name = '/decaNLP/.data/squad/train-v1.1.json'

with open(file_name, encoding='utf-8') as fin:
    squad = json.load(fin)['data']
    print(type(squad))
