"""
@Project   : DuReader
@Module    : pprint_list.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/10/18 4:12 PM
@Desc      : 
"""
import pprint

a = [['human', 'interface', 'computer'],
     ['survey', 'user', 'computer', 'system', 'response', 'time'],
     ['eps', 'user', 'interface', 'system'],
     ['system', 'human', 'system', 'eps'],
     ['user', 'response', 'time'],
     ['trees'], ['graph', 'trees'],
     ['graph', 'minors', 'trees'],
     ['graph', 'minors', 'survey']]

pprint.pprint(a)
