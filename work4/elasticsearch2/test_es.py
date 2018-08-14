"""
@Project   : text-classification-cnn-rnn
@Module    : test_es.py
@Author    : Deco [deco@cubee.com]
@Created   : 6/8/18 2:01 PM
@Desc      : 测试es的使用
When you search within a single index, Elasticsearch forwards the search
request to a primary or replica of every shard in that index, and then gathers
the results from each shard.
在linux中应该是用多进程在处理
Searching one index that has five primary shards is exactly equivalent to
searching five indices that have one primary shard each.
都是5个进程来处理？
You can see that, in a distributed system, the cost of sorting results grows
exponentially the deeper we page. There is a good reason that web search
engines don’t return more than 1,000 results for any query.
"""
import json
import pprint

import requests
from elasticsearch import Elasticsearch

es = Elasticsearch([{'host': 'localhost', 'port': 9200}])


def check_es():
    res = requests.get('http://localhost:9200')
    content_string = res.content.decode('utf-8')
    pprint.pprint(json.loads(content_string))


def list_indices():
    res = requests.get('http://localhost:9200/_cat/indices?v')
    print(res.content.decode('utf-8'))
    # print(res.content)


def put_data():
    r = requests.get('http://localhost:9200')
    i = 1
    while r.status_code == 200 and i <= 20:
        r = requests.get('http://swapi.co/api/people/' + str(i))
        es.index(index='sw', doc_type='people', id=i,
                 body=json.loads(r.content))
        print(i)
        i = i + 1


def put_data2():
    r = requests.get('http://localhost:9200')
    i = 18
    while r.status_code == 200 and i <= 40:
        r = requests.get('http://swapi.co/api/people/' + str(i))
        es.index(index='sw', doc_type='people', id=i,
                 body=json.loads(r.content))
        print(i)
        i = i + 1


def put_data3():
    r = requests.get('http://localhost:9200')
    i = 41
    while r.status_code == 200 and i <= 60:
        r = requests.get('http://swapi.co/api/people/' + str(i))
        es.index(index='sw', doc_type='people', id=i,
                 body=json.loads(r.content))
        print(i)
        i = i + 1


def update_data1():
    r = requests.get('http://localhost:9200')
    i = 1
    while r.status_code == 200 and i <= 20:
        update = {"doc": {"person_id": i}}
        r = requests.get('http://swapi.co/api/people/' + str(i))
        es.update(index='sw', doc_type='people', id=i,
                  body=update)
        print(i)
        i = i + 1


def search_data_id():
    p5 = es.get(index='sw', doc_type='people', id=5)
    pprint.pprint(p5)


def search_data_match():
    p = es.search(index="sw",
                  body={"query": {"match": {'name': 'Darth Vader'}}})
    pprint.pprint(p)


def search_data_sort():
    p = es.search(index="sw",
                  body={"query": {"match_all": {}},
                        "sort": {"height.raw": {"order": "desc"}}})
    pprint.pprint(p)


def search_data_match_all():
    p = es.search(index="sw",
                  body={"query": {"match_all": {}},
                        "sort": {"person_id": {"order": "desc"}}})
    pprint.pprint(p)


def search_data_prefix():
    p = es.search(index="sw", body={"query": {"prefix": {"name": "lu"}}})
    pprint.pprint(p)


def search_data_fuzzy():
    p = es.search(index="sw",
                  body={"query": {"fuzzy": {"name": "jaba"}}})
    pprint.pprint(p)


def count_data():
    r = requests.get('http://localhost:9200/_count?pretty')
    pprint.pprint(json.loads(r.content))


def create_mapping():
    mapping = {
        "mappings": {
            "people": {
                "properties": {
                    "height": {
                        "type": "text",
                        "fields": {
                            "raw": {
                                "type": "integer"
                            }
                        }
                    }
                }
            }
        }
    }

    setting = {
        "mappings": {
            "people": {
                "properties": {
                    "person_id": {
                        "type": "integer",
                    }
                }
            }
        }
    }

    # es.indices.create(index='sw', ignore=400, body=mapping)
    # es.indices.create(index='sw', body=mapping)
    es.indices.create(index='sw', body=setting)


def view_mapping():
    r = requests.get('http://localhost:9200/sw/_mapping/people')
    pprint.pprint(json.loads(r.content))


def del_sw():
    es.indices.delete(index='sw', ignore=[400, 404])


def del_megacorp():
    es.indices.delete(index='megacorp', ignore=[400, 404])


if __name__ == '__main__':
    list_indices()
