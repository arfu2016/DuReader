"""
@Project   : DuReader
@Module    : __init__.py.py
@Author    : Deco [deco@cubee.com]
@Created   : 8/14/18 5:06 PM
@Desc      :
1. cd scrapy2/dongqiudi
python data_preprocessing/clear_items.py
python data_preprocessing/show_items.py
scrapy crawl manual -o items.json
python data_preprocessing/show_items.py
2. cd elasticsearch2
python add_dongqiudi.py
python extract_person.py
3. cd similar_sentence
python similar_info.py

爬虫：scrapy
实体抽取：elastic search
事件抽取：sentence embedding
"""
