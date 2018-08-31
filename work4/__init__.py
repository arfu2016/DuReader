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
实体抽取：elastic search；模板规则，正则规则，词性规则，句法规则；word window classification
事件抽取：sentence embedding

聚类用于改进事件抽取策略：看看抽取出时间的句子的聚类在整个句子聚类中的分布，
特别是用pca降到2维后画图来看，
看大片的没有被抽取事件的句子到底是在描述什么，反过来改进需要抽取的事件

自然语言数据挖掘的工具化
"""
