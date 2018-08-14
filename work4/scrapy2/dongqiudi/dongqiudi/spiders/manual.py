"""
@Project   : text-classification-cnn-rnn
@Module    : manual.py
@Author    : Deco [deco@cubee.com]
@Created   : 6/15/18 2:05 PM
@Desc      : 爬取json数据，并存入.json文件中；主要的爬虫文件
scrapy check manual
scrapy crawl manual -o items.json
"""
import socket
import datetime
import json

import scrapy
from scrapy.loader import ItemLoader
from ..items import DongqiudiItem


class BasicSpider(scrapy.Spider):
    name = 'manual'
    allowed_domains = ['www.dongqiudi.com']
    start_urls = ['http://www.dongqiudi.com/archives/1?page={}'.format(i)
                  for i in range(1, 11)]
    # request related

    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/63.0.3239.84 Safari/537.36',
        "DOWNLOAD_DELAY": 5,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1
    }

    def parse(self, response):
        """This function parses a news page. response related

        :param response:
        :return:
        """
        delay = self.crawler.engine.downloader.slots["www.dongqiudi.com"].delay
        concurrency = self.crawler.engine.downloader.slots[
            "www.dongqiudi.com"].concurrency
        self.log(
            "Delay {}, concurrency {} for request {}".format(delay,
                                                             concurrency,
                                                             response.request))
        js = json.loads(response.body)
        data = js['data']
        for item in data:
            yield self.parse_item(item, response)

    def parse_item(self, item, response):

        l_item = ItemLoader(item=DongqiudiItem())

        l_item.add_value('title', item['title'].strip())
        l_item.add_value('description', item['description'])
        l_item.add_value('url', item['web_url'])
        l_item.add_value('display_time', item['display_time'])

        l_item.add_value('page_url', response.url)
        l_item.add_value('project', self.settings.get('BOT_NAME'))
        l_item.add_value('spider', self.name)
        l_item.add_value('server', socket.gethostname())
        l_item.add_value('date', datetime.datetime.now())

        return l_item.load_item()
