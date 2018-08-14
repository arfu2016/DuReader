"""
@Project   : text-classification-cnn-rnn
@Module    : test.py
@Author    : Deco [deco@cubee.com]
@Created   : 6/15/18 2:52 PM
@Desc      : 爬取并打印json数据，当给出的翻页数字太大时，处理异常情况
"""

import json

import scrapy
import pprint
from ..items import DongqiudiItem


def parse(response):
    item = DongqiudiItem()
    item['title'] = response.xpath(
        '//*[@id="news_list"]/ol/li[1]/h2/a/text()').extract()
    item['url'] = response.xpath(
        '//*[@id="news_list"]/ol/li[1]/h2/a/@href').extract()
    return item


class BasicSpider(scrapy.Spider):
    name = 'test'
    allowed_domains = ['www.dongqiudi.com']
    start_urls = ['http://www.dongqiudi.com/archives/1?page=1']

    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/63.0.3239.84 Safari/537.36',
    }

    def parse(self, response):
        """This function parses a news page.

        :param response:
        :return:
        """
        if response.status == 200:
            js = json.loads(response.body)
            data = js['data']
            pprint.pprint(data)
        elif response.status == 404:
            self.log('404 error: this page does not exist')
        else:
            self.log('{} error'.format(response.status_code))
