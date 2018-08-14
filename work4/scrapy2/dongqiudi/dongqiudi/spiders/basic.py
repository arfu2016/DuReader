# -*- coding: utf-8 -*-
"""
The use of scrapy.Spider to get the title, description and url of a news
"""
import socket
import datetime

import scrapy
from scrapy.loader import ItemLoader
from scrapy.loader.processors import MapCompose
from ..items import DongqiudiItem


def parse(response):
    item = DongqiudiItem()
    item['title'] = response.xpath(
        '//*[@id="news_list"]/ol/li[1]/h2/a/text()').extract()
    item['url'] = response.xpath(
        '//*[@id="news_list"]/ol/li[1]/h2/a/@href').extract()
    return item


class BasicSpider(scrapy.Spider):
    name = 'basic'
    allowed_domains = ['www.dongqiudi.com']
    start_urls = ['http://www.dongqiudi.com/']

    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/63.0.3239.84 Safari/537.36',
    }

    def parse(self, response):
        """This function parses a news page.

        :param response:
        :return:

        @url http://www.dongqiudi.com/
        @returns items 1
        @scrapes title url
        @scrapes page_url project spider server date
        """
        l_item = ItemLoader(item=DongqiudiItem(), response=response)

        l_item.add_xpath('title', '//*[@id="news_list"]/ol/li[1]/h2/a/text()',
                         MapCompose(str.strip))
        l_item.add_xpath('description',
                         '//*[@id="news_list"]/ol/li[1]/p/text()')
        l_item.add_xpath('url', '//*[@id="news_list"]/ol/li[1]/h2/a/@href')

        l_item.add_value('page_url', response.url)
        l_item.add_value('project', self.settings.get('BOT_NAME'))
        l_item.add_value('spider', self.name)
        l_item.add_value('server', socket.gethostname())
        l_item.add_value('date', datetime.datetime.now())

        return l_item.load_item()
