# -*- coding: utf-8 -*-
"""
The use of scrapy.Spider to get a list of news in html page
"""

import socket
import datetime

import scrapy
from scrapy.loader import ItemLoader
from scrapy.loader.processors import MapCompose
from ..items import DongqiudiItem


class FastSpider(scrapy.Spider):
    name = 'fast'
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
        selectors = response.xpath(
            '//*[@id="news_list"]/ol/li')
        for selector in selectors:
            yield self.parse_item(selector, response)

    def parse_item(self, selector, response):
        """This function parses a page to retrieve information.

        :param selector:
        :param response:
        :return:
        """

        l_item = ItemLoader(item=DongqiudiItem(), selector=selector)

        l_item.add_xpath('title', './h2/a/text()',
                         MapCompose(str.strip))
        l_item.add_xpath('description',
                         './p/text()')
        l_item.add_xpath('url', './h2/a/@href')

        l_item.add_value('page_url', response.url)
        l_item.add_value('project', self.settings.get('BOT_NAME'))
        l_item.add_value('spider', self.name)
        l_item.add_value('server', socket.gethostname())
        l_item.add_value('date', datetime.datetime.now())

        return l_item.load_item()
