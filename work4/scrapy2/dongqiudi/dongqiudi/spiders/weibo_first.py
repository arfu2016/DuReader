"""
@Project   : DuReader
@Module    : weibo_first.py
@Author    : Deco [deco@cubee.com]
@Created   : 9/6/18 10:23 AM
@Desc      : 
"""
import scrapy


class WeiboSpider(scrapy.Spider):
    name = 'weibo_first'
    allowed_domains = ['www.weibo.com']
    start_urls = ['https://www.weibo.com']

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
        return {
            'page_url': response.url,
            'content': response.body.decode('gb2312')
        }
