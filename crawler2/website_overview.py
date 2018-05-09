"""
@Project   : DuReader
@Module    : website_overview.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/9/18 3:25 PM
@Desc      : Show secondary domains of a website
"""
import jieba
import newspaper
from newspaper import Article


def check_version():
    print('newspaper version:', newspaper.__version__)
    print('jieba version:', jieba.__version__)
# jieba may be overrided by jieba3k or an old version of jieba,
# such as 0.34 version
# pip uninstall jieba
# pip install jieba -i https://pypi.mirrors.ustc.edu.cn/simple

# On python3 you must install newspaper3k, not newspaper.
# newspaper is our python2 library. Although installing newspaper is simple
# with pip, you will run into fixable issues if you are trying to install
# on ubuntu.


def secondary_domains():
    qq_paper = newspaper.build('http://www.qq.com')

    for category in qq_paper.category_urls():
        print(category)


def bulid_website():

    # bioon_paper = newspaper.build('http://www.bioon.com/', language='zh')
    # bioon_paper = newspaper.build('http://news.bioon.com/', language='zh')

    q_paper = newspaper.build('http://history.news.qq.com', language='zh')

    print(q_paper.size())
    print(len(q_paper.articles))

    for article in q_paper.articles[0:10]:
        print(article.url)

    '''
    for article in q_paper.articles[0:10]:
    
        article.download()
    
        article.parse()
    
        # print('text:', article.text)
    
        print('title:', article.title)
    
        print('url:', article.url)
    '''


def download_title(url):

    a = Article(url, language='zh')  # Chinese

    a.download()

    a.parse()

    print('title:', a.title)

    print('url:', a.url)

    print('text:', a.text)


if __name__ == '__main__':
    check_version()
    secondary_domains()
    bulid_website()

    url0 = 'https://www.dongqiudi.com/archive/648933.html'
    download_title(url0)
    url1 = 'http://news.bioon.com/article/6721849.html'
    download_title(url1)
    url2 = ('http://sports.sina.com.cn/china/j/2018-05-07/'
            'doc-ifyuwqfa7533416.shtml')
    download_title(url2)

    url_pbc = ('http://www.pbc.gov.cn/goutongjiaoliu/113456/113469/' 
               '3519042/index.html')
    download_title(url_pbc)
