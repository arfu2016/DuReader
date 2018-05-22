"""
@Project   : DuReader
@Module    : tokenizer.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/22/18 5:00 PM
@Desc      : 
"""

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

# import nltk
#
# nltk.download('punkt')

if __name__ == '__main__':
    print(word_tokenize('Hello World.'))
    print(word_tokenize("this’s a test"))

    text = ("this’s a sent tokenize test. this is sent two. "
            "is this sent three? sent 4 is cool! Now it’s your turn.")
    sent_tokenize_list = sent_tokenize(text)
    print(sent_tokenize_list)
