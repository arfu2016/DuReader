"""
@Project   : DuReader
@Module    : build_sentences.py
@Author    : Deco [deco@cubee.com]
@Created   : 8/13/18 11:29 AM
@Desc      : 构建一组sentences
"""
import pprint

messages = [
    # Smartphones
    "I like my phone",
    "My phone is not good.",
    "Your cellphone looks great.",

    # Weather
    "Will it snow tomorrow?",
    "Recently a lot of hurricanes have hit the US",
    "Global warming is real",

    # Food and health
    "An apple a day, keeps the doctors away",
    "Eating strawberries is healthy",
    "Is paleo better than keto?",

    # Asking about age
    "How old are you?",
    "what is your age?",
]


if __name__ == '__main__':
    print('messages:')
    pprint.pprint(messages)
