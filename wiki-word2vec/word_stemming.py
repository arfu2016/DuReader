"""
@Project   : DuReader
@Module    : word_stemming.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/25/18 3:59 PM
@Desc      : 
"""
from nltk.stem.lancaster import LancasterStemmer

st = LancasterStemmer()
print(st.stem('stemmed'))

print(st.stem('stemming'))

print(st.stem('stemmer'))

print(st.stem('running'))

print(st.stem('maximum'))

print(st.stem('presumably'))
