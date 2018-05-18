"""
@Project   : DuReader
@Module    : load_movie_review.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/17/18 5:33 PM
@Desc      : 
"""

import os
from string import punctuation
from collections import Counter
import numpy as np

file_dir = os.path.dirname(os.path.abspath(__file__))
fn_review = os.path.join(file_dir, 'data/reviews.txt')
fn_label = os.path.join(file_dir, 'data/labels.txt')

with open(fn_review, 'r', encoding='utf-8') as f:
    reviews = f.read()
with open(fn_label, 'r', encoding='utf-8') as f:
    labels = f.read()

print(reviews[:2000])

all_text = ''.join([c for c in reviews if c not in punctuation])
reviews = all_text.split('\n')

all_text = ' '.join(reviews)
words = all_text.split()

print(all_text[:2000])

print(words[:100])

counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

reviews_ints = []
for each in reviews:
    reviews_ints.append([vocab_to_int[word] for word in each.split()])

labels = labels.split('\n')
labels = np.array([1 if each == 'positive' else 0 for each in labels])

review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))
