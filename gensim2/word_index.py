"""
@Project   : DuReader
@Module    : word_index.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/16/18 5:46 PM
@Desc      : 
"""
from collections import Counter


def word_index():
    word2index = {}
    index2word = {}
    vocab = ['word', 'another']
    # vocab = list({'word', 'another'})
    for i, word in enumerate(vocab):
        word2index[word] = i
        index2word[i] = word
    return word2index, index2word


def word_index2():
    words = ['word', 'another', 'word']
    counts = Counter(words)
    print('counts:', counts)
    vocab = sorted(counts, key=counts.get, reverse=True)
    print('vocab:', vocab)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    return vocab_to_int


if __name__ == '__main__':
    print(word_index())
    print(word_index2())
