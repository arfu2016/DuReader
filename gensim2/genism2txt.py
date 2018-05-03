"""
@Project   : DuReader
@Module    : genism2txt.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/2/18 4:33 PM
@Desc      : From a gensim native model of word vectors to a text file
"""

import os

file_dir = os.path.dirname(os.path.abspath(__file__))
file_from_path = os.path.join(file_dir, 'data/word2vec_sports.model')
file_to_path = os.path.join(file_dir, 'data/word2vec_sports.txt')


def test1():
    from gensim.models.keyedvectors import KeyedVectors

    model = KeyedVectors.load_word2vec_format(file_from_path, binary=True)
    # It may not work
    model.save_word2vec_format(file_to_path, binary=False)


def test2():
    from gensim.models import word2vec
    model = word2vec.Word2Vec.load_word2vec_format(file_from_path,
                                                   binary=True)
    # It may not work
    model.save_word2vec_format(file_to_path,
                               binary=False)


def test3():
    """This works and can produce the txt file"""
    from gensim.models import word2vec
    model = word2vec.Word2Vec.load(file_from_path)
    model.wv.save_word2vec_format(file_to_path, binary=False)


def test4():
    from gensim.models import word2vec
    fname = os.path.join(file_dir, 'data/word2vec_sports_again.model')
    model = word2vec.Word2Vec.load(
        file_from_path)  # you can continue training with the loaded model!
    # train the model again, and then save
    model.save(fname)

    print(model.wv['computer'])
    # numpy vector of a word
    print(model.wv.most_similar(positive=['woman', 'king'], negative=['man']))
    # according to euclidean distance?

    print(model.wv.most_similar_cosmul(
        positive=['woman', 'king'], negative=['man']))
    # according to cosine similarity

    print(model.wv.similarity('woman', 'man'))  # similarity score
    print(model.wv.doesnt_match("breakfast cereal dinner lunch".split()))
    print(model.score(["The fox jumped over a lazy dog".split()]))
    # Probability of a text under the model
    # probability of a sentence and word2vec

    word_vectors = model.wv
    del model
    # to trim unneeded model memory = use much less RAM.


def test5():
    from gensim.models import KeyedVectors
    word_vectors = KeyedVectors.load_word2vec_format(file_to_path,
                                                     binary=False)
    # C text format


def test6(sentences):
    """fit a word2vec model for phrases"""
    import gensim
    bigram_transformer = gensim.models.Phrases(sentences)
    # 作用相当于分词器，tokenizer
    model = gensim.models.word2vec.Word2Vec(bigram_transformer[sentences],
                                            size=100)


def test7(sentences):
    """fit an ordinary model"""
    from gensim.models.word2vec import Word2Vec
    model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)


def test8():
    from gensim.models.word2vec import Word2Vec
    model = Word2Vec.load_word2vec_format(file_to_path, binary=False)
    # C text format


if __name__ == '__main__':
    test3()
