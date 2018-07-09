"""
@Project   : DuReader
@Module    : sen_sim_spacy.py
@Author    : Deco [deco@cubee.com]
@Created   : 7/9/18 10:37 AM
@Desc      : 
"""
import spacy
import numpy as np
import time

nlp = spacy.load('en_core_web_md')

tokens = nlp("The quick brown fox jumps over the lazy dog.")


def most_similar(training_sentences: tuple, test_sentence: str) -> str:

    docs = [nlp(st) for st in training_sentences]
    doc_test = nlp(test_sentence)

    print('doc_test:', doc_test.has_vector, doc_test.vector_norm)

    sim_scores = [doc_test.similarity(doc) for doc in docs]
    print('sim_scores', sim_scores)

    idx = np.argmax(sim_scores)
    return training_sentences[idx]


# test

def t_most_similar():
    start = time.perf_counter()
    print()
    print('Test t_most_similar()')
    training_sentences = ("The quick brown fox jumps over the lazy dog.",
                          "Who is Messy?")
    test_sentence = 'Something about football.'
    top_sentence = most_similar(training_sentences, test_sentence)
    print("Most similar sentence of "
          "'Something about football.':")
    print(top_sentence)
    print('time of t_most_similar():', time.perf_counter() - start)


def t2_most_similar():
    start = time.perf_counter()
    print()
    print('Test t2_most_similar()')
    training_sentences = ("The quick brown fox jumps over the lazy dog.",
                          "Who is Messy?")
    test_sentence = 'Can you tell me something about Cristiano Ronaldo?'
    top_sentence = most_similar(training_sentences, test_sentence)
    print("Most similar sentence of "
          "'Can you tell me something about Cristiano Ronaldo?':")
    print(top_sentence)
    print('time of t2_most_similar():', time.perf_counter() - start)


if __name__ == '__main__':

    for token in tokens:
        print(token.text, token.has_vector, token.vector_norm)
        # print(token.text, token.has_vector, token.vector_norm, token.vector)

    t_most_similar()
    t2_most_similar()
