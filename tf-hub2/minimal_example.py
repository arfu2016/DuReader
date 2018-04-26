"""
@Project   : DuReader
@Module    : minimal_example.py
@Author    : Deco [deco@cubee.com]
@Created   : 4/26/18 11:16 AM
@Desc      : 
"""

import tensorflow_hub as hub

embed = hub.Module("https://tfhub.dev/google/"
                   "universal-sentence-encoder/1")

if __name__ == '__main__':
    embedding = embed([
        "The quick brown fox jumps over the lazy dog."])
    print(embedding)
