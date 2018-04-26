"""
@Project   : DuReader
@Module    : minimal_example.py
@Author    : Deco [deco@cubee.com]
@Created   : 4/26/18 11:16 AM
@Desc      : 
"""

import tensorflow as tf
import tensorflow_hub as hub
import time
import os

file_dir = os.path.dirname(os.path.abspath(__file__))

start_time = time.time()
# print(start_time)

embed = hub.Module("https://tfhub.dev/google/"
                   "universal-sentence-encoder/1")

if __name__ == '__main__':
    embedding = embed([
        "The quick brown fox jumps over the lazy dog."])
    print(embedding)
    print(type(embed))

    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.save(sess, os.path.join(file_dir, 'data/universal-sentence-encoder'))

    embed.export(os.path.join(file_dir, 'data/universal-sentence-encoder'),
                 sess)

    print(time.time()-start_time)
