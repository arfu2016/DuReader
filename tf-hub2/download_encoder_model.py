"""
@Project   : DuReader
@Module    : download_encoder_model.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/3/18 3:22 PM

@Desc      :
The Universal Sentence Encoder encodes text into high dimensional vectors
that can be used for text classification, semantic similarity, clustering
and other natural language tasks.
https://www.tensorflow.org/hub/modules/google/universal-sentence-encoder/1
"""

import tensorflow as tf
import tensorflow_hub as hub
import os

file_dir = os.path.dirname(os.path.abspath(__file__))

embed = hub.Module("https://tfhub.dev/google/"
                   "universal-sentence-encoder/1")

sess = tf.Session()
embed.export(os.path.join(file_dir, 'data/universal-sentence-encoder'),
             sess)

embedding = embed([
    "The quick brown fox jumps over the lazy dog.",
    "Who is Messy"])
message_embedding = sess.run(embedding)

print('message_embedding:', message_embedding)
print('Array shape of message_embedding:', message_embedding.shape)
