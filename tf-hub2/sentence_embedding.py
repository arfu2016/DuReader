"""
@Project   : DuReader
@Module    : sentence_embedding.py
@Author    : Deco [deco@cubee.com]
@Created   : 4/26/18 2:46 PM
@Desc      : 
"""
import tensorflow as tf
import tensorflow_hub as hub
import os
import numpy as np

if __name__ == '__main__':
    file_dir = os.path.dirname(os.path.abspath(__file__))
    # sess = tf.Session()
    # saver = tf.train.Saver()
    # sess.run(tf.global_variables_initializer())
    # saver.restore(sess,
    #               os.path.join(file_dir, 'data/universal-sentence-encoder'))
    # https://stackoverflow.com/questions/34982492/restoring-tensorflow-model

    embed = hub.Module(os.path.join(file_dir,
                                    'data/universal-sentence-encoder'))

    embedding = embed([
        "The quick brown fox jumps over the lazy dog.",
        "Who is Messy"])

    with tf.Session() as session:
        session.run(
            [tf.global_variables_initializer(), tf.tables_initializer()])
        message_embedding = session.run(embedding)

    print(message_embedding)
    print(message_embedding.shape)
    print(type(message_embedding))
    print(np.linalg.norm(message_embedding[0, :]))
    print(np.linalg.norm(message_embedding[1, :]))
