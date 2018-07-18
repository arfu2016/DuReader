"""
@Project   : DuReader
@Module    : lstm_testing.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/18/18 10:24 AM
@Desc      :
tensorflow==1.1
"""
import numpy as np
import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# 2压制warning，只输出error以上级别

file_dir = os.path.dirname(os.path.abspath(__file__))
fn_review = os.path.join(file_dir, 'data/reviews.txt')
fn_label = os.path.join(file_dir, 'data/labels.txt')

with open(fn_review, 'r', encoding='utf-8') as f:
    reviews = f.read()
with open(fn_label, 'r', encoding='utf-8') as f:
    labels = f.read()

from string import punctuation
all_text = ''.join([c for c in reviews if c not in punctuation])
reviews = all_text.split('\n')

all_text = ' '.join(reviews)
words = all_text.split()

from collections import Counter
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

non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]

reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
labels = np.array([labels[ii] for ii in non_zero_idx])

seq_len = 200
features = np.zeros((len(reviews_ints), seq_len), dtype=int)
for i, row in enumerate(reviews_ints):
    features[i, -len(row):] = np.array(row)[:seq_len]

split_frac = 0.8
split_idx = int(len(features)*0.8)
train_x, val_x = features[:split_idx], features[split_idx:]
train_y, val_y = labels[:split_idx], labels[split_idx:]

test_idx = int(len(val_x)*0.5)
val_x, test_x = val_x[:test_idx], val_x[test_idx:]
val_y, test_y = val_y[:test_idx], val_y[test_idx:]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape),
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))

lstm_size = 256
lstm_layers = 1
batch_size = 500
learning_rate = 0.001

n_words = len(vocab_to_int) + 1
# Adding 1 because we use 0's for padding, dictionary started at 1
# n_words把表示padding的字符也算了进来，所以加了1

# Create the graph object
graph = tf.Graph()
# Add nodes to the graph
with graph.as_default():
    inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
    labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# Size of the embedding vectors (number of units in the embedding layer)
embed_size = 300

with graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs_)

with graph.as_default():
    # Your basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

    # Add dropout to the cell
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)

    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)

with graph.as_default():
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed,
                                             initial_state=initial_state)

    print('outputs.get_shape():', outputs.get_shape())
    print('outputs[:, -1]', outputs[:, -1])
    # tf.shape()得到的是一个tensor
    # get_shape()得到的是一个nd.array

with graph.as_default():
    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1,
                                                    activation_fn=tf.sigmoid)
    cost = tf.losses.mean_squared_error(labels_, predictions)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with graph.as_default():
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def get_batches(x, y, batch_size=100):
    n_batches = len(x) // batch_size
    x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii + batch_size], y[ii:ii + batch_size]


with graph.as_default():
    saver = tf.train.Saver()

test_acc = []
with tf.Session(graph=graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

    all_params = tf.trainable_variables()
    param = [v for v in all_params]
    param_name = [v.name for v in all_params]
    param_shape = [sess.run(tf.shape(v)) for v in all_params]
    embedding_param_num = np.prod(param_shape[0])
    lstm_param_num = sum([np.prod(v) for v in param_shape[1:3]])
    param_num = sum([np.prod(sess.run(tf.shape(v))) for v in all_params])

    print('Parameter tensors:', param)
    print('Names of parameter tensors:', param_name)
    print('Shapes of parameter tensors:', param_shape)
    print('Number of embedding parameters:', embedding_param_num)
    print('Number of lstm parameters:', lstm_param_num)
    print('Total number of parameters:', param_num)

    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
    for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
        feed = {inputs_: x,
                labels_: y[:, None],
                keep_prob: 1,
                initial_state: test_state}
        batch_acc, output, f_state = sess.run(
            [accuracy, outputs, final_state], feed_dict=feed)
        test_acc.append(batch_acc)
    print('ii:', ii)
    print('y.shape:', y.shape)  # 一维
    print('y.shape:', y[:, None].shape)  # 二维
    print('output.shape:', output.shape)
    print('f_state.shape:', [f_state[0].c.shape, f_state[0].h.shape])
    print("Test accuracy: {:.3f}".format(np.mean(test_acc)))
