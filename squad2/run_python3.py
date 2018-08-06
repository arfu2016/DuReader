"""
@Project   : DuReader
@Module    : run_python3.py
@Author    : Deco [deco@cubee.com]
@Created   : 7/23/18 5:47 PM
@Desc      : 
"""

import argparse
import json
import logging
import os
import pickle
import random
import sys
from importlib import import_module

import tensorflow as tf

base_dir = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)))

try:

    from mreading.dataset import BRCDataset
    from mreading.vocab import Vocab
    from mreading.rc_model import RCModel

except ImportError:

    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
        # 以base_dir为基准开始导入

    module_dataset = import_module('.dataset', package='mreading')
    module_vocab = import_module('.vocab', package='mreading')
    module_rc_model = import_module('.rc_model', package='mreading')

    BRCDataset = getattr(module_dataset, 'BRCDataset')
    Vocab = getattr(module_vocab, 'Vocab')
    RCModel = getattr(module_rc_model, 'RCModel')

os.chdir(os.path.join(base_dir, 'squad2'))
# 改变当前目录，因为后面要用到父目录，祖父目录
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# 设置环境变量，控制tensorflow的log level


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(
        'Reading Comprehension on BaiduRC dataset')

    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, '
                             'prepare the vocabulary and embeddings')
    # args = parser.parse_args()
    # args.prepare is available
    # when action='store_true' and --prepare exists, args.prepare is True

    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set '
                             'with trained model')

    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    # 算loss的时候，要不要加l2 regularization，默认不加
    train_settings.add_argument('--dropout_keep_prob', type=float, default=1,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=32,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=10,
                                help='train epochs')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', choices=['BIDAF', 'MLSTM'],
                                default='MLSTM',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_size', type=int, default=300,
                                help='size of the embeddings')
    # 可以调参，默认300
    model_settings.add_argument('--hidden_size', type=int, default=150,
                                help='size of LSTM hidden units')
    # 可以调参，默认150
    model_settings.add_argument('--max_p_num', type=int, default=5,
                                help='max passage num in one sample')
    # 最多5个document备选
    model_settings.add_argument('--max_p_len', type=int, default=500,
                                help='max length of passage')
    # passage长度最多500？似乎看到过2500，500以后的tf计算时都不考虑
    model_settings.add_argument('--max_q_len', type=int, default=60,
                                help='max length of question')
    # 问题长度最长60
    model_settings.add_argument('--max_a_len', type=int, default=200,
                                help='max length of answer')
    # 回答长度最长200

    path_settings = parser.add_argument_group('path settings')

    path_settings.add_argument(
        '--train_files', nargs='+',
        default=['../data/demo/trainset/search.train.json'],
        help='list of files that contain the preprocessed train data')
    # nargs='+'表示--train_files之后可以有一个或者多个参数

    path_settings.add_argument(
        '--dev_files', nargs='+',
        default=['../data/demo/devset/search.dev.json'],
        help='list of files that contain the preprocessed dev data')

    path_settings.add_argument(
        '--test_files', nargs='+',
        default=['../data/demo/testset/search.test.json'],
        help='list of files that contain the preprocessed test data')

    path_settings.add_argument(
        '--brc_dir', default='../data/baidu',
        help='the dir with preprocessed baidu reading comprehension data')

    path_settings.add_argument('--vocab_dir', default='../data/vocab/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir',
                               default='../data/models/simple_preprocess/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir',
                               default='../data/results/simple_preprocess/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir',
                               default='../data/summary/simple_preprocess/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, '
                                    'logs are printed to console')
    return parser.parse_args()


def prepare(args):
    """
    checks data, creates the directories, prepare the vocabulary and embeddings
    """
    logger = logging.getLogger("squad2.run_python3")
    logger.info('Checking the data files...')
    for data_path in args.train_files + args.dev_files + args.test_files:
        assert os.path.exists(data_path), \
            '{} file does not exist.'.format(data_path)
        # 对输入的容错
    logger.info('Preparing the directories...')
    for dir_path in [args.vocab_dir, args.model_dir, args.result_dir,
                     args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    logger.info('Building vocabulary...')
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          args.train_files, args.dev_files, args.test_files)
    # 数据列表的准备
    vocab = Vocab(lower=True)
    # obtain token2id, id2token, token_cnt
    for word in brc_data.word_iter('train'):
        # we yield words from a generator
        vocab.add(word)

    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt=2)
    # 出现频数少于2次的不做统计
    filtered_num = unfiltered_vocab_size - vocab.size()
    # 被过滤掉的词的数目
    logger.info('After filter {} tokens, the final vocab size is {}'.
                format(filtered_num, vocab.size()))

    logger.info('Assigning embeddings...')
    vocab.randomly_init_embeddings(args.embed_size)
    # 随机分配args.embed_size维度大小的word embedding

    logger.info('Saving vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'wb') as fout:
        pickle.dump(vocab, fout)
        # serialize vocab and store it in a file

    logger.info('Done with preparing!')


def train(args, restore=True):
    """
    trains the reading comprehension model
    """
    logger = logging.getLogger("mreading.run_python3")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          args.train_files, args.dev_files)
    # 准备training data
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    # convert tokens of questions and paragraphs in training data to ids
    # 结果保存在brc_data中
    logger.info('Initialize the model...')
    rc_model = RCModel(vocab, args)
    if restore:
        try:
            rc_model.restore(model_dir=args.model_dir,
                             model_prefix=args.algo)
            # todo: 上面这句可能需要改，model_prefix=args.algo + '_' + str(2)
        # except Exception as e:
        except tf.errors.InvalidArgumentError:
            # logger.info('Exception in train() in run_python3.py', e)
            logger.info('InvalidArgumentError... '
                        'Initialize the model from beginning')
            # str(e) or repr(e)
        except Exception:
            logger.info('Unknown exception. '
                        'Initialize the model from beginning')

    logger.info('Training the model...')
    rc_model.train(brc_data, args.epochs, args.batch_size,
                   save_dir=args.model_dir,
                   save_prefix=args.algo,
                   dropout_keep_prob=args.dropout_keep_prob)
    logger.info('Done with model training!')


def evaluate(args):
    """
    evaluate the trained model on dev files
    在改变超参数时可以参考
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.dev_files) > 0, 'No dev files are provided.'
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          dev_files=args.dev_files)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Restoring the model...')
    rc_model = RCModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir,
                     model_prefix=args.algo)
    # todo: 上面这句可能需要改，model_prefix=args.algo + '_' + str(2)
    logger.info('Evaluating the model on dev set...')
    dev_batches = brc_data.gen_mini_batches('dev', args.batch_size,
                                            pad_id=vocab.get_id(
                                                vocab.pad_token),
                                            shuffle=False)
    dev_loss, dev_bleu_rouge = rc_model.evaluate(
        dev_batches, result_dir=args.result_dir, result_prefix='dev.predicted')
    logger.info('Loss on dev set: {}'.format(dev_loss))
    logger.info('Result on dev set: {}'.format(dev_bleu_rouge))
    logger.info(
        'Predicted answers are saved to {}'.format(
            os.path.join(args.result_dir)))


def predict(args):
    """
    predicts answers for test files
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.test_files) > 0, 'No test files are provided.'
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          test_files=args.test_files)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Restoring the model...')
    rc_model = RCModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)
    # todo: 上面这句可能需要改，model_prefix=args.algo + '_' + str(2)
    logger.info('Predicting answers for test set...')
    test_batches = brc_data.gen_mini_batches('test', args.batch_size,
                                             pad_id=vocab.get_id(
                                                 vocab.pad_token),
                                             shuffle=False)
    rc_model.evaluate(test_batches,
                      result_dir=args.result_dir,
                      result_prefix='test.predicted')
    # 同样使用evaluate函数

    result_dir = args.result_dir
    question_answer = list()
    answer_string = 'Question and answer for testing:\n'

    if result_dir is not None:
        result_file = os.path.join(result_dir, 'test.predicted.json')
        with open(result_file, 'r', encoding='utf8') as fin:
            for line in fin:
                answer_dict = json.loads(line.strip())
                question_answer.append((answer_dict['question'],
                                        answer_dict['answers']))
        answer_samples = random.sample(question_answer, 10)
        for sample in answer_samples:
            answer_string += '{}: {}\n'.format(sample[0], sample[1])
        logger.info(answer_string)


def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()

    logger = logging.getLogger("mreading.run_python3")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        # 会通过命令行传进来args.log_path，或者用默认值
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # https://stackoverflow.com/questions/13781738/how-does-cuda-assign-device-ids-to-gpus?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # 指定使用哪个或者哪些gpu，当使用两个以上时，显存似乎主要还是用第一个gpu的显存，另一个gpu只提供计算上的帮助，不提供显存上的帮助

    if args.prepare:
        prepare(args)
    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)
    if args.predict:
        predict(args)


if __name__ == '__main__':
    run()
