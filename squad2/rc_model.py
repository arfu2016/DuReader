"""
@Project   : DuReader
@Module    : rc_model.py
@Author    : Deco [deco@cubee.com]
@Created   : 7/25/18 11:32 AM
@Desc      : 
"""
import json
import logging
import os
import time

import numpy as np
import tensorflow as tf

from mreading.layers.match_layer import AttentionFlowMatchLayer
from mreading.layers.match_layer import MatchLSTMLayer
from mreading.layers.pointer_net import PointerNetDecoder
from utils import compute_bleu_rouge
from utils import normalize


class RCModel:
    """
    Implements the main reading comprehension model.
    """

    def __init__(self, vocab, args):

        # logging
        logger = logging.getLogger("squad2.rc_model")
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s')
        if args.log_path:
            file_handler = logging.FileHandler(args.log_path)
            # 会通过命令行传进来args.log_path，或者用默认值
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        else:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        self.logger = logger

        # basic config
        self.algo = args.algo
        self.hidden_size = args.hidden_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay

        self.use_dropout = args.dropout_keep_prob < 1

        self.max_p_num = args.max_p_num
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
        self.max_a_len = args.max_a_len

        self.vocab = vocab

        sess_config = tf.ConfigProto()

        sess_config.gpu_options.allow_growth = True
        # 并不是把gpu一开始就全部占据，而是逐步增加占掉的显存和计算力
        self.sess = tf.Session(config=sess_config)

        self._build_graph()

        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        self._setup_placeholders()
        self._embed()
        self._encode()
        self._match()
        self._fuse()
        self._decode()
        self._compute_loss()
        self._create_train_op()
        self.logger.info(
            'Time to build graph: {} s'.format(time.time() - start_t))
        param_num = sum([np.prod(self.sess.run(tf.shape(v)))
                         for v in self.all_params])
        # 列表中数字的连乘，先乘后加，计算参数个数
        # parameter number of the model
        self.logger.info(
            'There are {} parameters in the model'.format(param_num))
        word_dim = self.sess.run(tf.shape(self.word_embeddings))
        self.logger.debug(
            'Word parameter number: {}'.format(word_dim[0] * word_dim[1]))

    def _setup_placeholders(self):
        """
        Placeholders, the variables that are not trainable
        一个变量，但不是在训练中会改变的那种，而且，在最后的计算中是要作为系数feed
        过去的
        """
        self.p = tf.placeholder(tf.int32, [None, None])
        self.q = tf.placeholder(tf.int32, [None, None])
        self.p_length = tf.placeholder(tf.int32, [None])
        self.q_length = tf.placeholder(tf.int32, [None])
        self.start_label = tf.placeholder(tf.int32, [None])
        self.end_label = tf.placeholder(tf.int32, [None])
        # 从实际回答到文中标注出来，比如用正则表达式，回答在文档中的什么位置，格式和实体
        # 抽取所用的格式是类似的
        self.dropout_keep_prob = tf.placeholder(tf.float32)

    def _embed(self):
        """
        The embedding layer, question and passage share embeddings
        只要词是一样的，embedding就是一样的
        """
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            # 此处指定使用cpu
            # 第一次建立这个variable_scope, 而不是reuse
            # reuse中最重要的是模型中的trainable variable的复用
            self.word_embeddings = tf.get_variable(
                'word_embeddings',
                shape=(self.vocab.size(), self.vocab.embed_dim),
                initializer=tf.constant_initializer(self.vocab.embeddings),
                # 把已经初始化好的embedding传过来
                trainable=True
            )
            # 生成variable，一般是可训练的；如果不可训练，就是始终使用pretrained
            # embedding
            self.p_emb = tf.nn.embedding_lookup(self.word_embeddings, self.p)
            # paragraph
            self.q_emb = tf.nn.embedding_lookup(self.word_embeddings, self.q)

    def _encode(self):
        with tf.variable_scope('passage_encoding'):
            self.sep_p_encodes = self.p_emb
            self.sep_q_encodes = self.q_emb

    def _match(self):
        """
        The core of RC model, get the question-aware passage
        encoding with either BIDAF or MLSTM
        The attention process
        文档的加权句向量，权重由问句决定
        """
        if self.algo == 'MLSTM':
            match_layer = MatchLSTMLayer(self.hidden_size)
        elif self.algo == 'BIDAF':
            match_layer = AttentionFlowMatchLayer(self.hidden_size)
        else:
            raise NotImplementedError(
                'The algorithm {} is not implemented.'.format(self.algo))
        self.match_p_encodes, _ = match_layer.match(self.sep_p_encodes,
                                                    self.sep_q_encodes,
                                                    self.p_length,
                                                    self.q_length)

        if self.use_dropout:
            self.match_p_encodes = tf.nn.dropout(self.match_p_encodes,
                                                 self.dropout_keep_prob)

    def _fuse(self):

        with tf.variable_scope('fusion'):
            self.fuse_p_encodes = self.match_p_encodes

    def _decode(self):
        """
        Employs Pointer Network to get the the probs of each position
        to be the start or end of the predicted answer.
        Note that we concat the fuse_p_encodes for the passages in the same
        document.
        And since the encodes of queries in the same document is same,
        we select the first one.
        """
        with tf.variable_scope('same_question_concat'):
            batch_size = tf.shape(self.start_label)[0]
            self.concat_passage_encodes = tf.reshape(
                self.fuse_p_encodes,
                [batch_size, -1, 2 * self.hidden_size]
            )
            # 维度调整，主要是第一维从batch_size*5调整为batch_size，
            # 第二维原来是passage的长度，现在变为之前值的5倍，因为有5段

            self.no_dup_question_encodes = tf.reshape(
                self.sep_q_encodes,
                [batch_size, -1, tf.shape(self.sep_q_encodes)[1],
                 2 * self.hidden_size])[0:, 0, 0:, 0:]
            # 维度调整，主要是第一维从batch_size*5调整为batch_size

        decoder = PointerNetDecoder(self.hidden_size)
        # self.fw_outputs, self.fw_outputs2, self.bw_outputs = \
        #     decoder.decode2(concat_passage_encodes, no_dup_question_encodes)
        # self.fw_cell, self.bw_cell, self.fw_cell1 = \
        #     decoder.decode2(concat_passage_encodes, no_dup_question_encodes)
        self.start_probs, self.end_probs = decoder.decode(
            self.concat_passage_encodes, self.no_dup_question_encodes)

    def _compute_loss(self):
        """
        The loss function
        计算损失函数就为了之后的参数优化，本质上是为了back propagation
        此处损失函数用的还是cross entropy，既考虑start loss，也考虑end loss，把
        二者结合起来。在做拟合和推测时，都用到start prob和end prob，但处理方法是不同的
        还有一种处理方法，更类似加强学习，就是拟合的结果不和answer相比，而是把目标
        函数变成start_prob*end_prob的最大化
        """

        def sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):
            """
            negative log likelyhood loss
            """
            with tf.name_scope(scope, "log_loss"):
                # 此处是name_scope
                labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
                # labels是具体位置的序号，此处要one hot encoding
                losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)
                # cross entropy的公式, + epsilon是为了处理probs为0的情况
            return losses

        self.start_loss = sparse_nll_loss(probs=self.start_probs,
                                          labels=self.start_label)
        self.end_loss = sparse_nll_loss(probs=self.end_probs,
                                        labels=self.end_label)
        # 要算两个cross entropy的loss，start和end各算一次
        self.all_params = tf.trainable_variables()
        self.loss = tf.reduce_mean(tf.add(self.start_loss, self.end_loss))
        # 优化的目标函数：两个loss的加和求最小值
        # 评估的时候使用的是bleu4和rouge，这里为了避免过拟合，并没有在loss function中
        # 针对bleu4和rouge做优化。如果做了优化，最后的评分肯定能更高，但不代表是更正确的
        # 模型
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
                # 做l2 regularization，使得拟合的weight不至于太大
            self.loss += self.weight_decay * l2_loss
            # self.weight_decay就是做l2 regularization时前面的那个系数，
            # 用来控制正则化的程度

    def _create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        if self.optim_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optim_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optim_type == 'rprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(
                self.learning_rate)
        else:
            raise NotImplementedError(
                'Unsupported optimizer: {}'.format(self.optim_type))
        self.train_op = self.optimizer.minimize(self.loss)
        # 直接上minimize函数最省事

    def _train_epoch(self, train_batches, dropout_keep_prob):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
            dropout_keep_prob: float value indicating dropout keep probability
        """
        total_num, total_loss = 0, 0
        # 这一个epoch中总共训练了多少样本
        log_every_n_batch, n_batch_loss = 50, 0
        # log_every_n_batch, n_batch_loss = 30, 0

        for bitx, batch in enumerate(train_batches, 1):
            # print('bitx in rc_model.py', bitx)
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.dropout_keep_prob: dropout_keep_prob}

            # self.logger.debug(self.sess.run(tf.shape(self.p), feed_dict))

            # self.logger.debug(self.sess.run(tf.shape(self.p_emb), feed_dict))
            # self.logger.debug(
            #     self.sess.run(
            #         tf.shape(self.concat_passage_encodes), feed_dict))
            # self.logger.debug(
            #     self.sess.run(
            #         tf.shape(self.no_dup_question_encodes), feed_dict))

            # self.logger.debug(batch['passage_token_ids'])
            # self.logger.debug(batch['passage_length'])
            # self.logger.debug(batch['start_id'])
            # self.logger.debug(batch['end_id'])
            # self.logger.debug(
            #     self.sess.run(tf.shape(self.start_probs), feed_dict))

            _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
            # variable自动更新，返回的也是更新后的variable，这里就不记录了

            total_loss += loss * len(batch['raw_data'])
            # loss是根据batch size平均后的结果，这里进行加总
            total_num += len(batch['raw_data'])
            # 累加batch size，或者最后一批剩下的数目
            # self.logger.info(
            # 'total_num in rc_model.py: {}'.format(total_num))
            n_batch_loss += loss
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info(
                    'Average loss from batch {} to {} is {}'.format(
                        bitx - log_every_n_batch + 1,
                        bitx, n_batch_loss / log_every_n_batch))
                n_batch_loss = 0
        return 1.0 * total_loss / total_num

    def train(self, data, epochs, batch_size, save_dir, save_prefix,
              dropout_keep_prob=1.0, evaluate=True):
        """
        Train the model with data
        Args:
            data: the BRCDataset class implemented in dataset.py
            epochs: number of training epochs
            batch_size:
            save_dir: the directory to save the model
            save_prefix: the prefix indicating the model type
            dropout_keep_prob: float value indicating dropout keep probability
            evaluate: whether to evaluate the model on test set after each
              epoch
        """
        pad_id = self.vocab.get_id(self.vocab.pad_token)
        # padding token的id
        max_bleu_4 = 0
        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            train_batches = data.gen_mini_batches('train', batch_size, pad_id,
                                                  shuffle=True)
            train_loss = self._train_epoch(train_batches, dropout_keep_prob)
            self.logger.info(
                'Average train loss for epoch {} is {}'.format(epoch,
                                                               train_loss))

            if evaluate:
                self.logger.info(
                    'Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches(
                        'dev', batch_size, pad_id, shuffle=False)
                    eval_loss, bleu_rouge = self.evaluate(eval_batches)
                    self.logger.info('Dev eval loss {}'.format(eval_loss))
                    self.logger.info('Dev eval result: {}'.format(bleu_rouge))

                    if bleu_rouge['Bleu-4'] > max_bleu_4:
                        self.save(save_dir, save_prefix)
                        # 保存模型
                        max_bleu_4 = bleu_rouge['Bleu-4']
                else:
                    self.logger.warning(
                        'No dev set is loaded for evaluation in the dataset!')
            else:
                self.save(save_dir, save_prefix + '_' + str(epoch))

    def evaluate(self, eval_batches, result_dir=None, result_prefix=None,
                 save_full_info=False):
        """
        Evaluates the model performance on eval_batches and results are saved
        if specified
        Args:
            eval_batches: iterable batch data
            result_dir: directory to save predicted answers,
            answers will not be saved if None
            result_prefix: prefix of the file for saving predicted answers,
                           answers will not be saved if None
            save_full_info: if True, the pred_answers will be added to
            raw sample and saved
        """
        pred_answers, ref_answers = [], []
        pred_refs = []
        total_loss, total_num = 0, 0
        for b_itx, batch in enumerate(eval_batches):
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.dropout_keep_prob: 1.0}
            # evaluate必然是没有dropout的
            start_probs, end_probs, loss = self.sess.run(
                [self.start_probs, self.end_probs, self.loss], feed_dict)

            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])

            padded_p_len = len(batch['passage_token_ids'][0])
            # self.p中最长的那个样本的长度？batch['passage_token_ids']应该是个
            # 多维的np.ndarray才对，有可能是当list of list来处理，就是第一行,
            # 也就是第一个样本

            # self.logger.info(len(batch['raw_data']))
            # self.logger.info(len(start_probs))
            # self.logger.info(len(end_probs))
            for sample, start_prob, end_prob in zip(batch['raw_data'],
                                                    start_probs, end_probs):

                best_answer = self.find_best_answer(sample, start_prob,
                                                    end_prob, padded_p_len)
                # self.logger.info(sample['question'])
                # self.logger.info(best_answer)
                # 在做evaluate和test的推测工作时，要这样利用start_prob和end_prob
                if save_full_info:
                    sample['pred_answers'] = [best_answer]
                    pred_answers.append(sample)
                else:
                    pred_answers.append({'question_id': sample['question_id'],
                                         'question_type':
                                             sample['question_type'],
                                         'question': sample['question'],
                                         'answers': [best_answer],
                                         'entity_answers': [[]],
                                         'yesno_answers': []})
                # if 'answers' in sample:
                if 'fake_answers' in sample:
                    ref_answers.append({'question_id': sample['question_id'],
                                        'question_type':
                                            sample['question_type'],
                                        # 'answers': sample['answers'],
                                        'answers': sample['fake_answers'],
                                        'entity_answers': [[]],
                                        'yesno_answers': []})

                pred_refs.append({
                                 'question': sample['question'],
                                 'predict_answer': best_answer,
                                 'real_answer': sample['fake_answers'],

                })

        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + '.json')
            with open(result_file, 'w', encoding='utf8') as fout:
                # for pred_answer in pred_answers:
                for pred_ref in pred_refs:
                    fout.write(json.dumps(pred_ref,  # pred_answer
                                          ensure_ascii=False) + '\n')

            self.logger.info(
                'Saving {} results to {}'.format(result_prefix, result_file))

        # this average loss is invalid on test set,
        # since we don't have true start_id and end_id
        ave_loss = 1.0 * total_loss / total_num

        # compute the bleu and rouge scores if reference answers is provided
        if len(ref_answers) > 0:
            pred_dict, ref_dict = {}, {}
            for pred, ref in zip(pred_answers, ref_answers):
                question_id = ref['question_id']
                if len(ref['answers']) > 0:
                    pred_dict[question_id] = normalize(pred['answers'])
                    # 利用utils包，normalize strings to space joined chars
                    ref_dict[question_id] = normalize(ref['answers'])
            bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
            # pred_dict是预测值，ref_dict是真实值
            # 利用utils包，calculate bleu and rouge metrics
        else:
            bleu_rouge = None
        return ave_loss, bleu_rouge

    def find_best_answer(self, sample, start_prob, end_prob, padded_p_len):
        """
        Finds the best answer for a sample given start_prob and end_prob for
        each position.
        This will call find_best_answer_for_passage because there are multiple
        passages in a sample
        """
        best_p_idx, best_span, best_score = None, None, 0
        for p_idx, passage in enumerate(sample['passages']):
            # 每篇document选了一段
            if p_idx >= self.max_p_num:
                continue
            passage_len = min(self.max_p_len, len(passage['passage_tokens']))
            # 如果passage长度超过self.max_p_len，在这儿会进行处理
            answer_span, score = self.find_best_answer_for_passage(
                start_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                end_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                passage_len)
            if score > best_score:
                # 来自不同文章的答案可以比较，因为能算出score
                best_score = score
                best_p_idx = p_idx
                best_span = answer_span
        if best_p_idx is None or best_span is None:
            best_answer = ''
        else:
            best_answer = ' '.join(
                sample['passages'][best_p_idx]['passage_tokens'][
                    best_span[0]: best_span[1] + 1])
            # best_span就是个2个元素的向量
        return best_answer

    def find_best_answer_for_passage(self, start_probs, end_probs,
                                     passage_len=None):
        """
        Finds the best answer with the maximum start_prob * end_prob
        from a single passage
        passage_len是start和end之间最长的长度？
        """
        if passage_len is None:
            passage_len = len(start_probs)
        else:
            passage_len = min(len(start_probs), passage_len)
            # 又是取最小处理
        best_start, best_end, max_prob = -1, -1, 0
        for start_idx in range(passage_len):
            for ans_len in range(self.max_a_len):
                end_idx = start_idx + ans_len
                # end_idx必须在start_idx后面
                if end_idx >= passage_len:
                    continue
                    # start_idx与end_idx的间隔不能太长
                prob = start_probs[start_idx] * end_probs[end_idx]
                if prob > max_prob:
                    best_start = start_idx
                    best_end = end_idx
                    max_prob = prob
        return (best_start, best_end), max_prob
    # 果然是只有两个元素的tuple

    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info(
            'Model saved in {}, with prefix {}.'.format(model_dir,
                                                        model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model
        indicator
        """
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info(
            'Model restored from {}, with prefix {}'.format(model_dir,
                                                            model_prefix))
