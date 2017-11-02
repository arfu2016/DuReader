# -*- coding:utf8 -*-
###############################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
###############################################################################
"""
This module implements the match-lstm algorithm described in
https://arxiv.org/abs/1608.07905

Authors: liuyuan(liuyuan04@baidu.com)
Data: 2017/09/20 12:00:00
"""

from paddle.trainer.config_parser import default_initial_std
import paddle.v2.layer as layer
import paddle.v2.attr as Attr
import paddle.v2.activation as Act
import paddle.v2 as paddle

from qa_model import QAModel


class MatchLstm(QAModel):
    """
    Implements Match-LSTM.
    """
    def get_enc(self, input, type='q'):
        """
        Encodes the input by feeding it into a bidirectional lstm and
        concatenates the forward and backward expression of each time step.
        """
        embs = self.get_embs(input)
        enc = paddle.networks.bidirectional_lstm(
              input=embs,
              size=self.emb_dim,
              fwd_mat_param_attr=Attr.Param('f_enc_mat.w' + type),
              fwd_bias_param_attr=Attr.Param('f_enc.bias' + type,
                  initial_std=0.),
              fwd_inner_param_attr=Attr.Param('f_enc_inn.w' + type),
              bwd_mat_param_attr=Attr.Param('b_enc_mat.w' + type),
              bwd_bias_param_attr=Attr.Param('b_enc.bias' + type,
                  initial_std=0.),
              bwd_inner_param_attr=Attr.Param('b_enc_inn.w' + type),
              return_seq=True)
        enc_dropped = self.drop_out(enc, drop_rate=0.5)
        return enc_dropped

    def __attention(self, direct, cur_token, prev, to_apply, to_apply_proj):
        with layer.mixed(size=cur_token.size,
                         bias_attr=Attr.Param(direct + '.bp',
                             initial_std=0.),
                         act=Act.Linear()) as proj:
            proj += layer.full_matrix_projection(
                    input=cur_token,
                    param_attr=Attr.Param(direct + '.wp'))
            proj += layer.full_matrix_projection(
                    input=prev,
                    param_attr=Attr.Param(direct + '.wr'))

        expanded = layer.expand(input=proj, expand_as=to_apply)
        att_context = layer.addto(input=[expanded, to_apply_proj],
                                  act=Act.Tanh(),
                                  bias_attr=False)

        att_weights = layer.fc(input=att_context,
                               param_attr=Attr.Param(direct + '.w'),
                               bias_attr=Attr.Param(direct + '.b',
                                   initial_std=0.),
                               act=Act.SequenceSoftmax(),
                               size=1)
        scaled = layer.scaling(input=to_apply, weight=att_weights)
        applied = layer.pooling(input=scaled,
                                pooling_type=paddle.pooling.Sum())
        return applied

    def __step(self, name, h_q_all, q_proj, h_p_cur):
        """
        Match-LSTM step. This function performs operations done in one
        time step.

        Args:
            h_p_cur: Current hidden of paragraph encodings: h_i.
                     This is the `REAL` input of the group, like
                     x_t in normal rnn.
            h_q_all: Question encodings.

        Returns:
            The $h^{r}_{i}$ in the paper.
        """
        direct = 'left' if 'left' in name else 'right'

        h_r_prev = paddle.layer.memory(name=name + '_out_',
                                       size=h_q_all.size,
                                       boot_layer=None)
        q_expr = self.__attention(direct, h_p_cur, h_r_prev, h_q_all, q_proj)
        z_cur = self.__fusion_layer(h_p_cur, q_expr)

        with layer.mixed(size=h_q_all.size * 4,
                         act=Act.Tanh(),
                         bias_attr=False) as match_input:
            match_input += layer.full_matrix_projection(
                           input=z_cur,
                           param_attr=Attr.Param('match_input_%s.w0' % direct))

        step_out = paddle.networks.lstmemory_unit(
                   name=name + '_out_',
                   out_memory=h_r_prev,
                   param_attr=Attr.Param('step_lstm_%s.w' % direct),
                   input_proj_bias_attr=Attr.Param(
                       'step_lstm_mixed_%s.bias' % direct,
                       initial_std=0.),
                   lstm_bias_attr=Attr.Param('step_lstm_%s.bias' % direct,
                       initial_std=0.),
                   #input=match_input,
                   input=match_input,
                   size=h_q_all.size)
        return step_out

    def __fusion_layer(self, input1, input2):
        # fusion layer
        neg_input2 = layer.slope_intercept(input=input2,
                slope=-1.0,
                intercept=0.0)
        diff1 = layer.addto(input=[input1, neg_input2],
                act=Act.Identity(),
                bias_attr=False)
        diff2 = layer.mixed(bias_attr=False,
                input=layer.dotmul_operator(a=input1, b=input2))

        fused = layer.concat(input=[input1, input2, diff1, diff2])
        return fused

    def recurrent_group(self, name, inputs, reverse=False):
        """
        Implements the Match-LSTM layer in the paper.

        Args:
            name: the name prefix of the layers created by this method.
            inputs: the inputs takes by the __step method.
            reverse: True if the paragraph encoding is processed from right
                     to left, otherwise the paragraph encoding is processed
                     from left to right.
        Returns:
            The Match-LSTM layer's output of one direction.
        """
        inputs.insert(0, name)
        seq_out = layer.recurrent_group(name=name,
                                        input=inputs,
                                        step=self.__step,
                                        reverse=reverse)
        return seq_out

    def network(self):
        """
        Implements the whole network of Match-LSTM.

        Returns:
            A tuple of LayerOutput objects containing the start and end
            probability distributions respectively.
        """
        self.check_and_create_data()
        self.create_shared_params()
        q_enc = self.get_enc(self.q_ids, type='q')
        p_encs = []
        p_matches_start = []
        p_matches_end = []
        p_matches = []
        for p in self.p_ids:
            p_encs.append(self.get_enc(p, type='p'))

        q_proj_left = layer.fc(size=self.emb_dim * 2,
                               bias_attr=False,
                               param_attr=Attr.Param(
                                   self.name + '_left_' + '.wq'),
                               input=q_enc)
        q_proj_right = layer.fc(size=self.emb_dim * 2,
                                bias_attr=False,
                                param_attr=Attr.Param(
                                    self.name + '_right_' + '.wq'),
                                input=q_enc)
        for i, p in enumerate(p_encs):
            left_out = self.recurrent_group(
                       self.name + '_left_' + str(i),
                       [layer.StaticInput(q_enc),
                           layer.StaticInput(q_proj_left), p],
                       reverse=False)
            right_out = self.recurrent_group(
                        self.name + '_right_' + str(i),
                        [layer.StaticInput(q_enc),
                            layer.StaticInput(q_proj_right), p],
                        reverse=True)
            match_seq = layer.concat(input=[left_out, right_out])
            match_seq_dropped = self.drop_out(match_seq, drop_rate=0.5)
            bi_match_seq = paddle.networks.bidirectional_lstm(
                    input=match_seq_dropped,
                    size=match_seq.size,
                    fwd_mat_param_attr=Attr.Param('pn_f_enc_mat.w'),
                    fwd_bias_param_attr=Attr.Param('pn_f_enc.bias',
                        initial_std=0.),
                    fwd_inner_param_attr=Attr.Param('pn_f_enc_inn.w'),
                    bwd_mat_param_attr=Attr.Param('pn_b_enc_mat.w'),
                    bwd_bias_param_attr=Attr.Param('pn_b_enc.bias',
                        initial_std=0.),
                    bwd_inner_param_attr=Attr.Param('pn_b_enc_inn.w'),
                    return_seq=True)
            #bi_match_seq_drop = self.drop_out(bi_match_seq, drop_rate=0.5)
            p_matches.append(bi_match_seq)

        all_docs = reduce(lambda x, y: layer.seq_concat(a=x, b=y),
                    p_matches)
        all_docs_dropped = self.drop_out(all_docs, drop_rate=0.5)
        start = self.decode('start', all_docs_dropped)
        end = self.decode('end', all_docs_dropped)
        return start, end

    def decode_with_context(self, name, match_seq, attention):
        """
        Decode with context information.
        """
        scaled = layer.scaling(input=match_seq, weight=attention)
        context = layer.pooling(input=scaled,
                pooling_type=paddle.pooling.Sum())
        expanded = layer.expand(input=context, expand_as=match_seq)
        match_seq_new = self.__fusion_layer(match_seq, expanded)
        probs = self.decode(name, match_seq_new)
        return probs