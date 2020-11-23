#!/usr/bin/env python
import torch


def reps_dot(sent1_reps: torch.Tensor, sent2_reps: torch.Tensor) -> torch.Tensor:
    """
    calculate representation dot production
    :param sent1_reps: (N, sent1_len, reps_dim)
    :param sent2_reps: (N, sent2_len, reps_dim)
    :return: (N, sent1_len, sent2_len)
    """
    return torch.bmm(sent1_reps, torch.transpose(sent2_reps, -1, -2))  # shape: (N, seq_len1, seq_len2)


def reps_l2_sim(sent1_reps: torch.Tensor, sent2_reps: torch.Tensor) -> torch.Tensor:
    """
    calculate representation L2 similarity
    :param sent1_reps: (N, sent1_len, reps_dim)
    :param sent2_reps: (N, sent2_len, reps_dim)
    :return: (N, sent1_len, sent2_len)
    """
    sent1_len = sent1_reps.shape[-2]
    sent2_len = sent2_reps.shape[-2]

    expand_shape1 = list(sent2_reps.shape)
    expand_shape1.insert(2, sent2_len)
    expand_shape2 = list(sent2_reps.shape)
    expand_shape2.insert(1, sent1_len)

    # shape: (N, seq_len1, seq_len2, emb_dim)
    expand_reps1 = sent1_reps.unsqueeze(2).expand(expand_shape1)
    expand_reps2 = sent2_reps.unsqueeze(1).expand(expand_shape2)

    # shape: (N, seq_len1, seq_len2)
    sim = torch.norm(expand_reps1 - expand_reps2, dim=-1, p=2)
    return -sim  # we calculate similarity not distance here


def reps_cosine_sim(sent1_reps: torch.Tensor, sent2_reps: torch.Tensor) -> torch.Tensor:
    """
    calculate representation cosine similarity, note that this is different from torch version(that compute parwisely)
    :param sent1_reps: (N, sent1_len, reps_dim)
    :param sent2_reps: (N, sent2_len, reps_dim)
    :return: (N, sent1_len, sent2_len)
    """
    dot_sim = torch.bmm(sent1_reps, torch.transpose(sent2_reps, -1, -2))  # shape: (batch, seq_len1, seq_len2)
    sent1_reps_norm = torch.norm(sent1_reps, dim=-1, keepdim=True)  # shape: (batch, seq_len1, 1)
    sent2_reps_norm = torch.norm(sent2_reps, dim=-1, keepdim=True)  # shape: (batch, seq_len2, 1)
    norm_product = torch.bmm(sent1_reps_norm,
                             torch.transpose(sent2_reps_norm, -1, -2))  # shape: (batch, seq_len1, seq_len2)
    sim_predicts = dot_sim / norm_product  # shape: (batch, seq_len1, seq_len2)
    return sim_predicts


class SimilarityScorerBase(torch.nn.Module):
    def __init__(self, sim_func, emb_log=None):
        super(SimilarityScorerBase, self).__init__()
        self.sim_func = sim_func
        self.emb_log = emb_log
        self.log_content = ''

    def forward(
            self,
            test_reps: torch.Tensor,
            support_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_output_mask: torch.Tensor,
            support_targets: torch.Tensor = None,
            label_reps: torch.Tensor = None, ) -> torch.Tensor:
        """
            :param test_reps: (batch_size, support_size, test_seq_len, dim)
            :param support_reps: (batch_size, support_size, support_seq_len)
            :param test_output_mask: (batch_size, test_seq_len)
            :param support_output_mask: (batch_size, support_size, support_seq_len)
            :param support_targets: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
            :param label_reps: (batch_size, num_tags, dim)
            :return: similarity
        """
        raise NotImplementedError()

    def mask_sim(self, sim: torch.Tensor, mask1: torch.Tensor, mask2: torch.Tensor):
        """
        mask invalid similarity to 0, i.e. sim to pad token is 0 here.
        :param sim: similarity matrix (num sim, test_len, support_len)
        :param mask1: (num sim, test_len, support_len)
        :param mask2: (num sim, test_len, support_len)
        :param min_value: the minimum value for similarity score
        :return:
        """
        mask1 = mask1.unsqueeze(-1).float()  # (b * s, test_label_num, 1)
        mask2 = mask2.unsqueeze(-1).float()  # (b * s, support_label_num, 1)
        mask = reps_dot(mask1, mask2)  # (b * s, test_label_num, support_label_num)
        sim = sim * mask
        return sim

    def expand_it(self, item: torch.Tensor, support_size):
        item = item.unsqueeze(1)
        expand_shape = list(item.shape)
        expand_shape[1] = support_size
        return item.expand(expand_shape)


class MatchingSimilarityScorer(SimilarityScorerBase):
    def __init__(self, sim_func, emb_log=None):
        super(MatchingSimilarityScorer, self).__init__(sim_func=sim_func, emb_log=emb_log)

    def forward(
            self,
            test_reps: torch.Tensor,
            support_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_output_mask: torch.Tensor,
            support_targets: torch.Tensor = None,
            label_reps: torch.Tensor = None, ) -> torch.Tensor:
        """
            calculate similarity between test token and support tokens.
            :param test_reps: (batch_size, support_size, test_seq_len, dim)
            :param support_reps: (batch_size, support_size, support_seq_len)
            :param test_output_mask: (batch_size, test_seq_len)
            :param support_output_mask: (batch_size, support_size, support_seq_len)
            :param support_targets: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
            :param label_reps: (batch_size, num_tags, dim)
            :return: similarity matrix: (batch_size, support_size, test_seq_len, support_seq_len)
        """
        support_size = support_reps.shape[1]
        test_len = test_reps.shape[-2]  # non-word-piece max test token num, Notice that it's different to input t len
        support_len = support_reps.shape[-2]  # non-word-piece max test token num
        emb_dim = test_reps.shape[-1]
        batch_size = test_reps.shape[0]
        # flatten representations to shape (batch_size * support_size, sent_len, emb_dim)
        test_reps = test_reps.view(-1, test_len, emb_dim)
        support_reps = support_reps.view(-1, support_len, emb_dim)

        # calculate dot product
        sim_score = self.sim_func(test_reps, support_reps)

        # the length in sc is `1` which is not same as sl
        test_mask = self.expand_it(test_output_mask, support_size).contiguous().view(batch_size * support_size, -1)
        support_mask = support_output_mask.contiguous().view(batch_size * support_size, -1)
        sim_score = self.mask_sim(sim_score, mask1=test_mask, mask2=support_mask)

        # reshape from (batch_size * support_size, test_len, support_len) to
        # (batch_size, support_size, test_len, support_len)
        sim_score = sim_score.view(batch_size, support_size, test_len, support_len)
        return sim_score


class PrototypeSimilarityScorer(SimilarityScorerBase):
    def __init__(self, sim_func, emb_log=None):
        super(PrototypeSimilarityScorer, self).__init__(sim_func=sim_func, emb_log=emb_log)

    def forward(
            self,
            test_reps: torch.Tensor,
            support_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_output_mask: torch.Tensor,
            support_targets: torch.Tensor = None,
            label_reps: torch.Tensor = None, ) -> torch.Tensor:
        """
            calculate similarity between token and each label's prototype.
            :param test_reps: (batch_size, support_size, test_seq_len, dim)
            :param support_reps: (batch_size, support_size, support_seq_len)
            :param test_output_mask: (batch_size, test_seq_len)
            :param support_output_mask: (batch_size, support_size, support_seq_len)
            :param support_targets: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
            :param label_reps: (batch_size, num_tags, dim)
            :return: similarity: (batch_size, test_seq_len, num_tags)
        """
        support_size = support_reps.shape[1]
        test_len = test_reps.shape[-2]  # non-word-piece max test token num, Notice that it's different to input t len
        support_len = support_reps.shape[-2]  # non-word-piece max test token num or possible lb num for sc (fix `1`)
        emb_dim = test_reps.shape[-1]
        batch_size = test_reps.shape[0]
        num_tags = support_targets.shape[-1]

        # average test representation over support set (reps for each support sent can be different)
        test_reps = torch.mean(test_reps, dim=1)  # shape (batch_size, sent_len, emb_dim)
        # flatten dim mention of support size and batch size.
        # shape (batch_size * support_size, sent_len, emb_dim)
        support_reps = support_reps.view(-1, support_len, emb_dim)
        # shape (batch_size * support_size, sent_len, num_tags)
        support_targets = support_targets.view(batch_size * support_size, support_len, num_tags).float()
        # get prototype reps
        # shape (batch_size, support_size, num_tags, emd_dim)
        sum_reps = torch.bmm(torch.transpose(support_targets, -1, -2), support_reps)
        # sum up tag emb over support set, shape (batch_size, num_tags, emd_dim)
        sum_reps = torch.sum(sum_reps.view(batch_size, support_size, num_tags, emb_dim), dim=1)
        # get num of each tag in support set, shape: (batch_size, num_tags, 1)
        tag_count = torch.sum(support_targets.view(batch_size, -1, num_tags), dim=1).unsqueeze(-1)
        tag_count = self.remove_0(tag_count)

        prototype_reps = torch.div(sum_reps, tag_count)  # shape (batch_size, num_tags, emd_dim)

        # calculate dot product
        sim_score = self.sim_func(test_reps, prototype_reps)  # shape (batch_size, sent_len, num_tags)

        return sim_score

    def remove_0(self, my_tensor):
        return my_tensor + 0.0001


class ProtoWithLabelSimilarityScorer(SimilarityScorerBase):
    def __init__(self, sim_func, scaler=None, emb_log=None):
        super(ProtoWithLabelSimilarityScorer, self).__init__(sim_func=sim_func, emb_log=emb_log)
        self.scaler = scaler
        self.emb_log = emb_log
        self.idx = 0

    def forward(
            self,
            test_reps: torch.Tensor,
            support_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_output_mask: torch.Tensor,
            support_targets: torch.Tensor = None,
            label_reps: torch.Tensor = None,) -> torch.Tensor:
        """
            calculate similarity between token and each label's prototype.
            :param test_reps: (batch_size, support_size, test_seq_len, dim)
            :param support_reps: (batch_size, support_size, support_seq_len)
            :param test_output_mask: (batch_size, test_seq_len)
            :param support_output_mask: (batch_size, support_size, support_seq_len)
            :param support_targets: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
            :param label_reps: (batch_size, num_tags, dim)
            :return: similarity: (batch_size, test_seq_len, num_tags)
        """
        '''get data attribute'''
        support_size = support_reps.shape[1]
        test_len = test_reps.shape[-2]  # non-word-piece max test token num, Notice that it's different to input t len
        support_len = support_reps.shape[-2]  # non-word-piece max test token num
        emb_dim = test_reps.shape[-1]
        batch_size = test_reps.shape[0]
        num_tags = support_targets.shape[-1]

        '''get prototype reps'''
        # flatten dim mention of support size and batch size.
        # shape (batch_size * support_size, sent_len, emb_dim)
        support_reps = support_reps.view(-1, support_len, emb_dim)

        # shape (batch_size * support_size, sent_len, num_tags)
        support_targets = support_targets.view(batch_size * support_size, support_len, num_tags).float()

        # shape (batch_size, support_size, num_tags, emd_dim)
        sum_reps = torch.bmm(torch.transpose(support_targets, -1, -2), support_reps)
        # sum up tag emb over support set, shape (batch_size, num_tags, emd_dim)
        sum_reps = torch.sum(sum_reps.view(batch_size, support_size, num_tags, emb_dim), dim=1)

        # get num of each tag in support set, shape: (batch_size, num_tags, 1)
        tag_count = torch.sum(support_targets.view(batch_size, -1, num_tags), dim=1).unsqueeze(-1)
        # divide by 0 occurs when the tags, such as "I-x", are not existing in support.
        tag_count = self.remove_0(tag_count)
        prototype_reps = torch.div(sum_reps, tag_count)  # shape (batch_size, num_tags, emd_dim)

        # add PAD label
        if label_reps is not None:
            label_reps = torch.cat((torch.zeros_like(label_reps).narrow(dim=-2, start=0, length=1), label_reps), dim=-2)
            prototype_reps = (1 - self.scaler) * prototype_reps + self.scaler * label_reps

        '''get final test data reps'''
        # average test representation over support set (reps for each support sent can be different)
        test_reps = torch.mean(test_reps, dim=1)  # shape (batch_size, sent_len, emb_dim)

        '''calculate dot product'''
        sim_score = self.sim_func(test_reps, prototype_reps)  # shape (batch_size, sent_len, num_tags)

        '''store visualization embedding'''
        if not self.training and self.emb_log:
            log_context = '\n'.join(
                ['test_' + str(self.idx) + '-' + str(idx) + '-' + str(idx2) + '\t' + '\t'.join(map(str, item2))
                 for idx, item in enumerate(test_reps.tolist()) for idx2, item2 in enumerate(item)]) + '\n'
            log_context += '\n'.join(
                ['proto_' + str(self.idx) + '-' + str(idx) + '-' + str(idx2) + '\t' + '\t'.join(map(str, item2))
                 for idx, item in enumerate(prototype_reps.tolist()) for idx2, item2 in enumerate(item)]) + '\n'
            self.idx += batch_size
            self.emb_log.write(log_context)

        return sim_score

    def remove_0(self, my_tensor):
        """

        """
        return my_tensor + 0.0001


class TapNetSimilarityScorer(SimilarityScorerBase):
    def __init__(self, sim_func, num_anchors, mlp_out_dim, random_init=False, random_init_r=1.0, mlp=False, emb_log=None,
                 tap_proto=False, tap_proto_r=1.0, anchor_dim=768):
        super(TapNetSimilarityScorer, self).__init__(sim_func=sim_func, emb_log=emb_log)
        self.num_anchors = num_anchors
        self.random_init = random_init

        self.bert_emb_dim = anchor_dim
        if self.random_init:  # create anchors
            self.anchor_reps = torch.nn.Parameter(
                torch.nn.init.xavier_normal_(torch.randn((self.num_anchors, self.bert_emb_dim))), requires_grad=True)
        self.mlp = mlp
        self.mlp_out_dim = mlp_out_dim
        if self.mlp:
            self.f_theta = torch.nn.Linear(self.bert_emb_dim, self.mlp_out_dim)  # project for support set and test set
            self.phi = torch.nn.Linear(self.bert_emb_dim, self.mlp_out_dim)  # project for label reps
            # init with xavier normal distribution
            torch.nn.init.xavier_normal_(self.f_theta.weight)
            torch.nn.init.xavier_normal_(self.phi.weight)
        self.tap_proto = tap_proto
        self.tap_proto_r = tap_proto_r
        self.random_init_r = random_init_r
        self.idx = 0

    def forward(
            self,
            test_reps: torch.Tensor,
            support_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_output_mask: torch.Tensor,
            support_targets: torch.Tensor = None,
            label_reps: torch.Tensor = None, ) -> torch.Tensor:
        """
            calculate similarity between token and each label's prototype.
            :param test_reps: (batch_size, support_size, test_seq_len, dim)
            :param support_reps: (batch_size, support_size, support_seq_len)
            :param test_output_mask: (batch_size, test_seq_len)
            :param support_output_mask: (batch_size, support_size, support_seq_len)
            :param support_targets: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
            :param label_reps: (batch_size, no_pad_num_tags, dim)
            :return: similarity: (batch_size, test_seq_len, num_tags)
        """
        '''get data attribute'''
        support_size = support_reps.shape[1]
        test_len = test_reps.shape[-2]  # non-word-piece max test token num, Notice that it's different to input t len
        support_len = support_reps.shape[-2]  # non-word-piece max test token num
        emb_dim = test_reps.shape[-1]
        batch_size = test_reps.shape[0]
        num_tags = support_targets.shape[-1]
        no_pad_num_tags = num_tags - 1

        if no_pad_num_tags > len(self.anchor_reps) and (not label_reps or self.random_init):
            raise RuntimeError("Too few anchors")

        if label_reps is None and not self.random_init:
            raise RuntimeError('Must provide at least one of: anchor and label_reps.')

        ''' get reps for each tag with anchors or label reps '''
        if self.random_init:
            random_label_idxs = torch.randperm(len(self.anchor_reps))
            random_label_reps = self.anchor_reps[random_label_idxs[:no_pad_num_tags], :]
            random_label_reps = random_label_reps.unsqueeze(0).repeat(batch_size, 1, 1).to(support_reps.device)
            if label_reps is not None:  # use schema and integrate achor and schema reps as label reps
                label_reps = (1 - self.random_init_r) * label_reps + self.random_init_r * random_label_reps
            else:  # use anchor only as label reps
                label_reps = random_label_reps

        '''project label reps embedding and support data embedding with a MLP'''
        if self.mlp:
            label_reps = torch.tanh(self.phi(label_reps.contiguous().view(-1, emb_dim)))
            label_reps = label_reps.contiguous().view(batch_size, no_pad_num_tags, self.mlp_out_dim)
            support_reps = torch.tanh(self.f_theta(support_reps.contiguous().view(-1, emb_dim)))

        '''get prototype reps'''
        # flatten dim mention of support size and batch size. shape (batch_size * support_size, sent_len, emb_dim)
        support_reps = support_reps.view(-1, support_len, emb_dim)

        # shape (batch_size * support_size, sent_len, num_tags)
        support_targets = support_targets.view(batch_size * support_size, support_len, num_tags).float()

        # shape (batch_size * support_size, num_tags, emd_dim)
        sum_reps = torch.bmm(torch.transpose(support_targets, -1, -2), support_reps)
        # sum up tag emb over support set, shape (batch_size, num_tags, emd_dim)
        sum_reps = torch.sum(sum_reps.view(batch_size, support_size, num_tags, emb_dim), dim=1)

        # get num of each tag in support set, shape: (batch_size, num_tags, 1)
        tag_count = torch.sum(support_targets.view(batch_size, -1, num_tags), dim=1).unsqueeze(-1)
        # divide by 0 occurs when the tags, such as "I-x", are not existing in support.
        tag_count = self.remove_0(tag_count)
        prototype_reps = torch.div(sum_reps, tag_count)  # shape (batch_size, num_tags, emd_dim)

        '''generate error for every class'''
        # get normalized label reps
        label_reps_sum = label_reps.sum(dim=1).unsqueeze(1).repeat(1, no_pad_num_tags, 1) - label_reps
        label_reps_sum = label_reps - 1 / (no_pad_num_tags - 1) * label_reps_sum
        label_reps_sum = label_reps_sum / (torch.norm(label_reps_sum, p=2, dim=-1).unsqueeze(-1).expand_as(label_reps_sum) + 1e-13)
        # add [PAD] label reps
        label_reps_sum_pad = torch.cat(
            (torch.zeros_like(label_reps_sum).narrow(dim=-2, start=0, length=1).to(label_reps_sum.device), label_reps_sum), dim=-2)
        # get normalized proto reps
        prototype_reps_sum = \
            prototype_reps / (torch.norm(prototype_reps, p=2, dim=-1).unsqueeze(-1).expand_as(prototype_reps) + 1e-13)
        # get the error distance for optimization
        error_every_class = label_reps_sum_pad - prototype_reps_sum

        '''generate projection space M'''
        try:
            # torch 1.2.0 has the batch process function
            _, s, vh = torch.svd(error_every_class, some=False)
        except RuntimeError:
            # others does not
            batch_size = error_every_class.shape[0]
            s, vh = [], []
            for i in range(batch_size):
                _, s_, vh_ = torch.svd(error_every_class[i], some=False)
                s.append(s_)
                vh.append(vh_)
            s, vh = torch.stack(s, dim=0), torch.stack(vh, dim=0)
        s_sum = max((s >= 1e-13).sum(dim=1))
        # shape (batch_size, emb_dim, D)
        M = torch.stack([vh[i][:, s_sum:].clone() for i in range(batch_size)], dim=0)

        '''get final test data reps'''
        # project query data embedding with a MLP
        if self.mlp:
            test_reps = torch.tanh(self.f_theta(test_reps.contiguous().view(-1, emb_dim)))
            test_reps = test_reps.contiguous().view(batch_size, support_size, -1, self.mlp_out_dim)
        # average test representation over support set (reps for each support sent can be different)
        test_reps = torch.mean(test_reps, dim=1)  # shape (batch_size, sent_len, emb_dim)
        # add [PAD] label reps
        label_reps_pad = torch.cat(
            (torch.zeros_like(label_reps).narrow(dim=-2, start=0, length=1).to(label_reps.device),
             label_reps), dim=-2)

        '''calculate dot product'''
        if self.tap_proto:
            # shape (batch_size, sent_len, num_tags)
            label_proto_reps = self.tap_proto_r * prototype_reps + (1 - self.tap_proto_r) * label_reps_pad
            sim_score = 2 * self.sim_func(torch.matmul(test_reps, M), torch.matmul(label_proto_reps, M)) \
                + torch.log(
                    torch.sum(
                        torch.exp(- self.sim_func(torch.matmul(test_reps, M), torch.matmul(label_proto_reps, M))),
                        dim=-1)).unsqueeze(-1).repeat(1, 1, num_tags)

            if not self.training and self.emb_log:
                log_context = '\n'.join(['test_' + str(self.idx) + '-' + str(idx) + '-' + str(idx2) + '\t' + '\t'.join(map(str, item2))
                                         for idx, item in enumerate(test_reps.tolist()) for idx2, item2 in enumerate(item)]) + '\n'
                log_context += '\n'.join(['proto_' + str(self.idx) + '-' + str(idx) + '-' + str(idx2) + '\t' + '\t'.join(map(str, item2))
                                          for idx, item in enumerate(prototype_reps.tolist()) for idx2, item2 in enumerate(item)]) + '\n'
                log_context += '\n'.join(['p_test_' + str(self.idx) + '-' + str(idx) + '-' + str(idx2) + '\t' + '\t'.join(map(str, item2))
                                          for idx, item in enumerate(torch.matmul(test_reps, M).tolist()) for idx2, item2 in enumerate(item)]) + '\n'
                log_context += '\n'.join(['p_proto_' + str(self.idx) + '-' + str(idx) + '-' + str(idx2) + '\t' + '\t'.join(map(str, item2))
                                          for idx, item in enumerate(torch.matmul(prototype_reps, M).tolist()) for idx2, item2 in enumerate(item)]) + '\n'
                self.idx += batch_size
                self.emb_log.write(log_context)
        else:
            # shape (batch_size, sent_len, num_tags)
            sim_score = 2 * self.sim_func(torch.matmul(test_reps, M), torch.matmul(label_reps_pad, M)) \
                + torch.log(
                        torch.sum(
                            torch.exp(- self.sim_func(torch.matmul(test_reps, M), torch.matmul(label_reps_pad, M))),
                            dim=-1)).unsqueeze(-1).repeat(1, 1, num_tags)

            if not self.training and self.emb_log:
                log_context = '\n'.join(
                    ['test_' + str(self.idx) + '-' + str(idx) + '-' + str(idx2) + '\t' + '\t'.join(map(str, item2))
                     for idx, item in enumerate(test_reps.tolist()) for idx2, item2 in enumerate(item)]) + '\n'
                log_context += '\n'.join(
                    ['label_' + str(self.idx) + '-' + str(idx) + '-' + str(idx2) + '\t' + '\t'.join(map(str, item2))
                     for idx, item in enumerate(prototype_reps.tolist()) for idx2, item2 in enumerate(item)]) + '\n'
                log_context += '\n'.join(
                    ['p_test_' + str(self.idx) + '-' + str(idx) + '-' + str(idx2) + '\t' + '\t'.join(map(str, item2))
                     for idx, item in enumerate(torch.matmul(test_reps, M).tolist()) for idx2, item2 in
                     enumerate(item)]) + '\n'
                log_context += '\n'.join(
                    ['p_label_' + str(self.idx) + '-' + str(idx) + '-' + str(idx2) + '\t' + '\t'.join(map(str, item2))
                     for idx, item in enumerate(torch.matmul(prototype_reps, M).tolist()) for idx2, item2 in
                     enumerate(item)]) + '\n'
                self.idx += batch_size
                self.emb_log.write(log_context)

        return sim_score

    def remove_0(self, my_tensor):
        """

        """
        return my_tensor + 0.0001
