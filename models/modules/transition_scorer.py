#!/usr/bin/env python
from typing import Tuple, Callable
import torch, math
from torch import nn
from models.modules.scale_controller import ScaleControllerBase


class TransitionScorerBase(torch.nn.Module):
    def __init__(self, num_tags: int, normalizer: ScaleControllerBase = None, scaler: ScaleControllerBase = None):
        """
        :param num_tags:
        :param normalizer: normalize the transition, such as p1 normalization or softmax
        :param scaler: function to keep transition non-negative: relu, exp ...

        """
        super(TransitionScorerBase, self).__init__()
        self.num_tags = num_tags
        self.normalizer = normalizer
        self.scaler = scaler

    def forward(self, test_reps: torch.Tensor, support_target: torch.Tensor, label_reps: torch.Tensor = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


class FewShotTransitionScorer(TransitionScorerBase):
    """
        This transition scorer learns a backoff transition as following shows:
          _____________________
            O	B	I	B'	I'
        O
        B
        I _____________________
        It unfolds the backoff transition to get a accurate source transition matrix,
        and combine source & target transition matrix to get the result transition for decoding.
        Btw, start or end transition is back-offed as [O, B, I]
    """
    def __init__(self, num_tags: int, normalizer: ScaleControllerBase = None, scaler: Callable = None,
                 r: float = 1, backoff_init='rand'):
        """
        :param num_tags:
        :param normalizer: normalize the backoff transition before unfold, such as p1 normalization or softmax
        :param r: trade-off between back-off transition and target transition
        :param scaler: function to keep transition non-negative, such as relu, exp.
        :param backoff_init: method to initialize the backoff transition: rand, big_pos, fix
        """
        super(FewShotTransitionScorer, self).__init__(num_tags, normalizer, scaler)
        self.r = r  # Interpolation rate between source transition and target transition
        self.num_tags = num_tags  # this num include [PAD] now
        self.no_pad_num_tags = self.num_tags - 1
        self.backoff_init = backoff_init

        ''' build transition matrices  '''
        self.backoff_trans_mat, self.backoff_start_trans_mat, self.backoff_end_trans_mat = None, None, None
        self.target_start_trans_mat, self.target_end_trans_mat, self.target_trans_mat = None, None, None
        if self.r > 0:  # source transition is used
            self.init_backoff_trans()
            # index used to map backoff transition to accurate transition
            self.unfold_index = self.build_unfold_index()  # non-parameter version
            self.start_end_unfold_index = self.build_start_end_unfold_index()  # non-parameter version
        if self.r < 1:  # target transition is used
            self.target_trans_mat = torch.randn(num_tags, num_tags, dtype=torch.float)
            self.target_start_trans_mat = torch.randn(num_tags, dtype=torch.float)
            self.target_end_trans_mat = torch.randn(num_tags, dtype=torch.float)

    def forward(self, test_reps: torch.Tensor, support_target: torch.Tensor, label_reps: torch.Tensor = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the transition given sentence.
        :param test_reps:  (useless now), left for let transition depend on query sentence in the future
        :param support_target:
        :param label_reps:
        :return:
        """
        # calculate source transition
        if self.r > 0:
            source_trans, source_start_trans, source_end_trans = self.unfold_backoff_trans()
        else:
            source_trans, source_start_trans, source_end_trans = None, None, None

        # restrict transition to [0, + inf), default not.
        if self.scaler:
            source_trans = self.scaler(source_trans)
            source_start_trans = self.scaler(source_start_trans)
            source_end_trans = self.scaler(source_end_trans)

        # calculate target transition
        if self.r < 1:
            target_trans, target_start_trans, target_end_trans = self.get_target_trans(support_target)
        else:
            target_trans, target_start_trans, target_end_trans = None, None, None

        # combine two transition
        if self.r == 1 or self.training:  # only source transition, target trans is only used in testing
            return source_trans, source_start_trans, source_end_trans
        else:  # return transition after interpolation
            trans = self.r * source_trans + (1 - self.r) * target_trans
            start_trans = self.r * source_start_trans + (1 - self.r) * target_start_trans
            end_trans = self.r * source_end_trans + (1 - self.r) * target_end_trans
            return trans, start_trans, end_trans

    def init_backoff_trans(self):
        if self.backoff_init == 'rand':
            self.backoff_trans_mat = nn.Parameter(nn.init.xavier_normal_(torch.randn(3, 5, dtype=torch.float)),
                                                  requires_grad=True)
            self.backoff_start_trans_mat = nn.Parameter(0.5 * torch.randn(3, dtype=torch.float), requires_grad=True)
            self.backoff_end_trans_mat = nn.Parameter(0.5 * torch.randn(3, dtype=torch.float), requires_grad=True)
        elif self.backoff_init == 'fix':  # initial it with a big positive number
            self.backoff_trans_mat = nn.Parameter(torch.tensor(
                [[0.5, 0.5, -0.5, 0.5, -0.5],
                 [0.4, 0.2, 0.5, 0.2, -0.5],
                 [0.5, 0.2, 0.5, 0.2, -0.5]]), requires_grad=True)
            self.backoff_start_trans_mat = nn.Parameter(torch.tensor([0.5, 0.2, -0.5]), requires_grad=True)
            self.backoff_end_trans_mat = nn.Parameter(torch.tensor([0.5, 0.2, 0.5]), requires_grad=True)
        else:
            raise ValueError("in-valid choice for transition initialization")

    def unfold_backoff_trans(self) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """ Label2id must follow rules:  1. id(PAD)=0 id(O)=1  2. id(B-X)=i  id(I-X)=i+1  """
        ''' build trans without [PAD] '''
        self.unfold_index = self.unfold_index.to(self.backoff_trans_mat.device)
        self.start_end_unfold_index = self.start_end_unfold_index.to(self.backoff_trans_mat.device)
        self.unfold_index = self.unfold_index.to(self.backoff_trans_mat.device)

        # Normalize the transition before unfold perform better
        if self.normalizer:
            source_trans = torch.take(self.normalizer(self.backoff_trans_mat, dim=1).view(-1), self.unfold_index)
            source_start_trans = torch.take(self.normalizer(self.backoff_start_trans_mat, dim=0).view(-1), self.start_end_unfold_index)
            source_end_trans = torch.take(self.normalizer(self.backoff_end_trans_mat, dim=0).view(-1), self.start_end_unfold_index)
        else:
            source_trans = torch.take(self.backoff_trans_mat.view(-1), self.unfold_index)
            source_start_trans = torch.take(self.backoff_start_trans_mat.view(-1), self.start_end_unfold_index)
            source_end_trans = torch.take(self.backoff_end_trans_mat.view(-1), self.start_end_unfold_index)
        return source_trans, source_start_trans, source_end_trans

    def build_unfold_index(self):
        """ This is designed to not consider  """
        # special tag num, exclude [O] tag and see B-x & I-x as one tag.
        # -1 to block [PAD] label
        sp_tag_num = int(self.no_pad_num_tags / 2)
        unfold_index = torch.zeros(self.no_pad_num_tags, self.no_pad_num_tags, dtype=torch.long)
        index_viewer = torch.tensor(range(15)).view(3, 5)  # 15 = backoff element num, unfold need a flatten index
        # O to O
        unfold_index[0][0] = index_viewer[0, 0]
        for j in range(1, sp_tag_num + 1):
            B_idx = 2 * j - 1
            I_idx = 2 * j
            # O to B
            unfold_index[0][B_idx] = index_viewer[0, 1]
            # O to I
            unfold_index[0][I_idx] = index_viewer[0, 2]
        for i in range(1, sp_tag_num + 1):
            B_idx = 2 * i - 1
            I_idx = 2 * i
            for j in range(1, sp_tag_num + 1):
                B_idx_other = 2 * j - 1
                I_idx_other = 2 * j
                # B to B'
                unfold_index[B_idx][B_idx_other] = index_viewer[1, 3]
                # B to I'
                unfold_index[B_idx][I_idx_other] = index_viewer[1, 4]
                # I to B'
                unfold_index[I_idx][B_idx_other] = index_viewer[2, 3]
                # I to I'
                unfold_index[I_idx][I_idx_other] = index_viewer[2, 4]
            # B to O
            unfold_index[B_idx][0] = index_viewer[1, 0]
            # I to O
            unfold_index[I_idx][0] = index_viewer[2, 0]
            # B to B
            unfold_index[B_idx][B_idx] = index_viewer[1, 1]
            # B to I
            unfold_index[B_idx][I_idx] = index_viewer[1, 2]
            # I to B
            unfold_index[I_idx][B_idx] = index_viewer[2, 1]
            # I to I
            unfold_index[I_idx][I_idx] = index_viewer[2, 2]
        return unfold_index

    def build_start_end_unfold_index(self):
        # special tag num, exclude [O] tag and see B-x & I-x as one tag.
        # -1 to block [PAD] label
        sp_tag_num = int(self.no_pad_num_tags / 2)
        unfold_index = torch.zeros(self.no_pad_num_tags, dtype=torch.long)
        # O with start and end
        unfold_index[0] = 0
        for i in range(1, sp_tag_num + 1):
            B_idx = 2 * i - 1
            I_idx = 2 * i
            # B with start and end
            unfold_index[B_idx] = 1
            # I with start and end
            unfold_index[I_idx] = 2
        return unfold_index

    def do_norm(self, tensor_input: torch.Tensor, dim: int):
        return nn.functional.normalize(tensor_input, p=1, dim=dim)

    def get_target_trans(self, support_targets: torch.Tensor):
        """ count target domain transition """
        ''' Waring: count_target_trans must be used in testing, 
        and all example in test batch must share same support set'''
        self.target_trans_mat, self.target_start_trans_mat, self.target_end_trans_mat = \
            self.transition_from_one_support_set(targets=support_targets[0])
        self.target_trans_mat = self.do_norm(self.target_trans_mat, dim=1)
        self.target_start_trans_mat = self.do_norm(self.target_start_trans_mat, dim=0)
        self.target_end_trans_mat = self.do_norm(self.target_end_trans_mat, dim=0)
        return self.target_trans_mat, self.target_start_trans_mat, self.target_end_trans_mat

    def transition_from_one_support_set(self, targets):
        """
        count transition from support set
        :param targets: one support set's targets, one-hot targets (support_size, support_len, num_tags).
                        notice that it is including target.
        :return:
        """
        trans = torch.zeros(self.num_tags, self.num_tags, dtype=torch.float).to(self.backoff_trans_mat.device)
        start_trans = torch.zeros(self.num_tags, dtype=torch.float).to(self.backoff_trans_mat.device)
        end_trans = torch.zeros(self.num_tags, dtype=torch.float).to(self.backoff_trans_mat.device)

        for target in targets:
            # target shape: (support len, num_tags)
            # count trans case
            target = torch.argmax(target, dim=-1)  # convert one-hot to int, shape: (support_len,)
            end_idx = -1
            for i in range(len(target) - 1):
                current_id, next_id = target[i], target[i + 1]
                if next_id == 0:
                    end_idx = i
                    break
                trans[current_id][next_id] += 1

            # count start and end trans
            start_trans[target[0]] += 1
            end_trans[target[end_idx]] += 1
        return trans[1:, 1:], start_trans[1:], end_trans[1:]  # block [PAD] label

    def pad_transition(self, source_trans, source_start_trans, source_end_trans):
        """
        add [PAD] label transition, i.e. [0, 0,..., 0] to the first column and row
        This is useful when emission and transition scorer predict on [PAD] labels.
        """
        pad2label = torch.zeros(1, source_trans.shape[1], dtype=source_trans.dtype).to(source_trans.device)
        source_trans = torch.cat([pad2label, source_trans], dim=0)

        label2pad = torch.zeros(source_trans.shape[0], 1, dtype=source_trans.dtype).to(source_trans.device)
        source_trans = torch.cat([label2pad, source_trans], dim=1)

        start_end2pad = torch.zeros(1, dtype=source_trans.dtype).to(source_trans.device)
        source_start_trans = torch.cat([start_end2pad, source_start_trans], dim=0)
        source_end_trans = torch.cat([start_end2pad, source_end_trans], dim=0)
        return source_trans, source_start_trans, source_end_trans


class LabelRepsTranserBase(torch.nn.Module):
    def __init__(self, num_tags: int, embed_dim: int = 0):
        """
        :param num_tags:
        """
        super(LabelRepsTranserBase, self).__init__()
        self.num_tags = num_tags
        self.embed_dim = embed_dim

    def forward(self, label_reps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


class LabelRepsBiaffineTranser(torch.nn.Module):
    '''
    Biaffine transition scorer
    '''
    def __init__(self, num_tags: int, emb_dim: int = 768):
        """
        :param num_tags: number of tags WITHOUT [PAD]
        """
        super(LabelRepsBiaffineTranser, self).__init__()
        self.num_tags = num_tags
        self.emb_dim = emb_dim

        self.use_mlp = True
        self.dk = 768
        self.mlp_linear = torch.nn.Linear(self.emb_dim, self.dk)
        self.mlp_active_fun = torch.nn.Tanh()

        self.biaffine_w_1 = nn.Parameter(torch.randn(self.dk, self.dk, dtype=torch.float), requires_grad=True)
        self.biaffine_w_2 = nn.Parameter(torch.randn(1, 2*self.dk, dtype=torch.float), requires_grad=True)
        self.biaffine_b = nn.Parameter(torch.randn(1, self.num_tags), requires_grad=True)
        self.active_fun = None

        self.start_reps, self.end_reps = nn.Parameter(torch.randn(1, emb_dim, dtype=torch.float), requires_grad=True), \
                                         nn.Parameter(torch.randn(1, emb_dim, dtype=torch.float), requires_grad=True)
        self.label_trans_mat, self.label_start_trans_mat, self.label_end_trans_mat = None, None, None

    def forward(self, label_reps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param label_reps: torch.Tensor(num_tags, embed_dim)
        :return:
        """
        label_reps = self.mlp(label_reps) if self.use_mlp else label_reps

        # (num_tags, num_tags)
        self.label_trans_mat = self.biaffine_scorer(label_reps, label_reps)
        # (num_tags)
        self.label_start_trans_mat = self.biaffine_scorer(self.start_reps, label_reps)[0, :].view(self.num_tags)
        # (num_tags)
        self.label_end_trans_mat = self.biaffine_scorer(label_reps, self.end_reps)[:, 0].view(self.num_tags)

        if self.active_fun is not None:
            return self.active_fun(self.label_trans_mat), \
                   self.active_fun(self.label_start_trans_mat), \
                   self.active_fun(self.label_end_trans_mat)
        else:
            return self.label_trans_mat, \
                   self.label_start_trans_mat, \
                   self.label_end_trans_mat

    def mlp(self, input):
        """
        :return:
        """
        input = self.mlp_linear(input)
        if self.mlp_active_fun is not None:
            return self.mlp_active_fun(input)
        else:
            return input


    def biaffine_scorer(self, l1_reps, l2_reps):
        """
        calulate transition score from l1 label to l2
        define: L1:label_head_reps L2:label_dep_reps
        l1 = MLP(L1) l2 = MLP(L2) score(1-2)=l1 dot U1 dot l2^T + (l1 cat l2) dot U2 + bias
        U1 = R^{hidden*hidden} U2 = R^{2hidden*1}
        :param l1_reps: (embed_dim) or (num_tags, embed_dim)
        :param l2_reps: (embed_dim) or (num_tags, embed_dim)
        :return: transition matrix: (num_tags, num_tags)
        """
        # !Use MLP
        l1, l2 = l1_reps.expand(self.num_tags, self.dk), l2_reps.expand(self.num_tags, self.dk)
        s1 = torch.matmul(
            torch.matmul(l1, self.biaffine_w_1), l2.transpose(0, 1)
        ) / self.dk

        s2 = torch.matmul(
            torch.cat((l1, l2), dim = -1), self.biaffine_w_2.transpose(0, 1)
        ).view(1, self.num_tags) / (2*math.sqrt(self.dk))

        s3 = self.biaffine_b

        return (s1 + s2 + s3) / 3


class LabelRepsCatTranser(torch.nn.Module):
    def __init__(self, num_tags: int, emb_dim: int = 768, use_mlp: bool = True):
        """
        :param num_tags: number of tags WITHOUT [PAD]
        """
        super(LabelRepsCatTranser, self).__init__()
        self.num_tags = num_tags
        self.emb_dim = emb_dim
        self.label_cat_linear = nn.Linear(emb_dim*2, 1)
        self.active_fun = None

        self.start_reps, self.end_reps = nn.Parameter(torch.randn(emb_dim, dtype=torch.float), requires_grad=True), \
                                         nn.Parameter(torch.randn(emb_dim, dtype=torch.float), requires_grad=True)
        self.label_trans_mat, self.label_start_trans_mat, self.label_end_trans_mat = None, None, None

        self.use_mlp = use_mlp
        if self.use_mlp:
            self.mlp_linear = torch.nn.Linear(self.emb_dim, self.emb_dim)
            self.mlp_active_fun = torch.nn.Tanh()

    def forward(self, label_reps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param label_reps: torch.Tensor(num_tags, embed_dim)
        :return:
        """

        label_reps = self.mlp(label_reps) if self.use_mlp else label_reps

        # dk = label_reps.shape[1]
        dk = self.emb_dim
        sqrt_dk = math.sqrt(dk)

        self.label_trans_mat = torch.cat((
            torch.unsqueeze(label_reps, dim=1).expand(-1, self.num_tags, -1),
            torch.unsqueeze(label_reps, dim=0).expand(self.num_tags, -1, -1)
        ), dim=-1)  # (num_tags, num_tags, 2*embed_dim)
        self.label_trans_mat = torch.squeeze(self.label_cat_linear(self.label_trans_mat), dim=-1) / (2*sqrt_dk)    # (num_tags, num_tags)

        self.label_start_trans_mat = torch.cat((
            torch.unsqueeze(self.start_reps, dim=0).expand(self.num_tags, -1),
            label_reps
        ), dim=-1)
        self.label_start_trans_mat = torch.squeeze(self.label_cat_linear(self.label_start_trans_mat), dim=-1) / (2*sqrt_dk)  # (num_tags)

        self.label_end_trans_mat = torch.cat((
            label_reps,
            torch.unsqueeze(self.end_reps, dim=0).expand(self.num_tags, -1)
        ), dim=-1)
        self.label_end_trans_mat = torch.squeeze(self.label_cat_linear(self.label_end_trans_mat), dim=-1) / (2*sqrt_dk) # (num_tags)

        if self.active_fun is not None:
            return self.active_fun(self.label_trans_mat), \
                   self.active_fun(self.label_start_trans_mat), \
                   self.active_fun(self.label_end_trans_mat)
        else:
            return self.label_trans_mat, \
                   self.label_start_trans_mat, \
                   self.label_end_trans_mat

    def mlp(self, input: torch.Tensor):
        input = self.mlp_linear(input)
        if self.mlp_active_fun is not None:
            input = self.mlp_active_fun(input)
        return input


if False:
    a = LabelRepsTranser(4, 5)
    input = torch.randn(5, 5, dtype=torch.float)
    # print(input)
    # output, output1, output2 = a(input)
    output1, output2, output3 = a(input)
    print(output1.shape, output2.shape, output3.shape)
    exit(0)
    # print(output1.shape, output2.shape)


class FewShotTransitionScorerFromLabel(TransitionScorerBase):
    """
        This transition scorer learns a backoff transition as following shows:
          _____________________
            O	B	I	B'	I'
        O
        B
        I _____________________
        It unfolds the backoff transition to get a accurate source transition matrix,
        and combine source & target transition matrix to get the result transition for decoding.
        Btw, start or end transition is back-offed as [O, B, I]
    """
    def __init__(self, num_tags: int, normalizer: ScaleControllerBase = None, scaler: Callable = None,
                 r: float = 1, backoff_init='rand', label_scaler: Callable = None):
        """
        :param num_tags:
        :param normalizer: normalize the backoff transition before unfold, such as p1 normalization or softmax
        :param r: trade-off between back-off transition and target transition
        :param scaler: function to keep transition non-negative, such as relu, exp.
        :param backoff_init: method to initialize the backoff transition: rand, big_pos, fix
        :param label_scaler: function to keep transition FROM LABEL non-negative, such as relu, exp.
        """
        super(FewShotTransitionScorerFromLabel, self).__init__(num_tags, normalizer, scaler)
        self.r = r  # Interpolation rate between source transition and target transition
        self.num_tags = num_tags  # this num include [PAD] now
        self.no_pad_num_tags = self.num_tags - 1
        self.backoff_init = backoff_init

        # self.label_reps_transer = LabelRepsBiaffineTranser(self.no_pad_num_tags, 768)
        self.label_reps_transer = LabelRepsCatTranser(self.no_pad_num_tags, 768)
        self.g = label_scaler  # to scale transition from label reps

        ''' build transition matrices  '''
        self.backoff_trans_mat, self.backoff_start_trans_mat, self.backoff_end_trans_mat = None, None, None
        self.target_start_trans_mat, self.target_end_trans_mat, self.target_trans_mat = None, None, None
        if self.r > 0:  # source transition is used
            self.init_backoff_trans()
            # index used to map backoff transition to accurate transition
            self.unfold_index = self.build_unfold_index()  # non-parameter version
            self.start_end_unfold_index = self.build_start_end_unfold_index()  # non-parameter version
        if self.r < 1:  # target transition is used
            self.target_trans_mat = torch.randn(num_tags, num_tags, dtype=torch.float)
            self.target_start_trans_mat = torch.randn(num_tags, dtype=torch.float)
            self.target_end_trans_mat = torch.randn(num_tags, dtype=torch.float)

    def forward(self, test_reps: torch.Tensor, support_target: torch.Tensor, label_reps: torch.Tensor = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the transition given sentence.
        :param test_reps:  (useless now), left for let transition depend on query sentence in the future
        :param support_target:
        :return:
        """
        # calculate source transition
        if self.r > 0:
            source_trans, source_start_trans, source_end_trans = self.unfold_backoff_trans()
        else:
            source_trans, source_start_trans, source_end_trans = None, None, None

        # restrict transition to [0, + inf), default not.
        if self.scaler:
            source_trans = self.scaler(source_trans)
            source_start_trans = self.scaler(source_start_trans)
            source_end_trans = self.scaler(source_end_trans)

        # calculate target transition
        if self.r < 1:
            target_trans, target_start_trans, target_end_trans = self.get_target_trans(support_target)
        else:
            target_trans, target_start_trans, target_end_trans = None, None, None

        # calculate label reps transition
        label_trans, label_start_trans, label_end_trans = self.label_reps_transer(label_reps)

        # combine two transition
        if self.r == 1 or self.training:  # only source transition, target trans is only used in testing
            return source_trans + self.g(label_trans), \
                   source_start_trans + self.g(label_start_trans), \
                   source_end_trans + self.g(label_end_trans)
        else:  # return transition after interpolation
            trans = self.r * source_trans + (1 - self.r) * target_trans + self.g(label_trans)
            start_trans = self.r * source_start_trans + (1 - self.r) * target_start_trans + self.g(label_start_trans)
            end_trans = self.r * source_end_trans + (1 - self.r) * target_end_trans + self.g(label_end_trans)
            return trans, start_trans, end_trans

    def init_backoff_trans(self):
        if self.backoff_init == 'rand':
            self.backoff_trans_mat = nn.Parameter(nn.init.xavier_normal_(torch.randn(3, 5, dtype=torch.float)),
                                                  requires_grad=True)
            self.backoff_start_trans_mat = nn.Parameter(0.5 * torch.randn(3, dtype=torch.float), requires_grad=True)
            self.backoff_end_trans_mat = nn.Parameter(0.5 * torch.randn(3, dtype=torch.float), requires_grad=True)
        elif self.backoff_init == 'fix':  # initial it with a big positive number
            self.backoff_trans_mat = nn.Parameter(torch.tensor(
                [[0.5, 0.5, -0.5, 0.5, -0.5],
                 [0.4, 0.2, 0.5, 0.2, -0.5],
                 [0.5, 0.2, 0.5, 0.2, -0.5]]), requires_grad=True)
            self.backoff_start_trans_mat = nn.Parameter(torch.tensor([0.5, 0.2, -0.5]), requires_grad=True)
            self.backoff_end_trans_mat = nn.Parameter(torch.tensor([0.5, 0.2, 0.5]), requires_grad=True)
        else:
            raise ValueError("in-valid choice for transition initialization")

    def unfold_backoff_trans(self) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """ Label2id must follow rules:  1. id(PAD)=0 id(O)=1  2. id(B-X)=i  id(I-X)=i+1  """
        ''' build trans without [PAD] '''
        self.unfold_index = self.unfold_index.to(self.backoff_trans_mat.device)
        self.start_end_unfold_index = self.start_end_unfold_index.to(self.backoff_trans_mat.device)
        self.unfold_index = self.unfold_index.to(self.backoff_trans_mat.device)

        # Normalize the transition before unfold perform better
        if self.normalizer:
            source_trans = torch.take(self.normalizer(self.backoff_trans_mat, dim=1).view(-1), self.unfold_index)
            source_start_trans = torch.take(self.normalizer(self.backoff_start_trans_mat, dim=0).view(-1), self.start_end_unfold_index)
            source_end_trans = torch.take(self.normalizer(self.backoff_end_trans_mat, dim=0).view(-1), self.start_end_unfold_index)
        else:
            source_trans = torch.take(self.backoff_trans_mat.view(-1), self.unfold_index)
            source_start_trans = torch.take(self.backoff_start_trans_mat.view(-1), self.start_end_unfold_index)
            source_end_trans = torch.take(self.backoff_end_trans_mat.view(-1), self.start_end_unfold_index)
        return source_trans, source_start_trans, source_end_trans

    def build_unfold_index(self):
        """ This is designed to not consider  """
        # special tag num, exclude [O] tag and see B-x & I-x as one tag.
        # -1 to block [PAD] label
        sp_tag_num = int(self.no_pad_num_tags / 2)
        unfold_index = torch.zeros(self.no_pad_num_tags, self.no_pad_num_tags, dtype=torch.long)
        index_viewer = torch.tensor(range(15)).view(3, 5)  # 15 = backoff element num, unfold need a flatten index
        # O to O
        unfold_index[0][0] = index_viewer[0, 0]
        for j in range(1, sp_tag_num + 1):
            B_idx = 2 * j - 1
            I_idx = 2 * j
            # O to B
            unfold_index[0][B_idx] = index_viewer[0, 1]
            # O to I
            unfold_index[0][I_idx] = index_viewer[0, 2]
        for i in range(1, sp_tag_num + 1):
            B_idx = 2 * i - 1
            I_idx = 2 * i
            for j in range(1, sp_tag_num + 1):
                B_idx_other = 2 * j - 1
                I_idx_other = 2 * j
                # B to B'
                unfold_index[B_idx][B_idx_other] = index_viewer[1, 3]
                # B to I'
                unfold_index[B_idx][I_idx_other] = index_viewer[1, 4]
                # I to B'
                unfold_index[I_idx][B_idx_other] = index_viewer[2, 3]
                # I to I'
                unfold_index[I_idx][I_idx_other] = index_viewer[2, 4]
            # B to O
            unfold_index[B_idx][0] = index_viewer[1, 0]
            # I to O
            unfold_index[I_idx][0] = index_viewer[2, 0]
            # B to B
            unfold_index[B_idx][B_idx] = index_viewer[1, 1]
            # B to I
            unfold_index[B_idx][I_idx] = index_viewer[1, 2]
            # I to B
            unfold_index[I_idx][B_idx] = index_viewer[2, 1]
            # I to I
            unfold_index[I_idx][I_idx] = index_viewer[2, 2]
        return unfold_index

    def build_start_end_unfold_index(self):
        # special tag num, exclude [O] tag and see B-x & I-x as one tag.
        # -1 to block [PAD] label
        sp_tag_num = int(self.no_pad_num_tags / 2)
        unfold_index = torch.zeros(self.no_pad_num_tags, dtype=torch.long)
        # O with start and end
        unfold_index[0] = 0
        for i in range(1, sp_tag_num + 1):
            B_idx = 2 * i - 1
            I_idx = 2 * i
            # B with start and end
            unfold_index[B_idx] = 1
            # I with start and end
            unfold_index[I_idx] = 2
        return unfold_index

    def do_norm(self, tensor_input: torch.Tensor, dim: int):
        return nn.functional.normalize(tensor_input, p=1, dim=dim)

    def get_target_trans(self, support_targets: torch.Tensor):
        """ count target domain transition """
        ''' Waring: count_target_trans must be used in testing, 
        and all example in test batch must share same support set'''
        self.target_trans_mat, self.target_start_trans_mat, self.target_end_trans_mat = \
            self.transition_from_one_support_set(targets=support_targets[0])
        self.target_trans_mat = self.do_norm(self.target_trans_mat, dim=1)
        self.target_start_trans_mat = self.do_norm(self.target_start_trans_mat, dim=0)
        self.target_end_trans_mat = self.do_norm(self.target_end_trans_mat, dim=0)
        return self.target_trans_mat, self.target_start_trans_mat, self.target_end_trans_mat

    def transition_from_one_support_set(self, targets):
        """
        count transition from support set
        :param targets: one support set's targets, one-hot targets (support_size, support_len, num_tags).
                        notice that it is including target.
        :return:
        """
        trans = torch.zeros(self.num_tags, self.num_tags, dtype=torch.float).to(self.backoff_trans_mat.device)
        start_trans = torch.zeros(self.num_tags, dtype=torch.float).to(self.backoff_trans_mat.device)
        end_trans = torch.zeros(self.num_tags, dtype=torch.float).to(self.backoff_trans_mat.device)

        for target in targets:
            # target shape: (support len, num_tags)
            # count trans case
            target = torch.argmax(target, dim=-1)  # convert one-hot to int, shape: (support_len,)
            end_idx = -1
            for i in range(len(target) - 1):
                current_id, next_id = target[i], target[i + 1]
                if next_id == 0:
                    end_idx = i
                    break
                trans[current_id][next_id] += 1

            # count start and end trans
            start_trans[target[0]] += 1
            end_trans[target[end_idx]] += 1
        return trans[1:, 1:], start_trans[1:], end_trans[1:]  # block [PAD] label

    def pad_transition(self, source_trans, source_start_trans, source_end_trans):
        """
        add [PAD] label transition, i.e. [0, 0,..., 0] to the first column and row
        This is useful when emission and transition scorer predict on [PAD] labels.
        """
        pad2label = torch.zeros(1, source_trans.shape[1], dtype=source_trans.dtype).to(source_trans.device)
        source_trans = torch.cat([pad2label, source_trans], dim=0)

        label2pad = torch.zeros(source_trans.shape[0], 1, dtype=source_trans.dtype).to(source_trans.device)
        source_trans = torch.cat([label2pad, source_trans], dim=1)

        start_end2pad = torch.zeros(1, dtype=source_trans.dtype).to(source_trans.device)
        source_start_trans = torch.cat([start_end2pad, source_start_trans], dim=0)
        source_end_trans = torch.cat([start_end2pad, source_end_trans], dim=0)
        return source_trans, source_start_trans, source_end_trans
