#!/usr/bin/env python
import torch
from models.modules.similarity_scorer_base import SimilarityScorerBase, MatchingSimilarityScorer, \
    PrototypeSimilarityScorer, ProtoWithLabelSimilarityScorer
from models.modules.scale_controller import ScaleControllerBase


class EmissionScorerBase(torch.nn.Module):
    def __init__(self, similarity_scorer: SimilarityScorerBase, scaler: ScaleControllerBase = None):
        """
        :param similarity_scorer: Module for calculating token similarity
        """
        super(EmissionScorerBase, self).__init__()
        self.similarity_scorer = similarity_scorer
        self.scaler = scaler

    def forward(
            self,
            test_reps: torch.Tensor,
            support_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_output_mask: torch.Tensor,
            support_targets: torch.Tensor,
            label_reps: torch.Tensor = None, ) -> torch.Tensor:
        """
        :param test_reps: (batch_size, support_size, test_seq_len, dim), notice: reps has been expand to support size
        :param support_reps: (batch_size, support_size, support_seq_len)
        :param test_output_mask: (batch_size, test_seq_len)
        :param support_output_mask: (batch_size, support_size, support_seq_len)
        :param support_targets: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
        :param label_reps: (batch_size, num_tags, dim)
        :return: emission, shape: (batch_size, test_len, no_pad_num_tags)
        """
        raise NotImplementedError()


class MNetEmissionScorer(EmissionScorerBase):
    def __init__(self, similarity_scorer: MatchingSimilarityScorer, scaler: ScaleControllerBase = None,
                 div_by_tag_num: bool = True):
        """
        :param similarity_scorer: Module for calculating token similarity
        :param div_by_tag_num: if true, model will div each types emission by its slot type num
        :param scaler: callable function that normalize the emission
        """
        super(MNetEmissionScorer, self).__init__(similarity_scorer, scaler)
        self.div_by_tag_num = div_by_tag_num

    def forward(
            self,
            test_reps: torch.Tensor,
            support_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_output_mask: torch.Tensor,
            support_targets: torch.Tensor,
            label_reps: torch.Tensor = None, ) -> torch.Tensor:
        """
        :param test_reps: (batch_size, support_size, test_seq_len, dim), notice: reps has been expand to support size
        :param support_reps: (batch_size, support_size, support_seq_len)
        :param test_output_mask: (batch_size, test_seq_len)
        :param support_output_mask: (batch_size, support_size, support_seq_len)
        :param support_targets: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
        :param label_reps: (batch_size, num_tags, dim)
        :return: emission, shape: (batch_size, test_len, no_pad_num_tags)
        """
        similarity = self.similarity_scorer(test_reps, support_reps, test_output_mask, support_output_mask)
        emission = self.get_emission(similarity, support_targets)  # shape(batch_size, test_len, no_pad_num_tag)
        return emission

    def get_emission(self, similarities: torch.Tensor, support_targets: torch.Tensor):
        """
        :param similarities: (batch_size, support_size, test_seq_len, support_seq_len)
        :param support_targets: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
        :return: emission: shape: (batch_size, test_len, no_pad_num_tags)
        """
        batch_size, support_size, test_len, support_len = similarities.shape
        # get num of each tag in support set, shape: (batch_size, num_tags)
        batch_size, support_size, support_len, num_tags = support_targets.shape
        no_pad_num_tags = num_tags - 1  # block emission on pad
        ''' reshape the two factor, to flatten dim of batch_size and support_size '''
        similarities = similarities.view((-1, test_len, support_len))
        support_targets = support_targets.view((-1, support_len, num_tags)).float()
        ''' inner product, get label distribution from each support sentence. '''
        emission = torch.bmm(similarities, support_targets)  # shape: (batch_size * support_size, test_len, num_tags)
        # change to support set shape, shape: (batch_size * support_size, test_len, num_tags)
        emission = emission.view(batch_size, support_size, test_len, num_tags)
        ''' sum over support set '''
        emission = torch.sum(emission, dim=1)  # shape: (batch_size, test_len, num_tags)

        ''' process emission '''
        # cut emission to block predictions on [PAD] label (we use 0 as [PAD] label id)
        emission = emission.narrow(-1, 1, no_pad_num_tags)
        # div/average emission to make it stable in number range
        if self.div_by_tag_num:
            emission = self.div_emission_by_tag_num(emission, support_targets, batch_size, num_tags)
        # normalize the emission score
        if self.scaler:
            emission = self.scaler(emission, p=3, dim=-1)

        return emission

    def div_emission_by_tag_num(self, emission, support_target, batch_size, num_tags):
        """ div/average emission to make it stable in number range """
        no_pad_num_tags = num_tags - 1
        tag_count = torch.sum(support_target.view(batch_size, -1, num_tags), dim=1).float()
        # shape: (batch_size, no_pad_num_tags)
        tag_count = tag_count.narrow(-1, 1, no_pad_num_tags)
        # shape: (batch_size, 1, no_pad_num_tags)
        tag_count = tag_count.unsqueeze(-2)
        tag_count = self.remove_0(tag_count)
        logits = torch.div(emission, tag_count)
        return logits

    def div_emission_by_spt_num(self, emission, support_num):
        """
        :param emission:  shape(batch_size, test_len, no_pad_num_tag)
        :param support_target: one-hot targets (batch_size, support_size, support_len)
        :param support_num: (batch_size, 1)
        :return: shape(batch_size, test_len, no_pad_num_tag)
        """
        logits = torch.div(emission, support_num.float().unsqueeze(-1))
        return logits

    def remove_0(self, my_tensor):
        """
        remove nan is not working, so add
        """
        return my_tensor + 0.0001

    def remove_nan(self, my_tensor):
        """
        Using 'torch.where' here because:
        modifying tensors in-place can cause issues with backprop.
        Notice: this may cause no decay of loss.
        """
        return torch.where(torch.isnan(my_tensor), torch.zeros_like(my_tensor), my_tensor)


class PrototypeEmissionScorer(EmissionScorerBase):
    def __init__(self, similarity_scorer: PrototypeSimilarityScorer, scaler: ScaleControllerBase = None):
        """
        :param similarity_scorer: Module for calculating token similarity
        """
        super(PrototypeEmissionScorer, self).__init__(similarity_scorer, scaler)

    def forward(
            self,
            test_reps: torch.Tensor,
            support_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_output_mask: torch.Tensor,
            support_targets: torch.Tensor,
            label_reps: torch.Tensor = None, ) -> torch.Tensor:
        """
        :param test_reps: (batch_size, support_size, test_seq_len, dim), notice: reps has been expand to support size
        :param support_reps: (batch_size, support_size, support_seq_len)
        :param test_output_mask: (batch_size, test_seq_len)
        :param support_output_mask: (batch_size, support_size, support_seq_len)
        :param support_targets: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
        :param label_reps: (batch_size, num_tags, dim)
        :return: emission, shape: (batch_size, test_len, no_pad_num_tags)
        """
        similarity = self.similarity_scorer(
            test_reps, support_reps, test_output_mask, support_output_mask, support_targets)
        emission = self.get_emission(similarity, support_targets)  # shape(batch_size, test_len, no_pad_num_tag)
        return emission

    def get_emission(self, similarities: torch.Tensor, support_targets: torch.Tensor):
        """
        :param similarities: (batch_size, support_size, test_seq_len, support_seq_len)
        :param support_targets: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
        :return: emission: shape: (batch_size, test_len, no_pad_num_tags)
        """
        batch_size, test_len, num_tags = similarities.shape
        no_pad_num_tags = num_tags - 1  # block emission on pad
        ''' cut emission to block predictions on [PAD] label (we use 0 as [PAD] label id) '''
        emission = similarities.narrow(-1, 1, no_pad_num_tags)
        if self.scaler:
            emission = self.scaler(emission, p=3, dim=-1)
        return emission


class ProtoWithLabelEmissionScorer(EmissionScorerBase):
    def __init__(self, similarity_scorer: ProtoWithLabelSimilarityScorer, scaler: ScaleControllerBase = None):
        """
        :param similarity_scorer: Module for calculating token similarity
        """
        super(ProtoWithLabelEmissionScorer, self).__init__(similarity_scorer, scaler)

    def forward(
            self,
            test_reps: torch.Tensor,
            support_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_output_mask: torch.Tensor,
            support_targets: torch.Tensor,
            label_reps: torch.Tensor = None, ) -> torch.Tensor:
        """
        :param test_reps: (batch_size, support_size, test_seq_len, dim), notice: reps has been expand to support size
        :param support_reps: (batch_size, support_size, support_seq_len)
        :param test_output_mask: (batch_size, test_seq_len)
        :param support_output_mask: (batch_size, support_size, support_seq_len)
        :param support_targets: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
        :param label_reps: (batch_size, num_tags, dim)
        :return: emission, shape: (batch_size, test_len, no_pad_num_tags)
        """
        similarity = self.similarity_scorer(
            test_reps, support_reps, test_output_mask, support_output_mask, support_targets, label_reps)
        emission = self.get_emission(similarity, support_targets)  # shape(batch_size, test_len, no_pad_num_tag)
        return emission

    def get_emission(self, similarities: torch.Tensor, support_targets: torch.Tensor):
        """
        :param similarities: (batch_size, support_size, test_seq_len, support_seq_len)
        :param support_targets: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
        :return: emission: shape: (batch_size, test_len, no_pad_num_tags)
        """
        batch_size, test_len, num_tags = similarities.shape
        no_pad_num_tags = num_tags - 1  # block emission on pad
        ''' cut emission to block predictions on [PAD] label (we use 0 as [PAD] label id) '''
        emission = similarities.narrow(-1, 1, no_pad_num_tags)
        if self.scaler:
            emission = self.scaler(emission, p=3, dim=-1)
        return emission


class TapNetEmissionScorer(EmissionScorerBase):
    def __init__(self, similarity_scorer: ProtoWithLabelSimilarityScorer, scaler: ScaleControllerBase = None):
        """
        :param similarity_scorer: Module for calculating token similarity
        """
        super(TapNetEmissionScorer, self).__init__(similarity_scorer, scaler)

    def forward(
            self,
            test_reps: torch.Tensor,
            support_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_output_mask: torch.Tensor,
            support_targets: torch.Tensor,
            label_reps: torch.Tensor = None, ) -> torch.Tensor:
        """
        :param test_reps: (batch_size, support_size, test_seq_len, dim), notice: reps has been expand to support size
        :param support_reps: (batch_size, support_size, support_seq_len)
        :param test_output_mask: (batch_size, test_seq_len)
        :param support_output_mask: (batch_size, support_size, support_seq_len)
        :param support_targets: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
        :param label_reps: (batch_size, num_tags, dim)
        :return: emission, shape: (batch_size, test_len, no_pad_num_tags)
        """
        similarity = self.similarity_scorer(
            test_reps, support_reps, test_output_mask, support_output_mask, support_targets, label_reps)
        emission = self.get_emission(similarity, support_targets)  # shape(batch_size, test_len, no_pad_num_tag)
        return emission

    def get_emission(self, similarities: torch.Tensor, support_targets: torch.Tensor):
        """
        :param similarities: (batch_size, support_size, test_seq_len, support_seq_len)
        :param support_targets: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
        :return: emission: shape: (batch_size, test_len, no_pad_num_tags)
        """
        batch_size, test_len, num_tags = similarities.shape
        no_pad_num_tags = num_tags - 1  # block emission on pad
        ''' cut emission to block predictions on [PAD] label (we use 0 as [PAD] label id)'''
        emission = similarities.narrow(-1, 1, no_pad_num_tags)
        if self.scaler:
            emission = self.scaler(emission, p=3, dim=-1)
        return emission

