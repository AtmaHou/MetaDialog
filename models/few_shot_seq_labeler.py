#!/usr/bin/env python
import torch
from typing import Tuple, List
from models.modules.context_embedder_base import ContextEmbedderBase
from models.modules.emission_scorer_base import EmissionScorerBase
from models.modules.transition_scorer import TransitionScorerBase
from models.modules.seq_labeler import SequenceLabeler
from models.modules.conditional_random_field import ConditionalRandomField


class FewShotSeqLabeler(torch.nn.Module):
    def __init__(self,
                 opt,
                 context_embedder: ContextEmbedderBase,
                 emission_scorer: EmissionScorerBase,
                 decoder: torch.nn.Module,
                 transition_scorer: TransitionScorerBase = None,
                 config: dict = None,  # store necessary setting or none-torch params
                 emb_log: str = None):
        super(FewShotSeqLabeler, self).__init__()
        self.opt = opt
        self.context_embedder = context_embedder
        self.emission_scorer = emission_scorer
        self.transition_scorer = transition_scorer
        self.decoder = decoder
        self.config = config
        self.emb_log = emb_log

    def forward(
            self,
            test_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_reps: torch.Tensor,
            support_output_mask: torch.Tensor,
            test_target: torch.Tensor,
            support_target: torch.Tensor,
            label_mask: torch.Tensor = None,
    ):
        """
        :param test_reps: (batch_size, test_len, emb_dim)
        :param test_output_mask: (batch_size, test_len)
        :param support_reps: (batch_size, support_size, support_len, emb_dim)
        :param support_output_mask: (batch_size, support_size, support_len)
        :param test_target: index targets (batch_size, test_len)
        :param support_target: one-hot targets (batch_size, support_size, support_len, num_tags)
        :param label_mask: the output label mask
        :return:
        """
        # calculate emission: shape(batch_size, test_len, no_pad_num_tag)
        emission = self.emission_scorer(test_reps, support_reps, test_output_mask, support_output_mask, support_target)
        logits = emission

        # as we remove pad label (id = 0), so all label id sub 1. And relu is used to avoid -1 index
        test_target = torch.nn.functional.relu(test_target - 1)

        if self.transition_scorer:
            transitions, start_transitions, end_transitions = self.transition_scorer(test_reps, support_target)

            if label_mask is not None:
                transitions = self.mask_transition(transitions, label_mask)

            self.decoder: ConditionalRandomField
            # the CRF staff
            llh = self.decoder.forward(
                inputs=logits,
                transitions=transitions,
                start_transitions=start_transitions,
                end_transitions=end_transitions,
                tags=test_target,
                mask=test_output_mask)
            loss = -1 * llh

        else:
            self.decoder: SequenceLabeler
            loss = self.decoder.forward(logits=logits,
                                        tags=test_target,
                                        mask=test_output_mask)

        return loss

    def decode(
            self,
            test_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_reps: torch.Tensor,
            support_output_mask: torch.Tensor,
            test_target: torch.Tensor,
            support_target: torch.Tensor,
            label_mask: torch.Tensor = None,
    ):
        """
        :param test_reps: (batch_size, test_len, emb_dim)
        :param test_output_mask: (batch_size, test_len)
        :param support_reps: (batch_size, support_size, support_len, emb_dim)
        :param support_output_mask: (batch_size, support_size, support_len)
        :param test_target: index targets (batch_size, test_len)
        :param support_target: one-hot targets (batch_size, support_size, support_len, num_tags)
        :param label_mask: the output label mask
        :return:
        """
        # calculate emission: shape(batch_size, test_len, no_pad_num_tag)
        emission = self.emission_scorer(test_reps, support_reps, test_output_mask, support_output_mask, support_target)
        logits = emission

        if self.transition_scorer:
            transitions, start_transitions, end_transitions = self.transition_scorer(test_reps, support_target)

            if label_mask is not None:
                transitions = self.mask_transition(transitions, label_mask)

            self.decoder: ConditionalRandomField
            best_paths = self.decoder.viterbi_tags(logits=logits,
                                                   transitions_without_constrain=transitions,
                                                   start_transitions=start_transitions,
                                                   end_transitions=end_transitions,
                                                   mask=test_output_mask)
            # split path and score
            prediction, path_score = zip(*best_paths)
            # we block pad label(id=0) before by - 1, here, we add 1 back
            prediction = self.add_back_pad_label(prediction)
        else:
            self.decoder: SequenceLabeler
            prediction = self.decoder.decode(logits=logits, masks=test_output_mask)
            # we block pad label(id=0) before by - 1, here, we add 1 back
            prediction = self.add_back_pad_label(prediction)
        return prediction

    def add_back_pad_label(self, predictions: List[List[int]]):
        for pred in predictions:
            for ind, l_id in enumerate(pred):
                pred[ind] += 1  # pad token is in the first place
        return predictions

    def mask_transition(self, transitions, label_mask):
        trans_mask = label_mask[1:, 1:].float()  # block pad label(at 0) here
        transitions = transitions * trans_mask
        return transitions


class SchemaFewShotSeqLabeler(FewShotSeqLabeler):
    def __init__(
            self,
            opt,
            context_embedder: ContextEmbedderBase,
            emission_scorer: EmissionScorerBase,
            decoder: torch.nn.Module,
            transition_scorer: TransitionScorerBase = None,
            config: dict = None,  # store necessary setting or none-torch params
            emb_log: str = None
    ):
        super(SchemaFewShotSeqLabeler, self).__init__(
            opt, context_embedder, emission_scorer, decoder, transition_scorer, config, emb_log)

    def forward(
            self,
            test_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_reps: torch.Tensor,
            support_output_mask: torch.Tensor,
            test_target: torch.Tensor,
            support_target: torch.Tensor,
            label_reps: torch.Tensor = None,
            label_mask: torch.Tensor = None,
    ):
        """
        few-shot sequence labeler using schema information
        :param test_reps: (batch_size, test_len, emb_dim)
        :param test_output_mask: (batch_size, test_len)
        :param support_reps: (batch_size, support_size, support_len, emb_dim)
        :param support_output_mask: (batch_size, support_size, support_len)
        :param test_target: index targets (batch_size, test_len)
        :param support_target: one-hot targets (batch_size, support_size, support_len, num_tags)
        :param label_reps: (batch_size, label_num, emb_dim)
        :param label_mask: the output label mask
        :return:
        """

        # calculate emission: shape(batch_size, test_len, no_pad_num_tag)
        emission = self.emission_scorer(test_reps, support_reps, test_output_mask, support_output_mask, support_target,
                                        label_reps)
        logits = emission

        # block pad of label_id = 0, so all label id sub 1. And relu is used to avoid -1 index
        test_target = torch.nn.functional.relu(test_target - 1)

        loss, prediction = torch.FloatTensor([0]).to(test_target.device), None
        if self.transition_scorer:
            transitions, start_transitions, end_transitions = self.transition_scorer(test_reps, support_target, label_reps[0])

            if label_mask is not None:
                transitions = self.mask_transition(transitions, label_mask)

            self.decoder: ConditionalRandomField
            # the CRF staff
            llh = self.decoder.forward(
                inputs=logits,
                transitions=transitions,
                start_transitions=start_transitions,
                end_transitions=end_transitions,
                tags=test_target,
                mask=test_output_mask)
            loss = -1 * llh
        else:
            self.decoder: SequenceLabeler
            loss = self.decoder.forward(logits=logits,
                                        tags=test_target,
                                        mask=test_output_mask)
        return loss

    def decode(
            self,
            test_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_reps: torch.Tensor,
            support_output_mask: torch.Tensor,
            test_target: torch.Tensor,
            support_target: torch.Tensor,
            label_reps: torch.Tensor = None,
            label_mask: torch.Tensor = None,
    ):
        """
        few-shot sequence labeler using schema information
        :param test_reps: (batch_size, test_len, emb_dim)
        :param test_output_mask: (batch_size, test_len)
        :param support_reps: (batch_size, support_size, support_len, emb_dim)
        :param support_output_mask: (batch_size, support_size, support_len)
        :param test_target: index targets (batch_size, test_len)
        :param support_target: one-hot targets (batch_size, support_size, support_len, num_tags)
        :param label_reps: (batch_size, label_num, emb_dim)
        :param label_mask: the output label mask
        :return:
        """

        # calculate emission: shape(batch_size, test_len, no_pad_num_tag)
        emission = self.emission_scorer(test_reps, support_reps, test_output_mask, support_output_mask, support_target,
                                        label_reps)
        logits = emission

        if self.transition_scorer:
            transitions, start_transitions, end_transitions = self.transition_scorer(test_reps, support_target, label_reps[0])

            if label_mask is not None:
                transitions = self.mask_transition(transitions, label_mask)

            self.decoder: ConditionalRandomField

            best_paths = self.decoder.viterbi_tags(logits=logits,
                                                   transitions_without_constrain=transitions,
                                                   start_transitions=start_transitions,
                                                   end_transitions=end_transitions,
                                                   mask=test_output_mask)
            # split path and score
            prediction, path_score = zip(*best_paths)
            # we block pad label(id=0) before by - 1, here, we add 1 back
            prediction = self.add_back_pad_label(prediction)
        else:
            self.decoder: SequenceLabeler

            prediction = self.decoder.decode(logits=logits, masks=test_output_mask)
            # we block pad label(id=0) before by - 1, here, we add 1 back
            prediction = self.add_back_pad_label(prediction)
        return prediction


def main():
    pass


if __name__ == "__main__":
    main()
