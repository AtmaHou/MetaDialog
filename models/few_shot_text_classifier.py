#!/usr/bin/env python
import torch
from torch import nn
from typing import Tuple, List
from models.modules.context_embedder_base import ContextEmbedderBase
from models.modules.emission_scorer_base import EmissionScorerBase


class FewShotTextClassifier(torch.nn.Module):
    def __init__(self,
                 opt,
                 context_embedder: ContextEmbedderBase,
                 emission_scorer: EmissionScorerBase,
                 decoder: torch.nn.Module,
                 config: dict = None,  # store necessary setting or none-torch params
                 emb_log: str = None):
        super(FewShotTextClassifier, self).__init__()
        self.opt = opt
        self.context_embedder = context_embedder
        self.emission_scorer = emission_scorer
        self.decoder = decoder
        self.no_embedder_grad = opt.no_embedder_grad
        self.config = config
        self.emb_log = emb_log

    def forward(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_token_ids: torch.Tensor,
            support_segment_ids: torch.Tensor,
            support_nwp_index: torch.Tensor,
            support_input_mask: torch.Tensor,
            support_output_mask: torch.Tensor,
            test_target: torch.Tensor,
            support_target: torch.Tensor,
            support_num: torch.Tensor,
            support_sentence_feature: torch.Tensor = None,
            test_sentence_feature: torch.Tensor = None,
            support_sentence_target: torch.Tensor = None,
            test_sentence_target: torch.Tensor = None
    ):
        """
        :param test_token_ids: (batch_size, test_len)
        :param test_segment_ids: (batch_size, test_len)
        :param test_nwp_index: (batch_size, test_len)
        :param test_input_mask: (batch_size, test_len)
        :param test_output_mask: (batch_size, test_len)
        :param support_token_ids: (batch_size, support_size, support_len)
        :param support_segment_ids: (batch_size, support_size, support_len)
        :param support_nwp_index: (batch_size, support_size, support_len)
        :param support_input_mask: (batch_size, support_size, support_len)
        :param support_output_mask: (batch_size, support_size, support_len)
        :param test_target: index targets (batch_size, multi-label_num)
        :param support_target: one-hot targets (batch_size, support_size, multi-label_num, num_tags)
        :param support_num: (batch_size, 1)
        :param support_sentence_feature: same to label token ids
        :param test_sentence_feature: same to label token ids
        :param support_sentence_target: same to label token ids
        :param test_sentence_target: same to label token ids
        :return:
        """
        # reps for whole sentences: (batch_size, support_size, 1, emb_len)
        test_reps, support_reps = self.get_context_reps(
            test_token_ids, test_segment_ids, test_nwp_index, test_input_mask, support_token_ids, support_segment_ids,
            support_nwp_index, support_input_mask
        )

        # calculate emission: shape(batch_size, 1, no_pad_num_tag)
        test_output_mask = torch.ones(test_output_mask.shape[0], 1).to(test_output_mask.device)  # for sc, each test has only 1 output
        emission = self.emission_scorer(test_reps, support_reps, test_output_mask, support_output_mask, support_target)
        logits = emission

        # as we remove pad label (id = 0), so all label id sub 1. And relu is used to avoid -1 index
        test_target = torch.nn.functional.relu(test_target - 1)
        loss, prediction = torch.FloatTensor(0).to(test_target.device), None

        if self.training:
            loss = self.decoder.forward(logits=logits, mask=test_output_mask, tags=test_target)
        else:

            prediction = self.decoder.decode(logits=logits)
            # we block pad label(id=0) before by - 1, here, we add 1 back
            prediction = self.add_back_pad_label(prediction)
        if self.training:
            return loss
        else:
            return prediction

    def get_context_reps(
        self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            support_token_ids: torch.Tensor,
            support_segment_ids: torch.Tensor,
            support_nwp_index: torch.Tensor,
            support_input_mask: torch.Tensor,
    ):
        if self.no_embedder_grad:
            self.context_embedder.eval()  # to avoid the dropout effect of reps model
            self.context_embedder.requires_grad = False
        else:
            self.context_embedder.train()  # to avoid the dropout effect of reps model
            self.context_embedder.requires_grad = True
        _, _, test_reps, support_reps = self.context_embedder(
            test_token_ids, test_segment_ids, test_nwp_index, test_input_mask, support_token_ids, support_segment_ids,
            support_nwp_index, support_input_mask
        )
        if self.no_embedder_grad:
            test_reps = test_reps.detach()  # detach the reps part from graph
            support_reps = support_reps.detach()  # detach the reps part from graph
        return test_reps, support_reps

    def add_back_pad_label(self, predictions: List[List[int]]):
        for pred in predictions:
            for ind, l_id in enumerate(pred):
                pred[ind] += 1  # pad token is in the first place
        return predictions


class SchemaFewShotTextClassifier(FewShotTextClassifier):
    def __init__(
            self,
            opt,
            context_embedder: ContextEmbedderBase,
            emission_scorer: EmissionScorerBase,
            decoder: torch.nn.Module,
            config: dict = None,  # store necessary setting or none-torch params
            emb_log: str = None):
        super(SchemaFewShotTextClassifier, self).__init__(
            opt, context_embedder, emission_scorer, decoder, config, emb_log)

    def forward(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_token_ids: torch.Tensor,
            support_segment_ids: torch.Tensor,
            support_nwp_index: torch.Tensor,
            support_input_mask: torch.Tensor,
            support_output_mask: torch.Tensor,
            test_target: torch.Tensor,
            support_target: torch.Tensor,
            support_num: torch.Tensor,
            label_token_ids: torch.Tensor = None,
            label_segment_ids: torch.Tensor = None,
            label_nwp_index: torch.Tensor = None,
            label_input_mask: torch.Tensor = None,
            label_output_mask: torch.Tensor = None,
            support_sentence_feature: torch.Tensor = None,
            test_sentence_feature: torch.Tensor = None,
            support_sentence_target: torch.Tensor = None,
            test_sentence_target: torch.Tensor = None
    ):
        """
        few-shot sequence labeler using schema information
        :param test_token_ids: (batch_size, test_len)
        :param test_segment_ids: (batch_size, test_len)
        :param test_nwp_index: (batch_size, test_len)
        :param test_input_mask: (batch_size, test_len)
        :param test_output_mask: (batch_size, test_len)
        :param support_token_ids: (batch_size, support_size, support_len)
        :param support_segment_ids: (batch_size, support_size, support_len)
        :param support_nwp_index: (batch_size, support_size, support_len)
        :param support_input_mask: (batch_size, support_size, support_len)
        :param support_output_mask: (batch_size, support_size, support_len)
        :param test_target: index targets (batch_size, test_len)
        :param support_target: one-hot targets (batch_size, support_size, support_len, num_tags)
        :param support_num: (batch_size, 1)
        :param label_token_ids:
            if label_reps=cat:
                (batch_size, label_num * label_des_len)
            elif:
                (batch_size, label_num, label_des_len)
        :param label_segment_ids: same to label token ids
        :param label_nwp_index: same to label token ids
        :param label_input_mask: same to label token ids
        :param label_output_mask: same to label token ids
        :param support_sentence_feature: same to label token ids
        :param test_sentence_feature: same to label token ids
        :param support_sentence_target: same to label token ids
        :param test_sentence_target: same to label token ids
        :return:
        """
        test_reps, support_reps = self.get_context_reps(
            test_token_ids, test_segment_ids, test_nwp_index, test_input_mask,
            support_token_ids, support_segment_ids, support_nwp_index, support_input_mask
        )

        # get label reps, shape (batch_size, max_label_num, emb_dim)
        label_reps = self.get_label_reps(
            label_token_ids, label_segment_ids, label_nwp_index, label_input_mask,
        )

        # calculate emission: shape(batch_size, test_len, no_pad_num_tag)
        emission = self.emission_scorer(test_reps, support_reps, test_output_mask, support_output_mask, support_target,
                                        label_reps)
        if not self.training and self.emb_log:
            self.emb_log.write('\n'.join(['test_target\t' + '\t'.join(map(str, one_target))
                                          for one_target in test_target.tolist()]) + '\n')

        logits = emission

        # block pad of label_id = 0, so all label id sub 1. And relu is used to avoid -1 index
        test_target = torch.nn.functional.relu(test_target - 1)

        loss, prediction = torch.FloatTensor([0]).to(test_target.device), None

        if self.training:
            loss = self.decoder.forward(logits=logits, mask=test_output_mask, tags=test_target)
        else:
            prediction = self.decoder.decode(logits=logits)
            # we block pad label(id=0) before by - 1, here, we add 1 back
            prediction = self.add_back_pad_label(prediction)
        if self.training:
            return loss
        else:
            return prediction

    def get_label_reps(
            self,
            label_token_ids: torch.Tensor,
            label_segment_ids: torch.Tensor,
            label_nwp_index: torch.Tensor,
            label_input_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param label_token_ids:
        :param label_segment_ids:
        :param label_nwp_index:
        :param label_input_mask:
        :return:  shape (batch_size, label_num, label_des_len)
        """
        return self.context_embedder(
            label_token_ids, label_segment_ids, label_nwp_index, label_input_mask,  reps_type='label')


def main():
    pass


if __name__ == "__main__":
    main()
