#!/usr/bin/env python
import torch
from typing import Tuple, Dict, List
from models.modules.context_embedder_base import ContextEmbedderBase
from models.modules.emission_scorer_base import EmissionScorerBase
from models.modules.transition_scorer import TransitionScorerBase
from models.few_shot_seq_labeler import FewShotSeqLabeler, SchemaFewShotSeqLabeler
from models.few_shot_text_classifier import FewShotTextClassifier, SchemaFewShotTextClassifier


class FewShotLearner(torch.nn.Module):

    def __init__(self,
                 opt,
                 context_embedder: ContextEmbedderBase,
                 # emission_scorer_map: Dict[str, EmissionScorerBase],
                 # decoder_map: Dict[str, torch.nn.Module],
                 model_map: Dict[str, torch.nn.Module],
                 # transition_scorer: TransitionScorerBase = None,
                 config: dict = None,  # store necessary setting or none-torch params
                 emb_log: str = None):
        super(FewShotLearner, self).__init__()
        self.opt = opt
        self.context_embedder = context_embedder
        # self.emission_scorer_map = emission_scorer_map
        # self.transition_scorer = transition_scorer
        # self.decoder_map = decoder_map
        self.no_embedder_grad = opt.no_embedder_grad
        self.label_mask = None
        self.config = config
        self.emb_log = emb_log

        # self.task_lst = decoder_map.keys()
        self.model_map = model_map
        # for task in self.task_lst:
        #     if task == 'sl':
        #         self.model_map[task] = FewShotSeqLabeler(opt=opt,
        #                                                  context_embedder=context_embedder,
        #                                                  emission_scorer=emission_scorer_map[task],
        #                                                  decoder=decoder_map[task],
        #                                                  transition_scorer=transition_scorer,
        #                                                  label_mask=self.label_mask,
        #                                                  config=config,
        #                                                  emb_log=emb_log)
        #     elif task == 'sc':
        #         self.model_map[task] = FewShotTextClassifier(opt=opt,
        #                                                      context_embedder=context_embedder,
        #                                                      emission_scorer=emission_scorer_map[task],
        #                                                      decoder=decoder_map[task],
        #                                                      config=config,
        #                                                      emb_log=emb_log)

    def forward(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            test_output_mask_map: Dict[str, torch.Tensor],
            support_token_ids: torch.Tensor,
            support_segment_ids: torch.Tensor,
            support_nwp_index: torch.Tensor,
            support_input_mask: torch.Tensor,
            support_output_mask_map: Dict[str, torch.Tensor],
            test_target_map: Dict[str, torch.Tensor],
            support_target_map: Dict[str, torch.Tensor],
            support_num: torch.Tensor,
    ):
        """
        :param test_token_ids: (batch_size, test_len)
        :param test_segment_ids: (batch_size, test_len)
        :param test_nwp_index: (batch_size, test_len)
        :param test_input_mask: (batch_size, test_len)
        :param test_output_mask_map: A dict of (batch_size, test_len)
        :param support_token_ids: (batch_size, support_size, support_len)
        :param support_segment_ids: (batch_size, support_size, support_len)
        :param support_nwp_index: (batch_size, support_size, support_len)
        :param support_input_mask: (batch_size, support_size, support_len)
        :param support_output_mask_map: A dict of (batch_size, support_size, support_len)
        :param test_target_map: A dict of index targets (batch_size, test_len)
        :param support_target_map: A dict of one-hot targets (batch_size, support_size, support_len, num_tags)
        :param support_num: (batch_size, 1)
        :return:
        """
        # reps for tokens: (batch_size, support_size, nwp_sent_len, emb_len)
        seq_test_reps, seq_support_reps, tc_test_reps, tc_support_reps = self.get_context_reps(
            test_token_ids, test_segment_ids, test_nwp_index, test_input_mask, support_token_ids, support_segment_ids,
            support_nwp_index, support_input_mask
        )

        reps_map = {'sl': {'test': seq_test_reps, 'support': seq_support_reps},
                    'sc': {'test': tc_test_reps, 'support': tc_support_reps}}

        if self.training:
            loss = 0.
            for task in self.opt.task:
                loss += self.model_map[task](reps_map[task]['test'], test_output_mask_map[task],
                                             reps_map[task]['support'], support_output_mask_map[task],
                                             test_target_map[task], support_target_map[task], support_num,
                                             self.training)
            return loss
        else:
            prediction_map = {}
            for task in self.opt.task:
                prediction = self.model_map[task](reps_map[task]['test'], test_output_mask_map[task],
                                                  reps_map[task]['support'], support_output_mask_map[task],
                                                  test_target_map[task], support_target_map[task], support_num,
                                                  self.training)
                prediction_map[task] = prediction
            return prediction_map

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
        seq_test_reps, seq_support_reps, tc_test_reps, tc_support_reps = self.context_embedder(
            test_token_ids, test_segment_ids, test_nwp_index, test_input_mask, support_token_ids,
            support_segment_ids,
            support_nwp_index, support_input_mask
        )
        if self.no_embedder_grad:
            seq_test_reps = seq_test_reps.detach()  # detach the reps part from graph
            seq_support_reps = seq_support_reps.detach()  # detach the reps part from graph
            tc_test_reps = tc_test_reps.detach()  # detach the reps part from graph
            tc_support_reps = tc_support_reps.detach()  # detach the reps part from graph
        return seq_test_reps, seq_support_reps, tc_test_reps, tc_support_reps


class SchemaFewShotLearner(FewShotLearner):
    def __init__(
            self,
            opt,
            context_embedder: ContextEmbedderBase,
            # emission_scorer_map: Dict[str, EmissionScorerBase],
            # decoder_map: Dict[str, torch.nn.Module],
            model_map: Dict[str, torch.nn.Module],
            # transition_scorer: TransitionScorerBase = None,
            config: dict = None,  # store necessary setting or none-torch params
            emb_log: str = None
    ):
        # super(SchemaFewShotLearner, self).__init__(
        #     opt, context_embedder, emission_scorer_map, decoder_map, transition_scorer, config, emb_log)
        super(SchemaFewShotLearner, self).__init__(opt, context_embedder, model_map, config, emb_log)

        # self.task_lst = decoder_map.keys()
        self.model_map = model_map
        # for task in self.task_lst:
        #     if task == 'sl':
        #         self.model_map[task] = SchemaFewShotSeqLabeler(opt=opt,
        #                                                        context_embedder=context_embedder,
        #                                                        emission_scorer=emission_scorer_map[task],
        #                                                        decoder=decoder_map[task],
        #                                                        transition_scorer=transition_scorer,
        #                                                        label_mask=self.label_mask,
        #                                                        config=config,
        #                                                        emb_log=emb_log)
        #     elif task == 'sc':
        #         self.model_map[task] = SchemaFewShotTextClassifier(opt=opt,
        #                                                            context_embedder=context_embedder,
        #                                                            emission_scorer=emission_scorer_map[task],
        #                                                            decoder=decoder_map[task],
        #                                                            config=config,
        #                                                            emb_log=emb_log)

    def forward(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            test_output_mask_map: Dict[str, torch.Tensor],
            support_token_ids: torch.Tensor,
            support_segment_ids: torch.Tensor,
            support_nwp_index: torch.Tensor,
            support_input_mask: torch.Tensor,
            support_output_mask_map: Dict[str, torch.Tensor],
            test_target_map: Dict[str, torch.Tensor],
            support_target_map: Dict[str, torch.Tensor],
            support_num: torch.Tensor,
            label_token_ids_map: torch.Tensor = None,
            label_segment_ids_map: torch.Tensor = None,
            label_nwp_index_map: torch.Tensor = None,
            label_input_mask_map: torch.Tensor = None,
            label_output_mask_map: torch.Tensor = None,
    ):
        """
        few-shot sequence labeler using schema information
        :param test_token_ids: (batch_size, test_len)
        :param test_segment_ids: (batch_size, test_len)
        :param test_nwp_index: (batch_size, test_len)
        :param test_input_mask: (batch_size, test_len)
        :param test_output_mask_map: A dict of (batch_size, test_len)
        :param support_token_ids: (batch_size, support_size, support_len)
        :param support_segment_ids: (batch_size, support_size, support_len)
        :param support_nwp_index: (batch_size, support_size, support_len)
        :param support_input_mask: (batch_size, support_size, support_len)
        :param support_output_mask_map: A dict of (batch_size, support_size, support_len)
        :param test_target_map: A dict of index targets (batch_size, test_len)
        :param support_target_map: A dict of one-hot targets (batch_size, support_size, support_len, num_tags)
        :param support_num: (batch_size, 1)
        :param label_token_ids_map: A dict of tensor which
            if label_reps=cat:
                (batch_size, label_num * label_des_len)
            elif:
                (batch_size, label_num, label_des_len)
        :param label_segment_ids_map: A dict of tensor which is same to label token ids
        :param label_nwp_index_map: A dict of tensor which is same to label token ids
        :param label_input_mask_map: A dict of tensor which is same to label token ids
        :param label_output_mask_map: A dict of tensor which is same to label token ids
        :return:
        """
        # reps for tokens: (batch_size, support_size, nwp_sent_len, emb_len)
        seq_test_reps, seq_support_reps, tc_test_reps, tc_support_reps = self.get_context_reps(
            test_token_ids, test_segment_ids, test_nwp_index, test_input_mask, support_token_ids, support_segment_ids,
            support_nwp_index, support_input_mask
        )

        reps_map = {'sl': {'test': seq_test_reps, 'support': seq_support_reps},
                    'sc': {'test': tc_test_reps, 'support': tc_support_reps}}

        # get label reps, shape (batch_size, max_label_num, emb_dim)
        label_reps_map = {}
        for task in self.opt.task:
            label_reps_map[task] = self.get_label_reps(
                label_token_ids_map[task], label_segment_ids_map[task],
                label_nwp_index_map[task], label_input_mask_map[task]
            )

        if self.training:
            loss = 0.
            for task in self.opt.task:
                loss += self.model_map[task](reps_map[task]['test'], test_output_mask_map[task],
                                             reps_map[task]['support'], support_output_mask_map[task],
                                             test_target_map[task], support_target_map[task],
                                             support_num, label_reps_map[task], self.training)
            return loss
        else:
            prediction_map = {}
            for task in self.opt.task:
                prediction = self.model_map[task](reps_map[task]['test'], test_output_mask_map[task],
                                                  reps_map[task]['support'], support_output_mask_map[task],
                                                  test_target_map[task], support_target_map[task],
                                                  support_num, label_reps_map[task], self.training)
                prediction_map[task] = prediction
            return prediction_map

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
