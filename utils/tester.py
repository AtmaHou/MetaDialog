# coding: utf-8
from typing import List, Tuple, Dict
import torch
import logging
import sys
import os
import copy
import json
import collections
import subprocess
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
# My Staff
from utils.iter_helper import PadCollate, FewShotDataset
from utils.preprocessor import FewShotFeature, ModelInput
from utils.device_helper import prepare_model
from utils.model_helper import make_model, load_model
from models.few_shot_learner import FewShotLearner


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


RawResult = collections.namedtuple("RawResult", ["feature", "prediction"])


class TesterBase:
    """
    Support features:
        - multi-gpu [accelerating]
        - distributed gpu [accelerating]
        - padding when forward [better result & save space]
    """
    def __init__(self, opt, device, n_gpu):
        if opt.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                opt.gradient_accumulation_steps))

        self.opt = opt
        # Following is used to split the batch to save space
        self.batch_size = opt.test_batch_size
        self.device = device
        self.n_gpu = n_gpu

    def do_test(self, model: torch.nn.Module, test_features: List[FewShotFeature], id2label_map: dict,
                log_mark: str = 'test_pred')->Dict[str, float]:
        logger.info("***** Running eval *****")
        # print("***** Running eval *****")
        logger.info("  Num features = %d", len(test_features))
        logger.info("  Batch size = %d", self.batch_size)
        task_lst = id2label_map.keys()
        all_results = []  # {task: [] for task in task_lst}

        model.eval()
        data_loader = self.get_data_loader(test_features)

        for batch in tqdm(data_loader, desc="Eval-Batch Progress"):
            if self.n_gpu == 1:
                # multi-gpu does scattering it-self
                batch = tuple(t.to(self.device) if not isinstance(t, dict)
                              else {task: item.to(self.device) for task, item in t.items()} for t in batch)
            with torch.no_grad():
                predictions = self.do_forward(batch, model)
            for i, feature_gid in enumerate(batch[0]):  # iter over feature global id
                tmp_dict = {}
                for task in task_lst:
                    prediction = predictions[task][i]
                    feature = test_features[feature_gid.item()]
                    tmp_dict[task] = RawResult(feature=feature, prediction=prediction)
                all_results.append(tmp_dict)
                if model.emb_log:
                    model.emb_log.write('text_' + str(feature_gid.item()) + '\t'
                                        + '\t'.join(feature.test_feature_item.data_item.seq_in) + '\n')

        # close file handler
        if model.emb_log:
            model.emb_log.close()

        scores_map = self.eval_predictions(all_results, id2label_map, log_mark)
        return scores_map

    def get_data_loader(self, features):
        dataset = TensorDataset([self.unpack_feature(f) for f in features])
        if self.opt.local_rank == -1:
            sampler = RandomSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size)
        return data_loader

    def clone_model(self, model, id2label):
        # get a new instance
        return copy.deepcopy(model)

    def unpack_feature(self, feature) -> List[torch.Tensor]:
        raise NotImplementedError

    def do_forward(self, batch, model):
        prediction = model(*batch)
        return prediction

    def eval_predictions(self, *args, **kwargs) -> float:
        raise NotImplementedError


class FewShotTester(TesterBase):
    """
        Support features:
            - multi-gpu [accelerating]
            - distributed gpu [accelerating]
            - padding when forward [better result & save space]
    """
    def __init__(self, opt, device, n_gpu):
        super(FewShotTester, self).__init__(opt, device, n_gpu)

    def get_data_loader(self, features):
        dataset = FewShotDataset([self.unpack_feature(f) for f in features])
        if self.opt.local_rank == -1:
            sampler = SequentialSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)
        pad_collate = PadCollate(dim=-1, sp_dim=-2, sp_item_idx=[3, 8, 12])  # nwp_index, spt_tgt need special padding
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size, collate_fn=pad_collate)
        return data_loader

    def eval_predictions(self, all_results: List[Dict[str, RawResult]], id2label_map: Dict[str, Dict[int, str]],
                         log_mark: str) -> Dict[str, float]:
        """ Our result score is average score of all few-shot batches. """
        all_batches = self.reform_few_shot_batch(all_results)
        all_scores = []
        for b_id, fs_batch in all_batches:
            f1_map = self.eval_one_few_shot_batch(b_id, fs_batch, id2label_map, log_mark)
            all_scores.append(f1_map)

        res = {task: sum([item[task] for item in all_scores]) * 1.0 / len(all_scores) for task in id2label_map.keys()}

        # TODO: change to more general
        if 'sc' in self.opt.task and 'sl' in self.opt.task:
            all_intents = [[item['sc'] for item in fs_batch] for b_id, fs_batch in all_batches]
            all_slots = [[item['sl'] for item in fs_batch] for b_id, fs_batch in all_batches]

            # prediction is directly the predict ids [pad is removed in decoder]
            all_intent_pred_ids = [[result.prediction for result in batch_intents] for batch_intents in all_intents]
            all_intent_features = [[result.feature for result in batch_intents] for batch_intents in all_intents]
            all_intent_pred_labels = [[id2label_map['sc'][pred_ids[0]] for pred_ids in batch_pred_ids]
                                      for batch_pred_ids in all_intent_pred_ids]
            all_intent_target_labels = [[feature.test_feature_item.data_item.label[0] for feature in batch_features]
                                        for batch_features in all_intent_features]
            all_intent_pred_labels = np.array(all_intent_pred_labels)
            all_intent_target_labels = np.array(all_intent_target_labels)
            success = (all_intent_pred_labels == all_intent_target_labels)
            # print('success: {} - {}'.format(np.mean(success), success))

            all_slot_pred_ids = [[result.prediction for result in batch_slots] for batch_slots in all_slots]
            all_slot_features = [[result.feature for result in batch_slots] for batch_slots in all_slots]
            all_slot_pred_labels = [[[id2label_map['sl'][pred_id] for pred_id in pred_ids]
                                     for pred_ids in batch_pred_ids] for batch_pred_ids in all_slot_pred_ids]
            all_slot_target_labels = [[feature.test_feature_item.data_item.seq_out for feature in batch_features]
                                      for batch_features in all_slot_features]

            for b_idx, (b_pred_labels, b_target_labels) in enumerate(zip(all_slot_pred_labels, all_slot_target_labels)):
                for i_idx, (pred_labels, target_labels) in enumerate(zip(b_pred_labels, b_target_labels)):
                    for p_label, t_label in zip(pred_labels, target_labels):
                        if p_label != t_label:
                            success[b_idx][i_idx] = False
                            break

            success = success.astype(float)
            success = np.mean(success)

            res['success'] = success

            # print('res: {}'.format(res))

        return res

    def eval_one_few_shot_batch(self, b_id, fs_batch: List[Dict[str, RawResult]], id2label_map: Dict[str, Dict[int, str]],
                                log_mark: str) -> Dict[str, float]:
        f1_map = {}
        if 'sl' in self.opt.task:
            pred_file_name = 'sl.{}.{}.txt'.format(log_mark, b_id)
            output_prediction_file = os.path.join(self.opt.output_dir, pred_file_name)
            sl_fs_batch = [item['sl'] for item in fs_batch]
            self.writing_sl_prediction(sl_fs_batch, output_prediction_file, id2label_map['sl'])
            precision, recall, f1 = self.eval_with_script(output_prediction_file)
            f1_map['sl'] = f1

        if 'sc' in self.opt.task:
            pred_file_name = 'sc.{}.{}.txt'.format(log_mark, b_id)
            output_prediction_file = os.path.join(self.opt.output_dir, pred_file_name)
            sc_fs_batch = [item['sc'] for item in fs_batch]
            precision, recall, f1 = self.writing_sc_prediction(sc_fs_batch, output_prediction_file, id2label_map['sc'])
            f1_map['sc'] = f1

        return f1_map

    def writing_sc_prediction(self, fs_batch: List[RawResult], output_prediction_file: str, id2label: dict):
        tp, fp, fn = 0, 0, 0
        writing_content = []
        for result in fs_batch:
            pred_ids = result.prediction  # prediction is directly the predict ids [pad is removed in decoder]
            feature = result.feature
            pred_label = set([id2label[pred_id] for pred_id in pred_ids])
            label = set(feature.test_feature_item.data_item.label)
            writing_content.append({
                'seq_in': feature.test_feature_item.data_item.seq_in,
                'pred': list(pred_label),
                'label': list(label),
            })
            tp, fp, fn = self.update_f1_frag(pred_label, label, tp, fp, fn)  # update tp, fp, fn

        with open(output_prediction_file, "w") as writer:
            json.dump(writing_content, writer, indent=2)
        return self.compute_f1(tp, fp, fn)

    def update_f1_frag(self, pred_label, label, tp=0, fp=0, fn=0):
        tp += len(pred_label & label)
        fp += len(pred_label - label)
        fn += len(label - pred_label)
        return tp, fp, fn

    def compute_f1(self, tp, fp, fn):
        tp += 0.0000001  # to avoid zero division
        fp += 0.0000001
        fn += 0.0000001
        precision = 1.0 * tp / (tp + fp)
        recall = 1.0 * tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    def writing_sl_prediction(self, fs_batch: List[RawResult], output_prediction_file: str, id2label: dict):
        writing_content = []
        for result in fs_batch:
            prediction = result.prediction
            feature = result.feature
            pred_ids = prediction  # prediction is directly the predict ids
            if len(pred_ids) != len(feature.test_feature_item.data_item.seq_in):
                raise RuntimeError("Failed to align the pred_ids to texts: {},{} \n{},{} \n{},{}".format(
                    len(pred_ids), pred_ids,
                    len(feature.test_feature_item.data_item.seq_in), feature.test_feature_item.data_item.seq_in,
                    len(feature.test_feature_item.data_item.seq_out), feature.test_feature_item.data_item.seq_out
                ))
            for pred_id, word, true_label in zip(pred_ids, feature.test_feature_item.data_item.seq_in, feature.test_feature_item.data_item.seq_out):
                pred_label = id2label[pred_id]
                writing_content.append('{0} {1} {2}'.format(word, true_label, pred_label))
            writing_content.append('')
        with open(output_prediction_file, "w") as writer:
            writer.write('\n'.join(writing_content))

    def eval_with_script(self, output_prediction_file):
        script_args = ['perl', self.opt.eval_script]
        with open(output_prediction_file, 'r') as res_file:
            p = subprocess.Popen(script_args, stdout=subprocess.PIPE, stdin=res_file)
            p.wait()

            std_results = p.stdout.readlines()
            if self.opt.verbose:
                for r in std_results:
                    print(r)
            std_results = str(std_results[1]).split()
        precision = float(std_results[3].replace('%;', ''))
        recall = float(std_results[5].replace('%;', ''))
        f1 = float(std_results[7].replace('%;', '').replace("\\n'", ''))
        f1 = f1 / 100  # normalize to [0, 1]
        return precision, recall, f1

    def reform_few_shot_batch(self, all_results: List[Dict[str, RawResult]]
                              ) -> List[List[Tuple[int, Dict[str, RawResult]]]]:
        """
        Our result score is average score of all few-shot batches.
        So here, we classify all result according to few-shot batch id.
        """
        all_batches = {}
        task_lst = all_results[0].keys()
        has_task = list(task_lst)[0]
        for result in all_results:
            b_id = result[has_task].feature.batch_gid
            if b_id not in all_batches:
                all_batches[b_id] = [result]
            else:
                all_batches[b_id].append(result)
        return sorted(all_batches.items(), key=lambda x: x[0])

    def unpack_feature(self, feature: FewShotFeature) -> List[torch.Tensor]:
        ret = [
            torch.LongTensor([feature.gid]),
            # test
            feature.test_input.token_ids,
            feature.test_input.segment_ids,
            feature.test_input.nwp_index,
            feature.test_input.input_mask,
            feature.test_input.output_mask_map,
            # support
            feature.support_input.token_ids,
            feature.support_input.segment_ids,
            feature.support_input.nwp_index,
            feature.support_input.input_mask,
            feature.support_input.output_mask_map,
            # target
            feature.test_target_map,
            feature.support_target_map,
            # Special
            torch.LongTensor([len(feature.support_feature_items)]),  # support num
        ]
        return ret

    def do_forward(self, batch, model):
        (
            gid,  # 0
            test_token_ids,  # 1
            test_segment_ids,  # 2
            test_nwp_index,  # 3
            test_input_mask,  # 4
            test_output_mask,  # 5
            support_token_ids,  # 6
            support_segment_ids,  # 7
            support_nwp_index,  # 8
            support_input_mask,  # 9
            support_output_mask,  # 10
            test_target,  # 11
            support_target,  # 12
            support_num,  # 13
        ) = batch

        prediction = model(
            test_token_ids,
            test_segment_ids,
            test_nwp_index,
            test_input_mask,
            test_output_mask,
            support_token_ids,
            support_segment_ids,
            support_nwp_index,
            support_input_mask,
            support_output_mask,
            test_target,
            support_target,
            support_num,
        )
        return prediction

    def get_value_from_order_dict(self, order_dict, key):
        """"""
        for k, v in order_dict.items():
            if key in k:
                return v
        return []

    def clone_model(self, model, id2label_map):
        """ clone only part of params """
        # deal with data parallel model
        new_model: FewShotLearner
        old_model: FewShotLearner
        if self.opt.local_rank != -1 or self.n_gpu > 1 and hasattr(model, 'module'):  # the model is parallel class here
            old_model = model.module
        else:
            old_model = model
        # emission_dict = old_model.emission_scorer.state_dict()
        # old_num_tags = len(self.get_value_from_order_dict(emission_dict, 'label_reps'))

        config = {'num_tags': len(id2label_map['sl']) if 'sl' in id2label_map else 0, 'id2label_map': id2label_map}
        if 'num_anchors' in old_model.config:
            config['num_anchors'] = old_model.config['num_anchors']  # Use previous model's random anchors.
        # get a new instance for different domain
        new_model = make_model(opt=self.opt, config=config)
        new_model = prepare_model(self.opt, new_model, self.device, self.n_gpu)
        if self.opt.local_rank != -1 or self.n_gpu > 1:
            sub_new_model = new_model.module
        else:
            sub_new_model = new_model
        ''' copy weights and stuff '''
        if 'sl' in old_model.opt.task and old_model.seq_labeler_model.transition_scorer:
            # copy one-by-one because target transition and decoder will be left un-assigned
            sub_new_model.context_embedder.load_state_dict(old_model.context_embedder.state_dict())
            sub_new_model.seq_labeler_model.emission_scorer.load_state_dict(
                old_model.seq_labeler_model.emission_scorer.state_dict())
            for param_name in ['backoff_trans_mat', 'backoff_start_trans_mat', 'backoff_end_trans_mat']:
                sub_new_model.seq_labeler_model.transition_scorer.state_dict()[param_name].copy_(
                    old_model.seq_labeler_model.transition_scorer.state_dict()[param_name].data)
        else:
            sub_new_model.load_state_dict(old_model.state_dict())

        return new_model


class SchemaFewShotTester(FewShotTester):
    def __init__(self, opt, device, n_gpu):
        super(SchemaFewShotTester, self).__init__(opt, device, n_gpu)

    def get_data_loader(self, features):
        """ add label index into special padding """
        dataset = FewShotDataset([self.unpack_feature(f) for f in features])
        if self.opt.local_rank == -1:
            sampler = SequentialSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)
        pad_collate = PadCollate(dim=-1, sp_dim=-2, sp_item_idx=[3, 8, 12, 16])  # nwp_index, spt_tgt need sp-padding
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size, collate_fn=pad_collate)
        return data_loader

    def unpack_feature(self, feature: FewShotFeature) -> List[torch.Tensor]:
        ret = [
            torch.LongTensor([feature.gid]),  # 1
            # test
            feature.test_input.token_ids,  # 2
            feature.test_input.segment_ids,  # 3
            feature.test_input.nwp_index,  # 4
            feature.test_input.input_mask,  # 5
            feature.test_input.output_mask_map,  # 6
            # support
            feature.support_input.token_ids,  # 7
            feature.support_input.segment_ids,  # 8
            feature.support_input.nwp_index,  # 9
            feature.support_input.input_mask,  # 10
            feature.support_input.output_mask_map,  # 11
            # target
            feature.test_target_map,  # 11
            feature.support_target_map,  # 12
            # Special
            torch.LongTensor([len(feature.support_feature_items)]),  # 13, support num
            # label feature
            {key: label_input.token_ids for key, label_input in feature.label_input_map.items()},  # 14
            {key: label_input.segment_ids for key, label_input in feature.label_input_map.items()},  # 15
            {key: label_input.nwp_index for key, label_input in feature.label_input_map.items()},  # 16
            {key: label_input.input_mask for key, label_input in feature.label_input_map.items()},  # 17
            {key: label_input.output_mask_map for key, label_input in feature.label_input_map.items()},  # 18
        ]
        return ret

    def do_forward(self, batch, model):
        (
            gid,  # 0
            test_token_ids,  # 1
            test_segment_ids,  # 2
            test_nwp_index,  # 3
            test_input_mask,  # 4
            test_output_mask,  # 5
            support_token_ids,  # 6
            support_segment_ids,  # 7
            support_nwp_index,  # 8
            support_input_mask,  # 9
            support_output_mask,  # 10
            test_target,  # 11
            support_target,  # 12
            support_num,  # 13
            # label feature
            label_token_ids,  # 14
            label_segment_ids,  # 15
            label_nwp_index,  # 16
            label_input_mask,  # 17
            label_output_mask,  # 18
        ) = batch

        prediction = model(
            test_token_ids,
            test_segment_ids,
            test_nwp_index,
            test_input_mask,
            test_output_mask,
            support_token_ids,
            support_segment_ids,
            support_nwp_index,
            support_input_mask,
            support_output_mask,
            test_target,
            support_target,
            support_num,
            # label feature
            label_token_ids,
            label_segment_ids,
            label_nwp_index,
            label_input_mask,
            label_output_mask,
        )
        return prediction


def eval_check_points(opt, tester, test_features, test_id2label_map, device):
    all_cpt_file = list(filter(lambda x: '.cpt.pl' in x, os.listdir(opt.saved_model_path)))
    all_cpt_file = sorted(all_cpt_file,
                          key=lambda x: int(x.replace('model.step', '').replace('.cpt.pl', '')))
    max_score = 0
    for cpt_file in all_cpt_file:
        cpt_model = load_model(os.path.join(opt.saved_model_path, cpt_file))
        testing_model = tester.clone_model(cpt_model, test_id2label_map)
        if opt.mask_transition and 'sl' in opt.task:
            testing_model.label_mask = opt.test_label_mask.to(device)
        test_score = tester.do_test(testing_model, test_features, test_id2label_map, log_mark='test_pred')
        if test_score > max_score:
            max_score = test_score
        logger.info('cpt_file:{} - test:{}'.format(cpt_file, test_score))
    return max_score
