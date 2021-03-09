# coding: utf-8
# author: Atma Hou
import random
from typing import Dict, List, Tuple
import copy
import itertools
from collections import Counter

O_LABEL = 'O'


class DataGeneratorBase:
    def __init__(self, opt):
        self.opt = opt

    def gen_data(self, raw_data):
        raise NotImplementedError


class VanillaDataGenerator(DataGeneratorBase):
    """ Simple data generator for N-way K-shot few-shot data set. """
    def __init__(self, opt):
        super(VanillaDataGenerator, self).__init__(opt)

    def gen_data(self, raw_data):
        """
        directly convert all data into few-shot data format.
        input:
            {"partition/domain name" : { 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}}
        output:
            data for each domain in few shot format
            {
            'intent/domain_name1':[  # list of fs episodes
                {
                    'support':{'seq_ins':[empty], 'labels':[empty], 'seq_outs':[empty]},
                    'query':{'seq_ins':[all_data], 'labels':[all_data], 'seq_outs':[all_data]},
                },
            ],
        """
        all_data = {}
        for domain_name, domain_data in raw_data.items():
            all_data[domain_name] = {
                'support': {'seq_ins': [], 'labels': [], 'seq_outs': []},
                'query': domain_data}
        return all_data


class MiniIncludeGenerator(DataGeneratorBase):
    """
    Data generator for the situation that one sample can have multiple label, for example: a sentence may have multiple
    slot or belong to multiple classes.
    """
    def __init__(self, opt):
        super(MiniIncludeGenerator, self).__init__(opt)

    def gen_data(self, raw_data):
        """
        Generate the few shot data
        output: few shot data for each domain
        {
        'intent/domain_name1':[  # list of fs episodes
            {
                'support':{'seq_ins':[], 'labels':[], 'seq_outs':[]},
                'query':{'seq_ins':[], 'labels':[], 'seq_outs':[]},
            },
            ...,
            episode_n
        ],
    'intent/domain_name2': [episode1, episode2, ..., episode_n]}
        """
        few_shot_data = {}
        abandoned_domains = []
        for domain_name, domain_data in raw_data.items():  # Sample one domain's few shot data
            episodes = []
            if len(domain_data['labels']) < self.opt.min_domain_size:  # abandon too small domains.
                abandoned_domains.append([domain_name, len(domain_data['labels'])])
                continue

            domain_data = self.add_slu_labels(domain_data)

            label_bucket, d_id2label = self.get_label_bucket(domain_data)
            all_labels, del_labels = self.get_all_label_set(label_bucket)  # get all label and filter bad labels

            ''' remove the samples who has deleted labels'''
            label_bucket = self.del_samples_in_label_bucket(label_bucket, del_labels)

            for episode_id in range(self.opt.episode_num):  # sample few-shot episodes
                label_set = self.sample_label_set(all_labels)
                support_set, remained_data = self.sample_support_set(domain_data, label_set, label_bucket, d_id2label)
                query_set = self.get_query_set(domain_data if self.opt.dup_query else remained_data, label_set)
                del support_set['slu_labels']
                del query_set['slu_labels']
                episodes.append({'support': support_set, 'query': query_set})
                if episode_id % 100 == 0:
                    print('\tDomain:', domain_name, episode_id, 'episodes finished')
            one_domain_few_shot_data = episodes
            few_shot_data[domain_name] = one_domain_few_shot_data
        print('abandoned_domains: {}'.format(abandoned_domains))
        return few_shot_data

    def check_all_O(self, slots):
        flag = True
        for slot in slots:
            if slot != 'O':
                flag = False
                break
        return flag

    def add_slu_labels(self, domain_data):
        domain_data['slu_labels'] = []
        for d_id in range(len(domain_data['seq_ins'])):
            if isinstance(domain_data['labels'][d_id], list):
                label = domain_data['labels'][d_id][0]
            else:
                label = domain_data['labels'][d_id]
            if not self.check_all_O(domain_data['seq_outs'][d_id]):
                slu_labels = [label + '-' + slot for slot in domain_data['seq_outs'][d_id]]
            else:
                slu_labels = [label for _ in domain_data['seq_outs'][d_id]]
            domain_data['slu_labels'].append(slu_labels)
            if d_id < 3:
                print('{} - labels: {}'.format(d_id, domain_data['labels'][d_id]))
                print('{} - seq_outs: {}'.format(d_id, domain_data['seq_outs'][d_id]))
                print('{} - slu_labels: {}'.format(d_id, domain_data['slu_labels'][d_id]))
        return domain_data

    def del_samples_in_label_bucket(self, label_bucket: Dict[str, List[int]], del_labels: List[str]):
        """
        some label has not enough samples,
        so these samples with that label should be removed from samples of other remained labels
        :param label_bucket:
        :param del_labels:
        :return:
        """
        del_samples = []
        for label in del_labels:
            del_samples.extend(label_bucket[label])
        del_samples = list(set(del_samples))

        for label in label_bucket.keys():
            for sample_id in del_samples:
                if sample_id in label_bucket[label]:
                    label_bucket[label].remove(sample_id)
        return label_bucket

    def get_label_bucket(self, domain_data: dict) -> Tuple[Dict[str, List[int]], List[List[str]]]:
        """
        Re-category data by label.
        :param domain_data:
        :return: label bucket: {label name: data id}
                 d_id2label: {data id: label set}
        """
        label_bucket, d_id2label = {}, []
        if self.opt.task == 'sl':  # use token labels
            label_field = 'seq_outs'
        elif self.opt.task == 'sc':  # use sentence labels
            label_field = 'labels'
        elif self.opt.task == 'slu':  # use sentence labels and token labels
            label_field = 'slu_labels'
        else:
            raise ValueError('Wrong task in args: {}!'.format(self.opt.task))
        for d_id in range(len(domain_data['seq_ins'])):
            labels = list(set(domain_data[label_field][d_id]))   # all appeared label within a sample
            for label in labels:  # add data id into buckets of labels
                if label in label_bucket:
                    label_bucket[label].append(d_id)
                else:
                    label_bucket[label] = [d_id]
            d_id2label.append(labels)
        return label_bucket, d_id2label

    def get_all_label_set(self, label_bucket: Dict[str, List[int]]) -> List[str]:
        """ filtering out bad labels & get all label """
        all_labels = []
        del_labels = []
        for label in label_bucket:
            if len(label_bucket[label]) < self.opt.min_label_appear and self.opt.min_label_appear > 1:
                del_labels.append(label)
                continue
            else:
                all_labels.append(label)
        return all_labels, del_labels

    def sample_label_set(self, all_labels: List[str]) -> List[str]:
        """ sample N-way label set or use all labels within a domain """
        return random.choices(all_labels, k=self.opt.way) if self.opt.way > 0 else all_labels

    def sample_support_set(self, data_part, label_set, label_bucket, d_id2label):
        """
        Given data part, sampling k-shot data for n-way with Mini-including Algorithm
        :param data_part: { 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}
        :param label_set: N-way label set for data sampling
        :param label_bucket:  dict, {slot_name:[data_id]}
        :param d_id2label: list, {data id: label set}
        :return: result few shot data part { 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}
                 remained data part { 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}
        """
        support_shot_num = self.opt.support_shots
        tmp_label_bucket = copy.deepcopy(label_bucket)
        shot_counts = {ln: 0 for ln in label_set}
        selected_data_ids = []

        ''' Step0: duplicate the label id if some label samples number is smaller than support shot num'''
        for label in label_set:
            if len(tmp_label_bucket[label]) < support_shot_num:
                tmp_sample_ids = tmp_label_bucket[label]
                while len(tmp_sample_ids) < support_shot_num:
                    dup_sample_id = random.choice(tmp_sample_ids)
                    tmp_sample_ids.append(dup_sample_id)
                tmp_label_bucket[label] = tmp_sample_ids
        # print({k: len(v) for k, v in tmp_label_bucket.items()})

        ''' Step1: Sample learning shots, and record the selected data's id '''
        for label in label_set:
            while shot_counts[label] < support_shot_num:
                sampled_id = random.choice(tmp_label_bucket[label])  # sample 1 data from all data contains current label.
                self.update_label_bucket(sampled_id, tmp_label_bucket, d_id2label)  # remove selected data ids
                self.update_shot_counts(sampled_id, shot_counts, d_id2label)  # +1 shot for all labels of sampled data
                selected_data_ids.append(sampled_id)
        num_before = len(selected_data_ids)

        ''' Step2: Remove excess learning shots '''
        for d_id in selected_data_ids:
            to_be_removed_labels = d_id2label[d_id]
            can_remove = True
            for label in to_be_removed_labels:
                if label in shot_counts and shot_counts[label] - 1 < support_shot_num:
                    can_remove = False
                    break
            if can_remove:
                if random.randint(1, 100) < self.opt.remove_rate:  # Not to remove all removable data to give chances to extreme cases.
                    selected_data_ids.remove(d_id)
                    for label in to_be_removed_labels:
                        if label in shot_counts:
                            shot_counts[label] -= 1
        num_after = len(selected_data_ids)

        ''' Pick data item by selected id '''
        selected_data = {'seq_ins': [], 'labels': [], 'seq_outs': [], 'slu_labels': []}
        remained_data = {'seq_ins': [], 'labels': [], 'seq_outs': [], 'slu_labels': []}
        for d_id in range(len(data_part['seq_ins'])):
            s_in, s_out, lb, slu_lb = data_part['seq_ins'][d_id], data_part['seq_outs'][d_id], data_part['labels'][d_id], data_part['slu_labels'][d_id]
            if d_id in selected_data_ids:  # decide where data go
                repeat_num = selected_data_ids.count(d_id)
                while repeat_num:
                    if self.opt.way > 0:  # remove non-label_set labels
                        s_out, lb, slu_lb = self.remove_out_set_labels(s_out, lb, slu_lb, label_set)
                    self.add_data_to_set(selected_data, s_in, s_out, lb, slu_lb)
                    repeat_num -= 1
            else:
                self.add_data_to_set(remained_data, s_in, s_out, lb, slu_lb)

        if self.opt.check:
            # print('in support check...')
            # check support shot
            if self.opt.task in ['sc', 'slu']:
                selected_labels = list(itertools.chain.from_iterable(selected_data['labels']))
                label_shots = Counter(selected_labels)
                error_shot = False
                for lb, s in label_shots.items():
                    if s < self.opt.support_shots:
                        error_shot = True
                        print("Error: Lack shots of intent:", lb, s)
                if error_shot:
                    raise RuntimeError('Error in support shot number of intent.')

            if self.opt.task in ['sl', 'slu']:
                selected_labels = list(itertools.chain.from_iterable(selected_data['seq_outs']))
                label_shots = Counter(selected_labels)
                error_shot = False
                for lb, s in label_shots.items():
                    if s < self.opt.support_shots:
                        error_shot = True
                        print("Error: Lack shots of slot:", lb, s)
                if error_shot:
                    raise RuntimeError('Error in support shot number of slot.')

            if self.opt.task == 'sl':  # use token labels
                label_field = 'seq_outs'
            elif self.opt.task == 'sc':  # use sentence labels
                label_field = 'labels'
            elif self.opt.task == 'slu':  # use sentence labels and token labels
                label_field = 'slu_labels'
            else:
                raise ValueError('Wrong task in args: {}!'.format(self.opt.task))
            selected_labels = list(itertools.chain.from_iterable(selected_data[label_field]))
            # all label must appear in support set and no label appeared out of label set.
            diff = (set(selected_labels) - set(label_set)) | (set(label_set) - set(selected_labels))
            if diff:
                raise RuntimeError("Error: non-match label-set, \n differ: {} \n label set: {}, \n selected: {}".format(
                    diff, set(label_set), set(selected_labels)))

        return selected_data, remained_data

    def get_query_set(self, data_part, label_set):
        idxes = list(range(len(data_part['seq_ins'])))
        random.shuffle(idxes)
        query_set = {'seq_ins': [], 'labels': [], 'seq_outs': [], 'slu_labels': []}
        i = 0

        if self.opt.task == 'sl':  # use token labels
            label_field = 'seq_outs'
        elif self.opt.task == 'sc':  # use sentence labels
            label_field = 'labels'
        elif self.opt.task == 'slu':  # use sentence labels and token labels
            label_field = 'slu_labels'
        else:
            raise ValueError('Wrong task in args: {}!'.format(self.opt.task))

        total_data = len(data_part['labels'])
        while len(query_set['labels']) < self.opt.query_shot and len(query_set['labels']) < total_data:
            d_id = idxes[i]
            s_in, s_out, lb, slu_lb = data_part['seq_ins'][d_id], data_part['seq_outs'][d_id], data_part['labels'][d_id], data_part['slu_labels'][d_id]
            org_lb = data_part[label_field][d_id]
            if set(org_lb) & set(label_set):  # select data contain current label set
                if self.opt.way > 0:  # remove non-label_set labels
                    s_out, lb, slu_lb = self.remove_out_set_labels(s_out, lb, slu_lb, label_set)
                self.add_data_to_set(query_set, s_in, s_out, lb, slu_lb)
                if self.opt.check and (set(data_part[label_field][d_id]) - set(label_set)):
                    print('Abandon labels:', set(data_part[label_field][d_id]) - set(label_set))
            i += 1
        if self.opt.check:
            # print('in query check...')
            selected_labels = itertools.chain.from_iterable(query_set[label_field])
            diff = (set(selected_labels) - set(label_set))
            if diff:
                print("Error: non-match label-set, \n excess: {}, \n lack: {}".format(
                    set(selected_labels) - set(label_set), set(selected_labels) - set(label_set)))
        return query_set

    def update_label_bucket(self, sampled_id, tmp_label_bucket, d_id2label):
        """ remove selected data ids """
        labels = d_id2label[sampled_id]
        for label in labels:
            if sampled_id in tmp_label_bucket[label]:
                tmp_label_bucket[label].remove(sampled_id)

    def update_shot_counts(self, sampled_id, shot_counts, d_id2label):
        """ update shots count for all selected number appeared in sampled data """
        labels = d_id2label[sampled_id]
        for label in labels:
            if label in shot_counts:
                shot_counts[label] += 1

    def add_data_to_set(self, data_set, s_in, s_out, lb, slu_lb):
        data_set['seq_ins'].append(s_in)
        data_set['seq_outs'].append(s_out)
        data_set['labels'].append(lb)
        data_set['slu_labels'].append(slu_lb)

    def remove_out_set_labels(self, seq_out, lb, slu_lb, label_set):
        """ remove non-label set labels """
        if self.opt.task == 'sc':
            lb = list(set(lb) & set(label_set))
        elif self.opt.task == 'sl':
            for ind, lb in seq_out:
                if lb != 'O' and lb not in label_set:
                    seq_out[ind] = 'O'
        elif self.opt.task == 'slu':
            for ind, lb in slu_lb:
                if not lb.endswith('-O') and lb not in label_set:
                    slu_lb[ind] = lb.split('-')[0] + 'O'
        return seq_out, lb, slu_lb

