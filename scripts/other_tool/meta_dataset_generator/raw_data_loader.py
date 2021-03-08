# coding: utf-8
# author: Atma Hou
from typing import Dict, List, Tuple
import os
import re
import json


class RawDataLoaderBase:
    """ Load raw data"""
    def __init__(self, opt):
        self.opt = opt

    def load_data(self, path: str) -> dict:
        """
        Load all data into one dict.
        :param path: path to the file or dir of data
        :return: a dict store all data: {"partition/domain name" : { 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}}
        """
        raise NotImplementedError


class SLUDataLoader(RawDataLoaderBase):
    """
    Data loader for ATIS and SNIPS of data-format available at:
    https://github.com/MiuLab/SlotGated-SLU/tree/master/data
    """
    def __init__(self, opt):
        super(SLUDataLoader, self).__init__(opt)

    def load_data(self, path: str) -> dict:
        """
        Load all data into one dict.
        :param path: path to the file or dir of data
        :return: a dict store all data:  {"partition/domain name" : { 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}}
        """
        print('Start loading ATIS data from: ', path)
        all_data = {'seq_ins': [],  'labels': [], 'seq_outs': []}
        for part_name in ['train', 'valid', 'test']:
            seq_in_path = os.path.join(path, part_name, 'seq.in')
            label_path = os.path.join(path, part_name, 'label')
            seq_out_path = os.path.join(path, part_name, 'seq.out')
            with open(seq_in_path, 'r', encoding='utf-8') as reader:
                for line in reader:
                    all_data['seq_ins'].append(line.replace('\n', '').split())
            with open(seq_out_path, 'r', encoding='utf-8') as reader:
                for line in reader:
                    all_data['seq_outs'].append(line.replace('\n', '').split())
            with open(label_path, 'r', encoding='utf-8') as reader:
                for line in reader:
                    all_data['labels'].append(line.replace('\n', '').split('#'))
        if self.opt.intent_as_domain and self.opt.task == 'sl':  # re-split data by regarding label/intent as domain
            return self.split_data_by_intent(all_data)
        return {self.opt.dataset: all_data}

    def split_data_by_intent(self, all_data):
        """ Split data according to sentence labels, example
        output:
        {
            'intent/domain_name1':{ 'seq_ins':[], 'labels'[]:, 'seq_outs':[] }
            'intent/domain_name2':{ 'seq_ins':[], 'labels'[]:, 'seq_outs':[] }
        }
        """
        splited_data_lst = [self.split_one_data_part(all_data[domain_name]) for domain_name in all_data]
        splited_data = self.merge_data_part(splited_data_lst)
        return splited_data

    def split_one_data_part(self, data_part):
        all_domains = {}
        if not (len(data_part['seq_ins']) == len(data_part['seq_outs']) == len(data_part['labels'])):
            print('ERROR: seq_ins, seq_outs, labels are not equal in amount')
            raise RuntimeError
        for item in zip(data_part['seq_ins'], data_part['seq_outs'], data_part['labels']):
            seq_in, seq_out, labels = item[0], item[1], item[2]
            for label in labels:  # multi intent is split by '#'
                if label in all_domains:
                    all_domains[label]['seq_ins'].append(seq_in)
                    all_domains[label]['seq_outs'].append(seq_out)
                    all_domains[label]['labels'].append(label)
                else:
                    all_domains[label] = {}
                    all_domains[label]['seq_ins'] = [seq_in]
                    all_domains[label]['seq_outs'] = [seq_out]
                    all_domains[label]['labels'] = [label]
        return all_domains

    def merge_data_part(self, splited_data_lst):
        merged_data = {}
        for data_part in splited_data_lst:
            for intent_name in data_part:
                if intent_name in merged_data:
                    merged_data[intent_name]['seq_ins'].extend(data_part[intent_name]['seq_ins'])
                    merged_data[intent_name]['seq_outs'].extend(data_part[intent_name]['seq_outs'])
                    merged_data[intent_name]['labels'].extend(data_part[intent_name]['labels'])
                else:
                    merged_data[intent_name] = data_part[intent_name]
        return merged_data


class StanfordDataLoader(RawDataLoaderBase):
    """ data loader for staford-labeled LU data """
    def __init__(self, opt):
        super(StanfordDataLoader, self).__init__(opt)

    def load_data(self, path: str) -> dict:
        """
        Load all data into one dict.
        :param path: path to the file or dir of data
        :return: a dict store all data:  {"partition/domain name" : { 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}}
        """
        print('Start loading Stanford-Labeled data from: ', path)
        all_data = {}
        for domain in ['schedule', 'navigate', 'weather']:
            seq_ins, seq_outs, labels = [], [], []
            for part_name in ['train', 'dev', 'test']:
                data_path = os.path.join(path, '{}_{}'.format(domain, part_name))
                file_data = self.unpack_data(data_path)
                seq_ins.extend(file_data[0])
                seq_outs.extend(file_data[1])
                labels.extend(file_data[2])
            all_data[domain] = {'seq_ins': seq_ins,  'labels': labels, 'seq_outs': seq_outs}
        return all_data

    def unpack_data(self, data_path: str):
        seq_ins, seq_outs, labels = [], [], []  # There might be multiple utterances in one sample.
        tmp_in, tmp_out, tmp_lb = [], [], []
        with open(data_path, 'r', encoding='utf-8') as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                token, tag = line.split()
                if token in ['intent1', 'intent2', 'intent3', 'intent4', 'intent5']:
                    if tag != 'O':
                        tmp_lb.append(tag)
                else:
                    tmp_in.append(token)
                    tmp_out.append(tag)
                # check utterance finish
                if token == 'intent3':  # For most case, sample end with intent3
                    seq_ins.append(tmp_in)
                    seq_outs.append(tmp_out)
                    labels.append(tmp_lb)
                    tmp_in, tmp_out, tmp_lb = [], [], []
                if token in ['intent4', 'intent5']:  # There are some special case that end with intent4.
                    labels[-1].append(tag)
        return seq_ins, seq_outs, labels


class TourSGDataLoader(SLUDataLoader):
    """ data loader for TourSG data """

    def __init__(self, opt):
        super(TourSGDataLoader, self).__init__(opt)
        self.dis_flu_ptn = re.compile(r'\%\w{2,3}')
        # self.tag_ptn = re.compile(r'<(\w+|\s|-|=|\\|\")+>\s*\w+(\s\w+)*<\/\w+>')
        self.tag_ptn = re.compile(r'<(\w+|\s|-|=|\\|\")+>.*?<\/\w+>')
        self.left_tag_ptn = re.compile(r'<(\w+|\s|-|=|\\|\")+>')
        self.right_tag_ptn = re.compile(r'<\/\w+>')
        self.multi_space_ptn = re.compile(r'\s{2,}')
        self.punctuation = [',', '.', '!', '?', '-']

    def load_data(self, path: str, is_test=False) -> dict:
        """
        Load all data into one dict.
        :param path: path to the file or dir of data
        :return: a dict store all data:  {"partition/domain name" : { 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}}
        """
        print('Start loading DSTC4 TourSG data from: ', path)
        all_data = {}
        all_sub_folders = os.listdir(path)
        all_sub_folders = [folder for folder in all_sub_folders if len(folder) == 2 and not folder.startswith('.')
                           or folder.startswith('0') and len(folder) == 3]
        if is_test:
            all_sub_folders = all_sub_folders[0:1]
        print('all_sub_folders: {}'.format(all_sub_folders))
        for idx, sub_folder in enumerate(all_sub_folders):
            print('handle {} - {}'.format(idx, sub_folder))
            sub_dir = os.path.join(path, sub_folder)
            with open(os.path.join(sub_dir, 'log.json'), 'r') as log_fr:
                log_dict = json.load(log_fr)
            with open(os.path.join(sub_dir, 'label.json'), 'r') as label_fr:
                label_dict = json.load(label_fr)
            log_dict = self.transfer_to_idx_dict(log_dict)
            label_dict = self.transfer_to_idx_dict(label_dict)
            topic_dict, topic_set = self.get_topic(log_dict)
            print('topic_set: {}'.format(topic_set))
            tag_dict = self.get_label_and_ner(label_dict)
            utter_keys = set(topic_dict.keys()) & set(tag_dict.keys())
            for utter_idx in utter_keys:
                domain = topic_dict[utter_idx]
                label = [item for item in tag_dict[utter_idx][2]
                         if item and not item.startswith('_') and not item.startswith('None')]
                if label:
                    if domain in all_data:
                        all_data[domain]['seq_ins'].append(tag_dict[utter_idx][0])
                        all_data[domain]['seq_outs'].append(tag_dict[utter_idx][1])
                        all_data[domain]['labels'].append(label)
                    else:
                        all_data[domain] = {}
                        all_data[domain]['seq_ins'] = [tag_dict[utter_idx][0]]
                        all_data[domain]['seq_outs'] = [tag_dict[utter_idx][1]]
                        all_data[domain]['labels'] = [label]
        return all_data

    def transfer_to_idx_dict(self, raw_dict):
        tranfer_dict = {}
        for utterance in raw_dict['utterances']:
            utter_index = utterance['utter_index']
            if utter_index not in tranfer_dict:
                tranfer_dict[utter_index] = utterance
            else:
                raise RuntimeError('the utterance index is duplicated')
        return tranfer_dict

    def get_label_and_ner(self, label_dict):
        tag_dict = {}
        for utter_idx, utterance in label_dict.items():
            semantic_tagged_lst = utterance['semantic_tagged']
            semantic_tagged_lst = [self.del_dis_fluency(tag_utter) for tag_utter in semantic_tagged_lst]
            semantic_tagged_res = [self.get_bio(tag_utter) for tag_utter in semantic_tagged_lst]
            semantic_tagged_res = [item for item in semantic_tagged_res if item[0] and item[1]]
            if semantic_tagged_res:
                new_utter = ' '.join([' '.join(item[0]) for item in semantic_tagged_res])
                bio_tag = ' '.join([' '.join(item[1]) for item in semantic_tagged_res])

                # get labels
                labels = set()
                for item in utterance['speech_act']:
                    if self.opt.label_type == 'act':
                        labels.add(item['act'].strip())
                    elif self.opt.label_type in ['cat', 'both', 'attribute']:
                        for attr in item['attributes']:
                            attr = attr.strip()
                            if self.opt.label_type == 'cat':
                                final_label = item['act'].strip() + ('_' + attr if attr else '')
                                labels.add(final_label)
                            elif self.opt.label_type == 'both' or self.opt.label_type == 'attribute':
                                labels.add(attr)
                        if self.opt.label_type == 'both':
                            labels.add(item['act'].strip())
                    else:
                        raise NotImplementedError
                labels = list(labels)

                tag_dict[utter_idx] = [new_utter.split(' '), bio_tag.split(' '), labels]
        return tag_dict

    def del_dis_fluency(self, tag_utterance):
        new_tag_utterance = re.sub(self.dis_flu_ptn, '', tag_utterance)
        new_tag_utterance = new_tag_utterance.strip()
        return new_tag_utterance

    def get_bio(self, tag_utterance):
        all_match = [m.group(0) for m in re.finditer(self.tag_ptn, tag_utterance)]
        target_dict = {}
        if all_match:
            for match in all_match:
                all_pieces = match.split('</')
                main_tag = all_pieces[1].replace('>', '')
                all_pieces_with_content = all_pieces[0].split('>')
                all_pieces = all_pieces_with_content[0].split(' ')
                raw_content = all_pieces_with_content[1]
                cat_type = [piece for piece in all_pieces if 'CAT' in piece]
                cat_type = cat_type[0].replace('CAT="', '').replace('"', '')
                if cat_type == 'MAIN':
                    target_tag = main_tag
                else:
                    target_tag = cat_type

                content = raw_content.strip()
                content = self.split_punc(content)
                content = re.sub(self.multi_space_ptn, ' ', content)

                target_dict[match] = [(word, 'I-' + target_tag) if word not in self.punctuation else (word, 'O')
                                      for word in content.split(' ')]
                target_dict[match][0] = (target_dict[match][0][0], target_dict[match][0][1].replace('I-', 'B-'))

                tag_utterance = tag_utterance.replace(match, ' ' + raw_content + ' ')

        tag_utterance = self.split_punc(tag_utterance)
        tag_utterance = re.sub(self.multi_space_ptn, ' ', tag_utterance)

        tag_utterance_lst = tag_utterance.split(' ')
        tags = ['O'] * len(tag_utterance_lst)
        for _, tag_lst in target_dict.items():
            for item in tag_lst:
                idx = tag_utterance_lst.index(item[0])
                tags[idx] = item[1]

        assert len(tag_utterance_lst) == len(tags)

        if re.search(self.left_tag_ptn, ' '.join(tag_utterance_lst)) or re.search(self.right_tag_ptn, ' '.join(tag_utterance_lst)):
            return None, None
        else:
            return tag_utterance_lst, tags

    def split_punc(self, utterance):
        for punc in self.punctuation:
            utterance = utterance.replace(punc, ' ' + punc)
        utterance = utterance.strip()
        return utterance

    def get_topic(self, log_dict):
        """
        topic(domain) list:
            - OPENING (need to be filter)
            - CLOSING (need to be filter)
            - ITINERARY
            - ACCOMMODATION
            - ATTRACTION
            - FOOD
            - SHOPPING
            - TRANSPORTATION
            - OTHER (need to be filter)
        """
        topic_dict = {}
        topic_set = set()
        for utter_idx, utterance in log_dict.items():
            utter_topic = utterance['segment_info']['topic']
            if utter_topic not in ['OPENING', 'CLOSING', 'OTHER']:
                topic_dict[utter_idx] = utter_topic
                topic_set.add(utter_topic)
        return topic_dict, topic_set


class SMPDataLoader(RawDataLoaderBase):
    """ data loader for SMP (Chinese) data """
    
    def __init__(self, opt):
        super(SMPDataLoader, self).__init__(opt)

    def load_data(self, path: str, with_dev: bool = True):
        """
        Load all data into one dict.
        :param path: path to the file or dir of data
        :param with_dev: decide whether handle dev data or not
        :return:
            a dict store train data:  {"partition/domain name" : { 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}}
            +
            a dict store dev & test data: {"partition/domain name" : { 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}}
            +
            a dict store support data: {"partition/domain name" : { 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}}
        """
        print('Start loading SMP (Chinese) data from: ', path)
        train_data = self.load_normal_data(os.path.join(path, 'train'))
        # dev_data = self.load_normal_data(os.path.join(path, 'dev'))
        dev_support_data, dev_data = self.load_support_test_data(os.path.join(path, 'dev'))
        test_support_data, test_data = self.load_support_test_data(os.path.join(path, 'test'))
        return {'train': train_data, 'dev': {'support': dev_support_data, 'query': dev_data},
                'test': {'support': test_support_data, 'query': test_data}}

    def load_normal_data(self, path: str):
        all_data = {}
        all_files = [os.path.join(path, filename) for filename in os.listdir(path) if filename.endswith('.json')]
        print('all_files: {} - {}'.format(path, all_files))
        for one_file in all_files:
            part_data = self.unpack_train_data(one_file)

            for domain, data in part_data.items():
                if domain not in all_data:
                    all_data[domain] = {"seq_ins": [], "seq_outs": [], "labels": []}
                all_data[domain]['seq_ins'].extend(part_data[domain]['seq_ins'])
                all_data[domain]['seq_outs'].extend(part_data[domain]['seq_outs'])
                all_data[domain]['labels'].extend(part_data[domain]['labels'])
        return all_data

    def load_support_test_data(self, path, support_folder_name='support', test_folder_name='correct'):
        support_files = [os.path.join(path, support_folder_name, filename)
                         for filename in os.listdir(os.path.join(path, support_folder_name))
                         if filename.endswith('.json')]
        support_data, support_text_set = self.unpack_support_data(support_files)

        test_files = [os.path.join(path, test_folder_name, filename)
                      for filename in os.listdir(os.path.join(path, test_folder_name))
                      if filename.endswith('.json')]

        test_data = {}
        for one_file in test_files:
            part_data = self.unpack_train_data(one_file)

            for domain, data in part_data.items():
                if domain not in test_data:
                    test_data[domain] = {"seq_ins": [], "seq_outs": [], "labels": []}
                test_data[domain]['seq_ins'].extend(part_data[domain]['seq_ins'])
                test_data[domain]['seq_outs'].extend(part_data[domain]['seq_outs'])
                test_data[domain]['labels'].extend(part_data[domain]['labels'])

        return support_data, test_data

    def unpack_support_data(self, all_data_path):
        support_data = {}
        support_text_set = set()
        for data_path in all_data_path:

            with open(data_path, 'r', encoding='utf-8') as reader:
                json_data = json.load(reader)

            print('support data num: {} - {}'.format(len(json_data), data_path))

            for item in json_data:
                domain = item['domain']

                if domain not in support_data:
                    support_data[domain] = {"seq_ins": [], "seq_outs": [], "labels": []}

                seq_in, seq_out, label = self.handle_one_utterance(item)

                support_text_set.add(''.join(seq_in))

                support_data[domain]['seq_ins'].append(seq_in)
                support_data[domain]['seq_outs'].append(seq_out)
                support_data[domain]['labels'].append([label])

        return support_data, support_text_set

    def unpack_train_data(self, data_path: str, remove_set: set = None):
        part_data = {}
        with open(data_path, 'r', encoding='utf-8') as reader:
            json_data = json.load(reader)

        print('all data num: {} - {}'.format(len(json_data), data_path))

        for item in json_data:
            domain = item['domain']
            if domain not in part_data:
                part_data[domain] = {"seq_ins": [], "seq_outs": [], "labels": []}

            seq_in, seq_out, label = self.handle_one_utterance(item)
            part_data[domain]['seq_ins'].append(seq_in)
            part_data[domain]['seq_outs'].append(seq_out)
            part_data[domain]['labels'].append([label])

        return part_data

    def handle_one_utterance(self, item):
        # text = item['text'].replace(' ', '')
        text = re.sub(r"\s+", "", item['text'])
        seq_in = list(text)

        seq_out = ['O'] * len(seq_in)
        if 'slots' in item:
            slots = item['slots']
            for slot_key, slot_value in slots.items():
                if not isinstance(slot_value, list):
                    if isinstance(slot_value, dict):
                        slot_value = self.flat_dict(slot_key, slot_value)
                    else:
                        slot_value = [(slot_key, slot_value)]
                else:
                    slot_value = [(slot_key, s_val) for s_val in slot_value]
                for (s_key, s_val) in slot_value:
                    s_val = s_val.replace(' ', '')
                    if s_val in text:
                        s_idx = text.index(s_val)
                        s_end = s_idx + len(s_val)
                        seq_out[s_idx] = 'B-' + s_key
                        for idx in range(s_idx + 1, s_end):
                            seq_out[idx] = 'I-' + s_key
                    else:
                        print('text: {}'.format(text))
                        print('  slot_key: {} - slot_value: {}'.format(s_key, s_val))

        label = item['intent'] if 'intent' in item else 'O'
        if 'intent' not in item:
            print('error: item: {}'.format(item))

        return seq_in, seq_out, label

    def flat_dict(self, prefix_key: str, data: dict):
        slots_lst = []
        for k, v in data.items():
            if isinstance(v, dict):
                slots_lst.extend(self.flat_dict(prefix_key + '-' + k, v))
            else:
                if isinstance(v, list):
                    for v1 in v:
                        slots_lst.append((prefix_key + '-' + k, v1))
                else:
                    slots_lst.append((prefix_key + '-' + k, v))
        return slots_lst


if __name__ == '__main__':
    print('Start unit test.')
    import argparse
    parse = argparse.ArgumentParser()
    opt = parse.parse_args()
    opt.intent_as_domain = False
    opt.task = 'sc'
    opt.dataset = 'smp'
    opt.label_type = 'intent'

    smp_path = '/Users/lyk/Work/Dialogue/FewShot/SMP/'
    smp_loader = SMPDataLoader(opt)

    smp_data = smp_loader.load_data(path=smp_path)
    train_data, dev_data, support_data = smp_data['train'], smp_data['dev'], smp_data['support']

    print("train: smp domain number: {}".format(len(train_data)))
    print("train: all smp domain: {}".format(train_data.keys()))
    print("dev: smp domain number: {}".format(len(dev_data)))
    print("dev: all smp domain: {}".format(dev_data.keys()))
    print("support: smp domain number: {}".format(len(support_data)))
    print("support: all smp domain: {}".format(support_data.keys()))

    print("support: {}".format(support_data))


