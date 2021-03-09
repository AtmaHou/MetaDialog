# coding: utf-8
import json
import argparse
import sys
import os
import random
from raw_data_loader import RawDataLoaderBase, SLUDataLoader, StanfordDataLoader, TourSGDataLoader, SMPDataLoader
from data_generator import DataGeneratorBase, MiniIncludeGenerator, VanillaDataGenerator
from data_statistic import raw_data_statistic, label_stats, multi_label_stats, few_shot_data_statistic

DEFAULT_ID = '0'
DEFAULT_INPUT_PATH = 'D:/Data/Project/Data/stanford/'
DEFAULT_OUTPUT_DIR = 'D:/Data/Project/Data/MetaData/'
DEFAULT_CONFIG_PATH = 'D:/Data/Project/Data/MetaData/config/config{}.json'.format(DEFAULT_ID)


def dump_data(opt, data):
    """
    data: json format data
    output: 4 files are dumped to opt.output_dir
        train/dev/test json file:
            in few_shot_data format:
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

            # notice that if split eval set with label, there will be no "domain" in format:
                [  # list of fs episodes
                    {
                        'support':{'seq_ins':[], 'labels':[], 'seq_outs':[]},
                        'query':{'seq_ins':[], 'labels':[], 'seq_outs':[]},
                    },
                    ...,
                    episode_n
                ]

        opt.json:
            a log file records generation settings.
    """
    output_dir = opt.output_dir
    print('Dump data to dir: ', output_dir)
    if opt.dataset == 'snips':
        dump_dir_name = '{}.spt_s_{}.q_s_{}.ep_{}.cross_id_{}'.format(opt.dataset, opt.support_shots, opt.query_shot,
                                                                      opt.episode_num, opt.eval_config_id)
    else:
        dump_dir_name = "{}.spt_s_{}.q_s_{}.ep_{}.cross_id_{}".format(
            opt.dataset, opt.support_shots, opt.query_shot, opt.episode_num, opt.eval_config_id
        )

    dump_path = os.path.join(output_dir, dump_dir_name)

    if os.path.exists(dump_path) and os.listdir(dump_path) and not opt.allow_override:
        raise ValueError("Output directory () already exists and is not empty.")
    else:
        os.makedirs(dump_path, exist_ok=True)

    print('Output to: {}'.format(dump_path))

    for part_name, part in data.items():
        file_path = os.path.join(dump_path, '{}.json'.format(part_name))
        with open(file_path, 'w') as writer:
            json.dump(part, writer, ensure_ascii=False)

    file_path = os.path.join(dump_path, 'opt.json')
    with open(file_path, 'w') as writer:
        json.dump(vars(opt), writer, indent=2, ensure_ascii=False)


def build_data_loader(opt) -> RawDataLoaderBase:
    if opt.dataset in ['atis', 'snips']:
        data_loader = SLUDataLoader(opt)
    elif opt.dataset == 'stanford':
        data_loader = StanfordDataLoader(opt)
    elif opt.dataset == 'toursg':
        data_loader = TourSGDataLoader(opt)
    elif opt.dataset == 'smp':
        data_loader = SMPDataLoader(opt)
    else:
        raise NotImplementedError
    return data_loader


def add_data(data_set, data_item):
    data_set['seq_ins'].append(data_item[0])
    data_set['seq_outs'].append(data_item[1])
    data_set['labels'].append(data_item[2])
    return data_set


def split_eval_set_with_label(opt, raw_data):
    """ Both input and output are raw data format """
    try:
        # load config:
        with open(opt.eval_labels, 'r') as reader:
            config = json.load(reader)
        # split labels
        train_data = {'seq_ins': [], 'labels': [], 'seq_outs': []}
        dev_data = {'seq_ins': [], 'labels': [], 'seq_outs': []}
        test_data = {'seq_ins': [], 'labels': [], 'seq_outs': []}
        for old_domain_name, old_domain in raw_data:
            for ind in range(len(old_domain['label'])):
                seq_ins, seq_outs, labels = old_domain['seq_ins'][ind], old_domain['seq_outs'][ind], old_domain['labels'][ind]
                # for label in labels:
                label = labels[0]  # we assume that co-occur label all belongs to same domain
                if label in config['test']:
                    test_data = add_data(test_data, [seq_ins, seq_outs, labels])
                elif label in config['dev']:
                    dev_data = add_data(dev_data, [seq_ins, seq_outs, labels])
                elif label in config['ignore']:
                    continue
                else:
                    train_data = add_data(train_data, [seq_ins, seq_outs, labels])
        return {'train': train_data, 'dev': dev_data, 'test': test_data}
    except:
        raise RuntimeError('Error: test_labels and dev_labels are  when split_basis is label')


def split_eval_set_with_domain(opt, few_shot_data):
    """
    Both input and output are few_shot_data:
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
    try:
        # load config:
        with open(opt.eval_config, 'r') as reader:
            config = json.load(reader)
        # split labels
        train_data, dev_data, test_data = {}, {}, {}
        for old_domain_name, old_domain in few_shot_data.items():
            if old_domain_name in config['test']:
                test_data[old_domain_name] = old_domain
            elif old_domain_name in config['dev']:
                dev_data[old_domain_name] = old_domain
            elif old_domain_name in config['ignore']:
                continue
            else:
                train_data[old_domain_name] = old_domain
        return {'train': train_data, 'dev': dev_data, 'test': test_data}
    except:
        few_shot_data_statistic(opt, few_shot_data)
        raise RuntimeError('Error: test domains and dev domains are required when split_basis is domain')


def main():
    parser = argparse.ArgumentParser()
    # file path
    parser.add_argument("--input_path", type=str, default=DEFAULT_INPUT_PATH, help="path to the raw data dir")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="path to the result data dir")
    parser.add_argument("--dataset", default='stanford', help='dataset name to be processed',
                        choices=['atis', 'stanford', 'toursg', 'snips', 'smp'])

    # data size
    parser.add_argument('--episode_num', type=int, default=200,
                        help='num of few-shot episodes, each contain a support and a query set')
    parser.add_argument('--support_shots', type=int, default=5,
                        help='num of learning shots of each class for K-shot setting')
    parser.add_argument('--query_shot', type=int, default=32, help='sample in a episode for few shot learning style')
    parser.add_argument('--way', type=int, default=-1,
                        help='number of classes in N-way-K-shot setting, set <=0 to use all labels (default)')

    # bad case filtering
    parser.add_argument('--min_domain_size', type=int, default=50,
                        help='Abandon domains that have data amount less than this value')
    parser.add_argument('--min_label_appear', type=int, default=2,
                        help='Abandon labels with appear-times less than this value, to avoid useless support sample')

    # general setting
    parser.add_argument("--task", default='sc', choices=['sl', 'sc', 'slu'],
                        help="Task: sl:sequence labeling, sc:single label sent classify")
    parser.add_argument("--mark", type=str, default=DEFAULT_ID, help="A special mark in output file name to distinguish.")
    parser.add_argument('--dup_query', action='store_true', help='allow duplication between query and support set.')
    parser.add_argument('--allow_override', action='store_true', help='allow override generated data.')
    parser.add_argument('--check', action='store_true', help='check data after generation.')
    parser.add_argument("--style", default='fs',  choices=["fs", "va"],
                        help="output data styles. fs: few-shot episode style, va: directly all data in few-shot format")
    parser.add_argument('--intent_as_domain', action='store_true',
                        help='For sequence labeling in atis & snips, set true to separate domain using sentence-label.')
    parser.add_argument('-sd', '--seed', type=int, default=0, help='random seed, do nothing when sd < 0')

    # train/dev/test split
    parser.add_argument("--split_basis", default='domain', help='basis to split the data into sub-partitions',
                        choices=['domain', 'sent_label'])
    parser.add_argument("--eval_config", type=str, default=DEFAULT_CONFIG_PATH,
                        help="path json file that specify test/dev/ignore domains/labels.")
    parser.add_argument("--eval_config_id", type=int, default=DEFAULT_ID, help="eval config id")
    parser.add_argument("--label_type", type=str, default='both', choices=['cat', 'both', 'act', 'attribute'], help="eval config id")
    parser.add_argument("--remove_rate", type=float, default=80, help="the rate for removing duplicate sample")
    parser.add_argument("--use_fix_support", default=False, action="store_true", help="use fix support in dev data")
    opt = parser.parse_args()
    print('Parameter:\n', json.dumps(vars(opt), indent=2))

    if opt.seed >= 0:
        random.seed(opt.seed)

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    data_loader = build_data_loader(opt)
    raw_data = data_loader.load_data(opt.input_path)
    if opt.check:
        if opt.dataset != 'smp':
            raw_data_statistic(opt, raw_data)

    if opt.split_basis == 'sent_label':  # we pre-split domain by sentence level label
        raw_data = split_eval_set_with_label(opt, raw_data)

    if opt.task == 'sl':
        opt.way = -1
        print('Sequence labeling only support use entire domains label set as ways.')

    if 'fs' == opt.style:
        print('Start generate meta data.')
        generator = MiniIncludeGenerator(opt)
        if opt.dataset == 'smp':
            """train"""
            train_meta_data = generator.gen_data(raw_data['train'])
            print('Train: Few_shot_data gathered and start to dump data')
            few_shot_data_statistic(opt, train_meta_data)

            """dev"""
            if opt.use_fix_support:
                domains = raw_data['dev']['support'].keys()
                dev_meta_data = {}
                for domain in domains:
                    dev_meta_data[domain] = [{'support': raw_data['dev']['support'][domain],
                                              'query': raw_data['dev']['query'][domain]}]
            else:
                dev_meta_data = generator.gen_data(raw_data['dev']['query'])
            print('Dev: Few_shot_data gathered and start to dump data')
            few_shot_data_statistic(opt, dev_meta_data)

            """test"""
            if opt.use_fix_support:
                domains = raw_data['test']['support'].keys()
                test_meta_data = {}
                for domain in domains:
                    test_meta_data[domain] = [{'support': raw_data['test']['support'][domain],
                                              'query': raw_data['test']['query'][domain]}]
            else:
                test_meta_data = generator.gen_data(raw_data['test']['query'])
            print('Test: Few_shot_data gathered and start to dump data')
            few_shot_data_statistic(opt, test_meta_data)

            """meta data"""
            meta_data = {'train': train_meta_data, 'dev': dev_meta_data, 'test': test_meta_data}
        else:
            meta_data = generator.gen_data(raw_data)
            print('Few_shot_data gathered and start to dump data')
            few_shot_data_statistic(opt, meta_data)

            if opt.split_basis == 'domain':  # we pre-split domain by sentence level label
                meta_data = split_eval_set_with_domain(opt, meta_data)

    elif 'va' == opt.style:
        raise NotImplementedError
    else:
        raise ValueError("Not supported data style:{}".format(opt.style))

    # Dump data to disk
    dump_data(opt, meta_data)
    print('Process finished')


if __name__ == "__main__":
    main()
