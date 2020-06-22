# coding: utf-8
# author: Atma Hou
from collections import Counter
import itertools


def ave(lst):
    return sum(lst) / float(len(lst))


def multi_label_stats(opt, raw_data):
    print('Statistic of multi-label co-occur')
    stats = {}
    for domain_name, domain in raw_data.items():
        multi_labels = [tuple(l) for l in filter(lambda x: len(x) > 1, domain['labels'])]
        cnt = Counter(multi_labels)
        stats[domain_name] = cnt
        print("domain: {}, total multi-label data: {}, total multi-label case\n".format(
            domain_name, len(multi_labels), len(cnt)))
        for i in cnt.items():
            print(i)
    return stats


def label_stats(opt, raw_data):
    print('Statistic of label occur')
    stats = {}
    for domain_name, domain in raw_data.items():
        cnt = Counter(list(itertools.chain.from_iterable(domain['labels'])))
        stats[domain_name] = cnt
        print("domain: {}, total label: {}, total data: {}\n".format(domain_name, len(cnt), len(domain['labels'])))
        for i in cnt.items():
            print(i)
    return stats


def raw_data_statistic(opt, raw_data):
    print('ALL: domains', raw_data.keys())
    multi_label_stats(opt, raw_data)
    label_stats(opt, raw_data)


def few_shot_data_statistic(opt, gen_data):
    print('Statistic of generated data')
    stats = {}
    for domain_name, domain in gen_data.items():
        support_shots_cnt = []
        query_shots_cnt = []
        support_sizes = []
        for episode in domain:
            support_cnt = Counter(list(itertools.chain.from_iterable(episode['support']['labels'])))
            support_shots_cnt.extend(support_cnt.values())
            query_cnt = Counter(list(itertools.chain.from_iterable(episode['query']['labels'])))
            query_shots_cnt.extend(query_cnt.values())
            support_sizes.append(len(episode['support']['labels']))

        stats[domain_name] = {
            'ave_spt_shots': ave(support_shots_cnt),
            'ave_query_shots': ave(query_shots_cnt),
            'ave_support_size': ave(support_sizes),
        }
    print("statistic: {}\n".format(stats))
    return stats
