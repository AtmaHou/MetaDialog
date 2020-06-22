# coding:utf-8
import json
import collections
import random
from typing import List, Tuple, Dict


class RawDataLoaderBase:
    def __init__(self, *args, **kwargs):
        pass

    def load_data(self, path: str):
        pass


DataItem = collections.namedtuple("DataItem", ["seq_in", "seq_out", "label"])


class FewShotExample(object):
    """  Each few-shot example is a pair of (one query example, support set) """

    def __init__(
            self,
            gid: int,
            batch_id: int,
            test_id: int,
            domain_name: str,
            support_data_items: List[DataItem],
            test_data_item: DataItem
    ):
        self.gid = gid
        self.batch_id = batch_id
        self.test_id = test_id  # query relative index in one episode
        self.domain_name = domain_name

        self.support_data_items = support_data_items  # all support data items
        self.test_data_item = test_data_item  # one query data items

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return 'gid:{}\n\tdomain:{}\n\ttest_data:{}\n\ttest_label:{}\n\tsupport_data:{}'.format(
            self.gid,
            self.domain_name,
            self.test_data_item.seq_in,
            self.test_data_item.seq_out,
            self.support_data_items,
        )


class FewShotRawDataLoader(RawDataLoaderBase):
    def __init__(self, opt):
        super(FewShotRawDataLoader, self).__init__()
        self.opt = opt
        self.debugging = opt.do_debug

    def load_data(self, path: str) -> (List[FewShotExample], List[List[FewShotExample]], int):
        """
            load few shot data set
            input:
                path: file path
            output
                examples: a list, all example loaded from path
                few_shot_batches: a list, of fewshot batch, each batch is a list of examples
                max_len: max sentence length
            """
        with open(path, 'r') as reader:
            raw_data = json.load(reader)
            examples, few_shot_batches, max_support_size = self.raw_data2examples(raw_data)
        if self.debugging:
            examples, few_shot_batches = examples[:8], few_shot_batches[:2]
        return examples, few_shot_batches, max_support_size

    def raw_data2examples(self, raw_data: Dict) -> (List[FewShotExample], List[List[FewShotExample]], int):
        """
        process raw_data into examples
        """
        examples = []
        all_support_size = []
        few_shot_batches = []
        for domain_n, domain in raw_data.items():
            # Notice: the batch here means few shot batch, not training batch
            for batch_id, batch in enumerate(domain):
                one_batch_examples = []
                support_data_items, test_data_items = self.batch2data_items(batch)
                all_support_size.append(len(support_data_items))
                ''' Pair each test sample with full support set '''
                for test_id, test_data_item in enumerate(test_data_items):
                    gid = len(examples)
                    example = FewShotExample(
                        gid=gid,
                        batch_id=batch_id,
                        test_id=test_id,
                        domain_name=domain_n,
                        test_data_item=test_data_item,
                        support_data_items=support_data_items,
                    )
                    examples.append(example)
                    one_batch_examples.append(example)
                few_shot_batches.append(one_batch_examples)
        max_support_size = max(all_support_size)
        return examples, few_shot_batches, max_support_size

    def batch2data_items(self, batch: dict) -> (List[DataItem], List[DataItem]):
        support_data_items = self.get_data_items(parts=batch['support'])
        test_data_items = self.get_data_items(parts=batch['query'])
        return support_data_items, test_data_items

    def get_data_items(self, parts: dict) -> List[DataItem]:
        data_item_lst = []
        for seq_in, seq_out, label in zip(parts['seq_ins'], parts['seq_outs'], parts['labels']):
            # todo: move word-piecing into preprocessing module
            # label = token_label if self.opt.task == 'ml' else sent_label   # decide label type according to task
            data_item = DataItem(seq_in=seq_in, seq_out=seq_out, label=label)
            data_item_lst.append(data_item)
        return data_item_lst

    # def get_data_items(self, parts: dict) -> List[DataItem]:
    #     data_item_lst = []
    #     for text, label, wp_text, wp_label, wp_mark in zip(
    #             parts['seq_ins'], parts['seq_outs'],
    #             parts['tokenized_texts'], parts['word_piece_labels'], parts['word_piece_marks']):
    #         data_item = DataItem(text=text, label=label, wp_text=wp_text, wp_label=wp_label, wp_mark=wp_mark)
    #         data_item_lst.append(data_item)
    #     return data_item_lst
