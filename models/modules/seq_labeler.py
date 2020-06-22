#!/usr/bin/env python
from typing import Tuple, Union, List
import torch
from allennlp.nn.util import masked_log_softmax, get_range_vector, get_device_of


class SequenceLabeler(torch.nn.Module):
    def __init__(self):
        super(SequenceLabeler, self).__init__()

    def forward(self,
                logits: torch.Tensor,
                mask: torch.Tensor,
                tags: torch.Tensor) -> Tuple[Union[None, torch.Tensor],
                                             Union[None, torch.Tensor]]:
        """

        :param logits: (batch_size, seq_len, n_tags)
        :param mask: (batch_size, seq_len)
        :param tags: (batch_size, seq_len)
        :return:
        """
        return self._compute_loss(logits, mask, tags)

    def _compute_loss(self,
                      logits: torch.Tensor,
                      mask: torch.Tensor,
                      targets: torch.Tensor) -> torch.Tensor:
        """

        :param logits:
        :param mask:
        :param targets:
        :return:
        """

        batch_size, seq_len = mask.shape
        normalised_emission = masked_log_softmax(logits, mask.unsqueeze(-1), dim=-1)
        loss = normalised_emission.gather(dim=-1, index=targets.unsqueeze(-1))
        return -1 * loss.sum() / batch_size

    def decode(self, logits: torch.Tensor, masks: torch.Tensor) -> List[List[int]]:
        return self.remove_pad(preds=logits.argmax(dim=-1), masks=masks)

    def remove_pad(self, preds: torch.Tensor, masks: torch.Tensor) -> List[List[int]]:
        # remove predict result for padded token
        ret = []
        for pred, mask in zip(preds, masks):
            temp = []
            for l_id, mk in zip(pred, mask):
                if mk:
                    temp.append(int(l_id))
            ret.append(temp)
        return ret


class RuleSequenceLabeler(SequenceLabeler):
    def __init__(self, id2label):
        super(RuleSequenceLabeler, self).__init__()
        self.id2label = id2label

    def forward(self,
                logits: torch.Tensor,
                mask: torch.Tensor,
                tags: torch.Tensor) -> Tuple[Union[None, torch.Tensor],
                                             Union[None, torch.Tensor]]:
        """

        :param logits: (batch_size, seq_len, n_tags)
        :param mask: (batch_size, seq_len)
        :param tags: (batch_size, seq_len)
        :return:
        """
        return self._compute_loss(logits, mask, tags)

    def decode(self, logits: torch.Tensor, masks: torch.Tensor) -> List[List[int]]:
        preds = self.get_masked_preds(logits)
        return self.remove_pad(preds=preds, masks=masks)

    def get_masked_preds(self, logits):
        preds = []
        for logit in logits:
            pred_mask = torch.ones(logits.shape[-1]).to(logits.device)  # init mask for a sentence
            pred = []
            for token_logit in logit:
                token_pred = self.get_one_step_pred(token_logit, pred_mask)
                pred_mask = self.get_pred_mask(token_pred).to(logits.device)
                pred.append(token_pred)
            preds.append(pred)
        return preds

    def get_one_step_pred(self, token_logit, pred_mask):
        masked_logit = token_logit * pred_mask
        return masked_logit.argmax(dim=-1)

    def get_pred_mask(self, current_pred):
        mask = [1] * len(self.id2label)  # not here exclude [pad] label
        label_now = self.id2label[current_pred.item() + 1]  # add back [pad]
        if label_now == 'O':
            for ind, label in self.id2label.items():
                if 'I-' in label:
                    mask[ind] = 0
        elif label_now == '[PAD]':
            mask = [0] * len(self.id2label)
        elif 'B-' in label_now:
            for ind, label in self.id2label.items():
                if 'I-' in label and label.replace('I-', '') != label_now.replace('B-', ''):
                    mask[ind] = 0
        elif 'I-' in label_now:
            for ind, label in self.id2label.items():
                if 'I-' in label and label.replace('I-', '') != label_now.replace('I-', ''):
                    mask[ind] = 0
        else:
            raise ValueError('Wrong label {}'.format(label_now))
        mask = torch.FloatTensor(mask[1:])  # exclude [pad] label
        return mask

    def remove_pad(self, preds: torch.Tensor, masks: torch.Tensor) -> List[List[int]]:
        # remove predict result for padded token
        ret = []
        for pred, mask in zip(preds, masks):
            temp = []
            for l_id, mk in zip(pred, mask):
                if mk:
                    temp.append(int(l_id))
            ret.append(temp)
        return ret


def unit_test():
    logits = torch.tensor([[[0.1, 0.2, 0.5, 0.7, 0.3], [1.2, 0.8, 0.5, 0.6, 0.1], [0.4, 0.5, 0.5, 0.9, 1.2]],
                           [[1.9, 0.3, 0.5, 0.2, 0.3], [0.2, 0.1, 0.5, 0.4, 0.1], [0.4, 0.5, 0.1, 0.1, 0.2]]])
    labels = ['[PAD]', 'O', 'B-x', 'I-x', 'B-y', 'I-y']
    id2label = dict(enumerate(labels))
    mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    a = RuleSequenceLabeler(id2label)
    print(a.decode(logits, mask))
    a = SequenceLabeler()
    print(a.decode(logits, mask))


if __name__ == '__main__':
    unit_test()
