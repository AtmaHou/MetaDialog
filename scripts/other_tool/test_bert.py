# coding: utf-8
import json
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
# from bert_siamese_seq_labeling import SiameseBert
# from bert_transfer_seq_labeling import BertSeqLabeler

# BERT_BASE_UNCASED = '/users4/ythou/Projects/Resources/pytorch_bert-base-uncased.tar.gz'
# BERT_BASE_UNCASED_VOCAB = '/users4/ythou/Projects/Resources/pytorch_bert-base-uncased.tar.gz'
BERT_BASE_UNCASED = '/users4/ythou/Projects/Resources/bert-base-uncased/uncased_L-12_H-768_A-12/'
BERT_BASE_UNCASED_VOCAB = '/users4/ythou/Projects/Resources/bert-base-uncased/uncased_L-12_H-768_A-12/vocab.txt'


print('== tokenizing ===')
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(BERT_BASE_UNCASED_VOCAB)

# Tokenized input
text = "organization_founder[B] is persuading  employer[A]"
# text = "Who was Jim Henson ? Jim Henson was a puppeteer"
tokenized_text = tokenizer.tokenize(text)
print(233333333333333, tokenized_text)

# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 6
tokenized_text[masked_index] = '[MASK]'
print('Tokenized correct:',
      tokenized_text == ['who', 'was', 'jim', 'henson', '?', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer'])

# padding check
# tokenized_text = ['who', 'was', 'jim', 'henson', '?', '[PAD]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer']


print('== Extracting hidden layer ===')
# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
print('indexed_tokens', indexed_tokens)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
# segments_ids = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

# padding check
segments_ids = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
input_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


# ======== Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# ======== Load pre-trained model (weights) ========
model = BertModel.from_pretrained(BERT_BASE_UNCASED)
# model = SiameseBert.from_pretrained(BERT_BASE_UNCASED, opt=None, max_len=100)
# model = BertSeqLabeler.from_pretrained(BERT_BASE_UNCASED, opt=None, max_len=100, )

# ======== check components ========
# print('state dict"\n', model.state_dict().keys())
# print('sub state dict"\n', model.bert.state_dict().keys())

model.eval()

# ======== Predict hidden states features for each layer ========
# all_encoded_layers, _ = model(tokens_tensor, segments_tensors)
# We have a hidden states for each of the 12 layers in model bert-base-uncased
# print('Check success:', len(all_encoded_layers) == 12)
# print('all encoded layers:\n', all_encoded_layers)
#
last_encoded_layers, _ = model(tokens_tensor, segments_tensors, output_all_encoded_layers=False)
# We have a hidden states for each of the 12 layers in model bert-base-uncased
print('Size:', last_encoded_layers.size())
# print('Last encoded layers:\n', last_encoded_layers)

# pad check
# print('Last encoded layers:\n', last_encoded_layers[0][5])


# ======== predict tokens ========
print('== LM predicting ===')
# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained(BERT_BASE_UNCASED)
model.eval()

# Predict all tokens
predictions = model(tokens_tensor, segments_tensors)

# confirm we were able to predict 'henson'
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
print('predicted_token', predicted_token)
