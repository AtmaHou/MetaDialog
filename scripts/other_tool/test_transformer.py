# coding: utf-8
import json
import torch
# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertTokenizer, BertModel, BertForMaskedLM, AutoModel, AutoTokenizer, AutoModelWithLMHead, ElectraModel, ElectraForMaskedLM
# from bert_siamese_seq_labeling import SiameseBert
# from bert_transfer_seq_labeling import BertSeqLabeler


# MODEL_PATH = 'google/electra-small-discriminator'
# VOCAB = 'google/electra-small-discriminator'

# MODEL_PATH = '/users5/ythou/Projects/resources/electra-base-discriminator'
# MODEL_PATH = '/users5/ythou/Projects/resources/electra-small-discriminator'
# MODEL_PATH = '/users4/yklai/corpus/electra/electra-small-discriminator'


# MODEL_PATH = '/users4/ythou/Projects/Resources/bert-base-uncased/uncased_L-12_H-768_A-12/'
MODEL_PATH = '/users4/yklai/corpus/BERT/pytorch/uncased_L-12_H-768_A-12/'

VOCAB = MODEL_PATH

print('== tokenizing ===')
# Load pre-trained model tokenizer (vocabulary)
# tokenizer = AutoTokenizer.from_pretrained(VOCAB)
tokenizer = BertTokenizer.from_pretrained(VOCAB)

# Tokenized input
text = "apple is good fruit. I like to eat apple."
# text = "Who was Jim Henson ? Jim Henson was a puppeteer"
tokenized_text = tokenizer.tokenize(text)

# Mask a token that we will try to predict back with `BertForMaskedLM`
# masked_index = 10
masked_index = 6
tokenized_text[masked_index] = '[MASK]'
# print('Tokenized correct:',
#       tokenized_text == ['who', 'was', 'jim', 'henson', '?', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer'])
print('Tokenized correct:',
      tokenized_text == ['apple', 'is', 'good', 'fruit', '.', 'I', 'like', 'to', 'eat', 'apple', '.'])

# padding check
tokenized_text = ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[PAD]', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']


print('== Extracting hidden layer ===')
# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
print('indexed_tokens', indexed_tokens)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
# segments_ids = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

# padding check
segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
input_mask = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
# segments_ids = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
# input_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# ======== Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# ======== Load pre-trained model (weights) ========
# model = ElectraModel.from_pretrained(MODEL_PATH)
# model = AutoModel.from_pretrained(MODEL_PATH)
model = BertModel.from_pretrained(MODEL_PATH)
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

# print(model.__dict__)

last_hidden_state = model(input_ids=tokens_tensor, attention_mask=None, token_type_ids=segments_tensors)[0]  # electra only have one output
# last_hidden_state, pooler_output = model(input_ids=tokens_tensor, attention_mask=None, token_type_ids=segments_tensors)

# We have a hidden states for each of the 12 layers in model bert-base-uncased
print('Default Output Size:', last_hidden_state.shape)
# print('Default Output Size:', last_hidden_state.shape, pooler_output.shape)


# pad check
# print('Last encoded layers:\n', last_encoded_layers[0][5])
print('Last encoded layers:\n', last_hidden_state[0][6])


# ======== predict tokens ========
print('== LM predicting ===')
# Load pre-trained model (weights)
# model = AutoModelWithLMHead.from_pretrained(MODEL_PATH)
# model = ElectraForMaskedLM.from_pretrained(MODEL_PATH)
model = BertForMaskedLM.from_pretrained(MODEL_PATH)
model.eval()

# Predict all tokens
predictions = model(tokens_tensor, segments_tensors)[0]

print('predictions', predictions.shape)
# confirm we were able to predict 'henson'
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
print('predicted_token', predicted_token)
