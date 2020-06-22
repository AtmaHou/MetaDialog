import torch
from torchcrf import CRF
seq_length, batch_size, num_tags = 3, 2, 5
emissions = torch.randn(seq_length, batch_size, num_tags)
tags = torch.tensor([[0, 1], [2, 4], [3, 1]], dtype=torch.long)  # (seq_length, batch_size)
model = CRF(num_tags)

''' Computing log likelihood '''
ll = model(emissions, tags)
print('Loss:', ll)

''' Computing log likelihood with mask '''
mask = torch.tensor([[1, 1], [1, 1], [1, 0]], dtype=torch.uint8)  # (seq_length, batch_size)
ll = model(emissions, tags, mask=mask)
print('mask loss', ll)

''' Decoding '''
res = model.decode(emissions)
print(res)

''' Decoding with mask '''
res = model.decode(emissions, mask=mask)
print(res)
