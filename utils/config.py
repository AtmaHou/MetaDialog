# coding: utf-8

""" Default values """
# Special tokens
MASK_TOKEN = '[MASK]'
SP_TOKEN_O = '[unused0]'  # append to support tokens: when attention on this token, then the label should be 'O'
SP_TOKEN_NO_MATCH = '[unused1]'  # append to support tokens: when attention on this token, then there's no label to pred

# Special labels
SP_LABEL_NO_MATCH = '[NO_MATCH]'  # to block the attention on other SP_TOKEN_NO_MATCH tokens
SP_LABEL_O = '[O]'  # replace the original support token's 'O' label, to block the attention on other 'O' tokens
