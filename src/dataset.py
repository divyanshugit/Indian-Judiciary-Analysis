import pandas as pd

import torch
from torch.utils.data import Dataset

class JusticeDataset(Dataset):

    def __init__(self, filename, maxlen, tokenizer):
        self.df = pd.read_csv(filename)
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sentence = self.df.loc[index, "location"]
        label = self.df.loc[index, "target"]
        tokens = self.tokenizer.tokenize(sentence)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        if len(tokens) < self.maxlen:
            tokens = tokens + ["[PAD]" for _ in range(self.maxlen - len(tokens))]
        else:
            tokens = tokens[: self.maxlen - 1] + ["[SEP]"]

        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))

        attention_mask = (input_ids != 0).long()
        
        return input_ids, attention_mask, label