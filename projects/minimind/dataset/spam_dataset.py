"""Raw SMS -> BOS/text/EOS, preserving URLs and numbers; right padding only."""
import torch
from torch.utils.data import Dataset


def encode_message(tokenizer, text, max_length=340):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Message must contain non-whitespace text.')
    if max_length < 3:
        raise ValueError('max_length must reserve BOS, text and EOS.')
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer must provide BOS and EOS IDs.')
    tokens = tokenizer.encode(text.strip(), add_special_tokens=False)
    return [tokenizer.bos_token_id] + tokens[:max_length-2] + [tokenizer.eos_token_id]


class SpamDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=340):
        self.items = []
        for row in rows:
            if row['label'] not in (0, 1):
                raise ValueError('Only labels 0 and 1 are supported.')
            self.items.append({'input_ids': encode_message(tokenizer, row['text'], max_length), 'label': row['label']})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class SpamCollator:
    def __init__(self, pad_token_id):
        if pad_token_id is None:
            raise ValueError('Tokenizer must have a pad token.')
        self.pad_token_id = pad_token_id

    def __call__(self, items):
        length = max(len(item['input_ids']) for item in items)
        ids = torch.full((len(items), length), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, item in enumerate(items):
            tokens = item['input_ids']
            ids[i, :len(tokens)] = torch.tensor(tokens)
            mask[i, :len(tokens)] = 1
        return {'input_ids': ids, 'attention_mask': mask,
                'labels': torch.tensor([item['label'] for item in items], dtype=torch.long)}
