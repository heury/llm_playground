"""Native MiniMind binary SMS classifier: last block + final norm + head tuning."""
import copy
import json
from pathlib import Path

import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from model.model_minimind import MiniMindConfig, MiniMindModel

LABELS = {0: '스팸아님', 1: '스팸'}


class MiniMindForSpamClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)
        if self.config.use_moe:
            raise ValueError('This classifier supports the Dense MiniMind backbone only.')
        self.config.tie_word_embeddings = False
        self.config.num_labels = 2
        self.config.id2label = LABELS.copy()
        self.config.label2id = {name: i for i, name in LABELS.items()}
        self.config.architectures = ['MiniMindForSpamClassification']
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, 2, bias=True)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        nn.init.zeros_(self.lm_head.bias)
        self.configure_trainable_parameters()

    def configure_trainable_parameters(self):
        prefixes = (f'model.layers.{len(self.model.layers)-1}.', 'model.norm.', 'lm_head.')
        for name, p in self.named_parameters():
            p.requires_grad_(name.startswith(prefixes))
        return [name for name, p in self.named_parameters() if p.requires_grad]

    def train(self, mode=True):
        super().train(mode)
        self.model.dropout.eval()
        for layer in self.model.layers[:-1]:
            layer.eval()
        return self

    def forward(self, input_ids, attention_mask=None, labels=None):
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError('input_ids must be a nonempty [batch, sequence] tensor.')
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape:
            raise ValueError('attention_mask shape must match input_ids.')
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError('attention_mask must contain only 0 and 1.')
        lengths = attention_mask.sum(-1)
        if torch.any(lengths == 0):
            raise ValueError('All-padding input is not supported.')
        expected = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :] < lengths[:, None]
        if not torch.equal(attention_mask.bool(), expected):
            raise ValueError('Only contiguous right padding is supported.')
        last = lengths.long() - 1
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        if not torch.all(input_ids[rows, last] == self.config.eos_token_id):
            raise ValueError('Each message must end with EOS at its last valid position.')
        hidden, _, _ = self.model(input_ids, attention_mask=attention_mask, use_cache=False)
        logits = self.lm_head(hidden[rows, last])
        loss = None
        if labels is not None:
            if labels.shape != (input_ids.shape[0],) or torch.any((labels < 0) | (labels > 1)):
                raise ValueError('labels must be [batch] class IDs 0 or 1.')
            loss = nn.functional.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)

    @classmethod
    def from_language_model(cls, path):
        path = Path(path)
        config = MiniMindConfig.from_pretrained(str(path), local_files_only=True)
        model = cls(config)
        if (path / 'model.safetensors').exists():
            from safetensors.torch import load_file
            state = load_file(str(path / 'model.safetensors'))
        else:
            state = torch.load(path / 'pytorch_model.bin', map_location='cpu', weights_only=True)
        if 'model.embed_tokens.weight' not in state and 'lm_head.weight' in state:
            state['model.embed_tokens.weight'] = state['lm_head.weight']
        backbone = {name[len('model.'):]: tensor for name, tensor in state.items() if name.startswith('model.')}
        model.model.load_state_dict(backbone, strict=True)
        return model

    def save_checkpoint(self, path, tokenizer=None, max_length=340):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.config.to_json_file(path / 'config.json')
        temporary = path / 'model.pth.tmp'
        torch.save({n: t.detach().cpu() for n, t in self.state_dict().items()}, temporary)
        temporary.replace(path / 'model.pth')
        (path / 'spam_config.json').write_text(json.dumps({
            'max_length': max_length, 'padding_side': 'right',
            'special_tokens': 'BOS + raw text tokens + EOS; preserve EOS when truncating',
            'id2label': LABELS, 'decision': 'argmax over two logits',
            'trainable': ['last Transformer block', 'final RMSNorm', 'lm_head'],
        }, ensure_ascii=False, indent=2) + '\n')
        if tokenizer is not None:
            tokenizer.save_pretrained(path)

    @classmethod
    def from_checkpoint(cls, path):
        path = Path(path)
        config = MiniMindConfig.from_pretrained(str(path), local_files_only=True)
        model = cls(config)
        model.load_state_dict(torch.load(path / 'model.pth', map_location='cpu', weights_only=True), strict=True)
        return model
