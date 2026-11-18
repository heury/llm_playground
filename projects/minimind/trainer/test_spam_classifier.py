import tempfile
import unittest
from pathlib import Path
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_spam import MiniMindForSpamClassification
from dataset.spam_dataset import encode_message


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [3] * len(text)


class SpamClassifierTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = MiniMindConfig(hidden_size=32, num_hidden_layers=2, vocab_size=64,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
            intermediate_size=64, flash_attn=False)
        self.model = MiniMindForSpamClassification(self.config)
        self.ids = torch.tensor([[1, 7, 2, 0], [1, 8, 9, 2]])
        self.mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])

    def test_exact_training_scope_and_updates(self):
        expected = {n for n, _ in self.model.named_parameters() if n.startswith(('model.layers.1.', 'model.norm.', 'lm_head.'))}
        actual = {n for n, p in self.model.named_parameters() if p.requires_grad}
        self.assertEqual(expected, actual)
        self.assertFalse(self.model.config.tie_word_embeddings)
        self.assertIsNot(self.model.lm_head.weight, self.model.model.embed_tokens.weight)
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        self.model.train()
        self.assertFalse(self.model.model.layers[0].training)
        result = self.model(self.ids, self.mask, torch.tensor([0, 1]))
        self.assertEqual(result.logits.shape, (2, 2))
        result.loss.backward()
        for n, p in self.model.named_parameters():
            self.assertEqual(p.grad is not None, n in actual)
        torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=.01).step()
        for prefix in ('model.layers.1.', 'model.norm.', 'lm_head.'):
            self.assertTrue(any(not torch.equal(p, before[n]) for n, p in self.model.named_parameters() if n.startswith(prefix)))
        for n, p in self.model.named_parameters():
            if n not in actual: self.assertTrue(torch.equal(p, before[n]), n)

    def test_padding_invariance_and_invalid_inputs(self):
        self.model.eval()
        with torch.no_grad():
            batch = self.model(self.ids, self.mask).logits
            single = self.model(self.ids[:1, :3], self.mask[:1, :3]).logits
        torch.testing.assert_close(batch[:1], single, atol=1e-6, rtol=1e-5)
        for ids, mask in [(self.ids, torch.zeros_like(self.mask)),
                          (self.ids, torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])),
                          (self.ids, torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]])),
                          (torch.tensor([[1, 3, 4]]), None)]:
            with self.assertRaises(ValueError): self.model(ids, mask)

    def test_base_loading_and_save_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'base';base.mkdir()
            lm = MiniMindForCausalLM(self.config)
            lm.config.to_json_file(base / 'config.json')
            torch.save(lm.state_dict(), base / 'pytorch_model.bin')
            model = MiniMindForSpamClassification.from_language_model(base).eval()
            for n, p in model.model.state_dict().items():
                torch.testing.assert_close(p, lm.model.state_dict()[n], rtol=0, atol=0)
            with torch.no_grad(): expected = model(self.ids, self.mask).logits
            checkpoint = Path(directory) / 'classifier'
            model.save_checkpoint(checkpoint)
            loaded = MiniMindForSpamClassification.from_checkpoint(checkpoint).eval()
            with torch.no_grad(): actual = loaded(self.ids, self.mask).logits
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_tokenization_empty_and_truncation(self):
        tokenizer = FakeTokenizer()
        self.assertEqual(encode_message(tokenizer, 'abcdef', max_length=5), [1, 3, 3, 3, 2])
        for text in ['', '   ']:
            with self.assertRaises(ValueError): encode_message(tokenizer, text)


if __name__ == '__main__': unittest.main()
