"""A40에서 확인한 실패 경로를 다운로드 없이 검증합니다."""

import ast
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from datetime import datetime
from zoneinfo import ZoneInfo
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
from torch import nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
    PreTrainedTokenizerFast,
    ViTConfig,
    ViTImageProcessor,
    ViTModel,
)

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
NOTEBOOKS = Path(os.environ.get('VLM_NOTEBOOK_DIR', PROJECT / 'notebooks'))
RUN_ID = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d-%H%M%S-regressions')
ARTIFACTS = Path(os.environ.get('VLM_TEST_OUTPUT_DIR', ROOT / 'outputs/vlm-from-scratch' / RUN_ID))
ARTIFACTS.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)


def notebook(part):
    return next(NOTEBOOKS.glob(f'part-{part:02d}-*.py'))


def cells(part):
    return [n for n in ast.parse(notebook(part).read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == '_']


def namespace(**kwargs):
    return dict(torch=torch, nn=nn, F=F, np=np, Image=Image,
                ImageEnhance=ImageEnhance, ImageFilter=ImageFilter, ImageOps=ImageOps,
                AutoModel=AutoModel, AutoModelForCausalLM=AutoModelForCausalLM,
                AutoTokenizer=AutoTokenizer, ViTModel=ViTModel,
                ViTImageProcessor=ViTImageProcessor, json=json,
                device=torch.device('cpu'), os=os, **kwargs)


def definitions(part, names, env):
    nodes = [copy.deepcopy(n) for cell in cells(part) for n in cell.body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(notebook(part)), 'exec'), env)
    return env


def tokenizer(vocab_size=32):
    vocab = {'[PAD]': 0, '[UNK]': 1, '[EOS]': 2}
    vocab.update({f'word{i}': i for i in range(3, vocab_size)})
    raw = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    raw.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='[PAD]',
                                  unk_token='[UNK]', eos_token='[EOS]')


def vision_config(hidden=32):
    return ViTConfig(image_size=16, patch_size=8, hidden_size=hidden,
                     num_hidden_layers=1, num_attention_heads=4, intermediate_size=hidden * 2)


def language_config(hidden=24, vocab_size=32):
    return LlamaConfig(hidden_size=hidden, intermediate_size=hidden * 2,
                       num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                       vocab_size=vocab_size, max_position_embeddings=128,
                       pad_token_id=0, eos_token_id=2)


class RegressionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.temp = tempfile.TemporaryDirectory(dir=ARTIFACTS)
        self.addCleanup(self.temp.cleanup)
        self.output_dir = Path(self.temp.name)

    def checkpoint(self, folder='mini-vlm-flickr8k', small=False, vocab_size=32):
        vdim, ldim = (16, 16) if small else (32, 24)
        vname, lname = ('test/vit-small', 'test/lm-small') if small else ('test/vit-large', 'test/lm-large')
        env = definitions(1, ['VisionProjector'], namespace())
        projector = env['VisionProjector'](vdim, ldim)
        language = LlamaForCausalLM(language_config(ldim, vocab_size))
        data = dict(config=dict(vision_model_name=vname, lm_model_name=lname,
                                vision_dim=vdim, language_dim=ldim),
                    projector_state_dict=projector.state_dict(),
                    language_model_state_dict=language.state_dict())
        path = self.output_dir / folder
        path.mkdir(exist_ok=True)
        filename = 'mini_vlm_multitask.pt' if folder == 'mini-vlm-multitask' else 'mini_vlm_full.pt'
        torch.save(data, path / filename)
        tokenizer(vocab_size).save_pretrained(path / 'tokenizer')
        ViTImageProcessor(size={'height': 16, 'width': 16}).save_pretrained(path / 'image_processor')
        return data, path / filename

    def run_loading_cell(self, part):
        env = definitions(part, ['VisionProjector', 'MultiImageVLM' if part == 5 else 'MiniVLM'], namespace())
        env['output_dir'] = self.output_dir
        cell = next(c for c in cells(part) if any(
            isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'vision_model_name'
                                             for t in n.targets) for n in c.body))
        for arg in cell.args.args:
            if arg.arg.startswith('execution_step_'):
                env[arg.arg] = 0
        calls = {'vision': [], 'language': [], 'tokenizer': [], 'processor': []}
        def load_vision(name):
            calls['vision'].append(name)
            return ViTModel(vision_config(32 if 'large' in name else 16))
        def load_language(name):
            calls['language'].append(name)
            return LlamaForCausalLM(language_config(24 if 'large' in name or '360M' in name else 16))
        real_tokenizer = AutoTokenizer.from_pretrained
        real_processor = ViTImageProcessor.from_pretrained
        def load_tokenizer(name):
            calls['tokenizer'].append(str(name))
            return real_tokenizer(name, local_files_only=True) if Path(name).is_dir() else tokenizer()
        def load_processor(name):
            calls['processor'].append(str(name))
            return real_processor(name, local_files_only=True) if Path(name).is_dir() else ViTImageProcessor(size={'height': 16, 'width': 16})
        body = [copy.deepcopy(n) for n in cell.body if not isinstance(n, ast.Return)]
        with mock.patch.object(ViTModel, 'from_pretrained', side_effect=load_vision), \
             mock.patch.object(AutoModelForCausalLM, 'from_pretrained', side_effect=load_language), \
             mock.patch.object(AutoTokenizer, 'from_pretrained', side_effect=load_tokenizer), \
             mock.patch.object(ViTImageProcessor, 'from_pretrained', side_effect=load_processor), \
             contextlib.redirect_stdout(io.StringIO()):
            exec(compile(ast.Module(body=body, type_ignores=[]), str(notebook(part)), 'exec'), env)
        return env, calls

    def assert_restored(self, env, data):
        self.assertEqual(env['vision_dim'], data['config']['vision_dim'])
        self.assertEqual(env['language_dim'], data['config']['language_dim'])
        for key, value in data['projector_state_dict'].items():
            torch.testing.assert_close(env['vlm'].projector.state_dict()[key], value, rtol=0, atol=0)
        for key, value in data['language_model_state_dict'].items():
            torch.testing.assert_close(env['vlm'].language_model.state_dict()[key], value, rtol=0, atol=0)
        # 복원된 조합이 실제 순전파와 생성에서도 동작하는지 확인합니다.
        model = env['vlm'].eval()
        ids = torch.tensor([[3, 4, 5, 2]])
        pixels = torch.randn(1, 3, 16, 16)
        images = [pixels, pixels] if hasattr(model, 'encode_images') else pixels
        image_argument = 'pixel_values_list' if isinstance(images, list) else 'pixel_values'
        with torch.no_grad():
            result = model(**{image_argument: images}, input_ids=ids,
                           attention_mask=torch.ones_like(ids), labels=ids)
            self.assertTrue(torch.isfinite(result.loss))
            self.assertIsInstance(model.generate(images, prompt='word3',
                                  max_new_tokens=2, do_sample=False), str)

    def test_caption_checkpoint_config_and_all_weights_restored_in_parts_2_to_9(self):
        data, path = self.checkpoint()
        for part in range(2, 10):
            with self.subTest(part=part):
                env, calls = self.run_loading_cell(part)
                self.assert_restored(env, data)
                self.assertEqual(calls['vision'], [data['config']['vision_model_name']])
                self.assertEqual(calls['language'], [data['config']['lm_model_name']])
                self.assertEqual(calls['tokenizer'], [str(path.parent / 'tokenizer')])
                self.assertEqual(calls['processor'], [str(path.parent / 'image_processor')])

    def test_multitask_checkpoint_precedes_caption_in_parts_6_to_9(self):
        self.checkpoint()
        data, _ = self.checkpoint('mini-vlm-multitask', small=True)
        for part in range(6, 10):
            with self.subTest(part=part):
                env, _ = self.run_loading_cell(part)
                self.assert_restored(env, data)

    def test_missing_checkpoint_uses_each_parts_default_model(self):
        for part in range(2, 10):
            with self.subTest(part=part):
                env, calls = self.run_loading_cell(part)
                self.assertEqual(calls['vision'], ['google/vit-large-patch16-224' if part <= 3 else 'google/vit-base-patch16-224'])
                self.assertEqual(calls['language'], ['HuggingFaceTB/SmolLM-360M' if part <= 3 else 'HuggingFaceTB/SmolLM-135M'])

    def test_saved_expanded_vocabulary_is_restored(self):
        data, _ = self.checkpoint(vocab_size=34)
        env, _ = self.run_loading_cell(4)
        self.assertEqual(len(env['tokenizer']), 34)
        self.assert_restored(env, data)

    def test_invalid_checkpoint_dimensions_raise_instead_of_discarding_weights(self):
        data, path = self.checkpoint()
        data['config']['vision_dim'] = 99
        torch.save(data, path)
        with self.assertRaisesRegex(ValueError, '체크포인트 설정과 모델 차원'):
            self.run_loading_cell(4)

    def edit(self, response):
        env = definitions(9, ['ImageEditor', 'predict_edit_operation', 'edit_image_with_instruction'], namespace())
        model = SimpleNamespace(eval=lambda: None, generate=lambda *a, **kw: response)
        image = Image.new('RGB', (16, 16), (255, 0, 0))
        return env['edit_image_with_instruction'](model, image, 'make it grayscale',
            ViTImageProcessor(size={'height': 16, 'width': 16}), tokenizer(), torch.device('cpu'))

    def test_invalid_edit_responses_fall_back_to_a_valid_operation(self):
        responses = ['0.5', 'null', 'true', '[]', '"grayscale"', '{', 'Edit: 0.5', None,
                     '{}', '{"operation": []}', '{"operation": "unknown"}',
                     '{"operation":"brightness","params":null}',
                     '{"operation":"brightness","params":0.5}',
                     '{"operation":"brightness","params":[]}']
        for response in responses:
            with self.subTest(response=response):
                image, op = self.edit(response)
                self.assertEqual(op, {'operation': 'grayscale', 'params': {}})
                self.assertEqual(len(set(image.getpixel((0, 0)))), 1)

    def test_valid_edit_object_preserves_operation_and_parameters(self):
        for params in ({'angle': 90}, {}):
            with self.subTest(params=params):
                response = 'Edit: ' + json.dumps({'operation': 'rotate', 'params': params})
                image, op = self.edit(response)
                self.assertIsInstance(image, Image.Image)
                self.assertEqual(op, {'operation': 'rotate', 'params': params})

    def test_ocr_converts_grayscale_rgba_rgb_and_numpy_to_rgb(self):
        processor = ViTImageProcessor(size={'height': 16, 'width': 16})
        seen = []
        class OCRProcessor:
            def __call__(self, image, **kwargs):
                seen.append(image.mode)
                return processor(image, **kwargs)
            def batch_decode(self, ids, **kwargs):
                return tokenizer().batch_decode(ids, **kwargs)
        class OCRModel:
            def generate(self, pixel_values):
                assert pixel_values.shape == (1, 3, 16, 16)
                return torch.tensor([[3, 4, 2]])
        env = definitions(12, ['ocr_image_trocr'], namespace())
        env.update(trocr_processor=OCRProcessor(), trocr_model=OCRModel())
        for image in [Image.new('L', (16, 16)), Image.new('RGBA', (16, 16)),
                      Image.new('RGB', (16, 16)), np.zeros((16, 16), dtype=np.uint8)]:
            with self.subTest(type=type(image).__name__, mode=getattr(image, 'mode', None)):
                self.assertEqual(env['ocr_image_trocr'](image), 'word3 word4')
        self.assertEqual(seen, ['RGB'] * 4)

    def test_equation_ocr_uses_real_llama_embeddings_in_training_and_generation(self):
        env = definitions(12, ['EquationOCRVLM', 'predict_latex'], namespace())
        def factory(name):
            return ViTModel(vision_config(16)) if 'vit' in name else LlamaModel(language_config(16))
        with mock.patch.object(AutoModel, 'from_pretrained', side_effect=factory):
            model = env['EquationOCRVLM']()
        ids = torch.tensor([[3, 4, 5, 2]])
        result = model(pixel_values=torch.randn(1, 3, 16, 16), input_ids=ids,
                       attention_mask=torch.ones_like(ids), labels=ids)
        self.assertTrue(torch.isfinite(result['loss']))
        result['loss'].backward()
        self.assertIsNotNone(model.projector.weight.grad)
        self.assertIsNotNone(model.language_model.get_input_embeddings().weight.grad)
        prediction = env['predict_latex'](model, Image.new('RGB', (16, 16)), tokenizer(),
            ViTImageProcessor(size={'height': 16, 'width': 16}), max_length=2)
        self.assertIsInstance(prediction, str)

    def test_both_size_methods_count_serialized_int8_packed_weights(self):
        env = definitions(10, ['SimpleVLM', 'TinyVLM'], namespace())
        model = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 256)).eval()
        quantized = torch.ao.quantization.quantize_dynamic(copy.deepcopy(model), {nn.Linear}, dtype=torch.qint8)
        for name in ('SimpleVLM', 'TinyVLM'):
            for candidate in (model, quantized):
                with self.subTest(class_name=name, quantized=candidate is quantized):
                    path = self.output_dir / 'state.pt'
                    with path.open('wb') as stream:
                        torch.save(candidate.state_dict(), stream)
                    measured = env[name].get_model_size_mb(candidate)
                    self.assertEqual(measured, path.stat().st_size / 1024**2)
                    if candidate is quantized:
                        omitted = sum(x.weight().numel() * x.weight().element_size()
                                      for x in candidate.modules()
                                      if isinstance(x, torch.ao.nn.quantized.dynamic.Linear))
                        self.assertGreater(measured * 1024**2, omitted)
            self.assertLess(env[name].get_model_size_mb(quantized), env[name].get_model_size_mb(model))


if __name__ == '__main__':
    unittest.main()
