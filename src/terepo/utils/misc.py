"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Misc utilities
"""
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from allennlp.modules import TimeDistributed
from allennlp.nn.util import batched_span_select

from .logger import LOGGER


class NoOp(object):
    """ useful for distributed training No-Ops """

    def __getattr__(self, name):
        return self.noop

    def noop(self, *args, **kwargs):
        return


def parse_with_config(parser):
    """
    Parse from config files < command lines < system env
    """
    args = parser.parse_args()
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {arg[2:].split('=')[0] for arg in sys.argv[1:]
                         if arg.startswith('--')}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
            if os.getenv(k.upper()):
                new_v = os.getenv(k.upper())
                if isinstance(v, int):
                    new_v = int(new_v)
                if isinstance(v, float):
                    new_v = float(new_v)
                if isinstance(v, bool):
                    new_v = bool(new_v)
                setattr(args, k, new_v)
                LOGGER.info(f"Replaced {k} from system environment {k.upper()}: {new_v}.")

    # del args.config
    # args.model_config = os.path.join(args.pretrained_model_name_or_path, 'config.json')
    return args


VE_ENT2IDX = {
    'contradiction': 0,
    'entailment': 1,
    'neutral': 2
}

VE_IDX2ENT = {
    0: 'contradiction',
    1: 'entailment',
    2: 'neutral'
}


class Struct(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)


def set_dropout(model, drop_p):
    for name, module in model.named_modules():
        # we might want to tune dropout for smaller dataset
        if isinstance(module, torch.nn.Dropout):
            if module.p != drop_p:
                module.p = drop_p
                LOGGER.info(f'{name} set to {drop_p}')


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_parent_dir(cur_dir):
    return os.path.abspath(os.path.join(cur_dir, os.path.pardir))


def is_word(word):
    for item in list(word):
        if item not in "qwertyuiopasdfghjklzxcvbnm":
            return False
    return True


def _is_chinese_char(char):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    cp = ord(char)
    if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def pad_tensors(tensors, lens=None, pad=0):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].size(-1)
    dtype = tensors[0].dtype
    output = torch.zeros(bs, max_len, hid, dtype=dtype)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output


def mismatched_embeddings(sequence_output, offsets, sub_token_mode):
    # span_embeddings: (batch_size, num_orig_tokens, max_span_length, embedding_size)
    # span_mask: (batch_size, num_orig_tokens, max_span_length)
    span_embeddings, span_mask = batched_span_select(sequence_output.contiguous(), offsets)
    span_mask = span_mask.unsqueeze(-1)
    span_embeddings *= span_mask  # zero out paddings
    # If "sub_token_mode" is set to "first", return the first sub-token embedding
    if sub_token_mode == "first":
        # Select first sub-token embeddings from span embeddings
        # Shape: (batch_size, num_orig_tokens, embedding_size)
        orig_embeddings = span_embeddings[:, :, 0, :]

    # If "sub_token_mode" is set to "avg", return the average of embeddings of all sub-tokens of a word
    elif sub_token_mode == "avg":
        # Sum over embeddings of all sub-tokens of a word
        # Shape: (batch_size, num_orig_tokens, embedding_size)
        span_embeddings_sum = span_embeddings.sum(2)

        # Shape (batch_size, num_orig_tokens)
        span_embeddings_len = span_mask.sum(2)

        # Find the average of sub-tokens embeddings by dividing `span_embedding_sum` by `span_embedding_len`
        # Shape: (batch_size, num_orig_tokens, embedding_size)
        orig_embeddings = span_embeddings_sum / torch.clamp_min(span_embeddings_len, 1)

        # All the places where the span length is zero, write in zeros.
        orig_embeddings[(span_embeddings_len == 0).expand(orig_embeddings.shape)] = 0

    # If invalid "sub_token_mode" is provided, throw error
    else:
        raise ValueError(f"Do not recognise 'sub_token_mode' {sub_token_mode}")

    return orig_embeddings


MASK_LOGIT_CONST = 1e9


def logits_mask(inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_add = -MASK_LOGIT_CONST * (1. - mask)
    scores = inputs * mask + mask_add
    return scores


class MatchingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_matching_dropout = config.use_matching_dropout
        self.use_matching_layernorm = config.use_matching_layernorm
        if self.use_matching_dropout:
            self._dropout = TimeDistributed(torch.nn.Dropout(config.hidden_dropout_prob))
        else:
            self._dropout = nn.Identity()

        self._matching = nn.Linear(2 * config.hidden_size, config.hidden_size)
        if self.use_matching_layernorm:
            self._layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self._layernorm = nn.Identity()

    def forward(self, inputs: torch.Tensor, masks: torch.Tensor = None):
        query = self._dropout(inputs)
        key = self._dropout(inputs)

        scores = torch.matmul(query, key.permute(0, 2, 1))
        scores = scores * (1 / (query.shape[-1] ** (1 / 2)))
        scores = logits_mask(scores, masks)
        attention_probs = nn.functional.softmax(scores, dim=-1)
        context = torch.matmul(attention_probs, inputs)
        matching = self._matching(torch.cat([inputs, context], dim=-1))
        return self._layernorm(matching)


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor
