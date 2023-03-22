# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for HCT."""

import collections
import os
from typing import List, Optional, Tuple, Union

from torch import TensorType
from transformers import BasicTokenizer, WordpieceTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import TruncationStrategy, BatchEncoding, PreTokenizedInputPair
from transformers.utils import logging, PaddingStrategy

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.txt",
    "pinyin_map_file": 'pinyin_data/zi_py.txt',
    "pinyin_vocab_file": 'pinyin_data/py_vocab.txt',
    "stroke_map_file": 'stroke_data/zi_sk.txt',
    "stroke_vocab_file": 'stroke_data/sk_vocab.txt',
    "confusion_same_pinyin_file": 'confusions/same_pinyin.txt',
    "confusion_similar_pinyin_file": 'confusions/simi_pinyin.txt',
    "confusion_same_stroke_file": 'confusions/same_stroke.txt'
}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def load_pydict(fpath):
    ans = {}
    for line in open(fpath, encoding='utf-8'):
        line = line.strip()  # .decode('utf8')
        tmps = line.split('\t')
        if len(tmps) != 2: continue
        ans[tmps[0]] = tmps[1]
    return ans


def load_pyvocab(fpath):
    ans = {'PAD': 0, 'UNK': 1}
    idx = 2
    for line in open(fpath, encoding='utf-8'):
        line = line.strip()  # .decode('utf8')
        if len(line) < 1: continue
        ans[line] = idx
        idx += 1
    return ans


def load_pinyin_confusion(in_file, vocab):
    confusion_datas = {}
    for line in open(in_file):
        tmps = line.strip().split('\t')
        if len(tmps) != 2:
            continue
        key = tmps[0]
        values = tmps[1].split()
        if len(key) != 1:
            continue
        all_ids = set()
        keyid = vocab.get(key, None)
        if keyid is None:
            continue
        for k in values:
            if vocab.get(k, None) is not None:
                all_ids.add(vocab[k])
        all_ids = list(all_ids)
        if len(all_ids) > 0:
            confusion_datas[keyid] = all_ids
    return confusion_datas


def load_stroke_confusion(in_file, vocab):
    confusion_datas = {}
    for line in open(in_file):
        tmps = line.strip().split(',')
        if len(tmps) < 2:
            continue
        values = tmps
        all_ids = set()
        for k in values:
            if k in vocab:
                all_ids.add(vocab[k])
        all_ids = list(all_ids)
        for k in all_ids:
            confusion_datas[k] = all_ids
    return confusion_datas


def load_rules(rule_path, mask='_', fmask='{}'):
    with open(rule_path, encoding='utf8') as f:
        rules = [''] + [l.strip().replace(mask, fmask) for l in f]
    rule_slot_cnts = [sum(int(y == fmask) for y in x.split()) for x in rules]
    return rules, rule_slot_cnts


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class HCTTokenizer(PreTrainedTokenizer):
    r"""
    Construct a BERT tokenizer. Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            Whether or not to do basic tokenization before WordPiece.
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
    """

    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
            self,
            vocab_file,
            rule_file,
            do_lower_case=True,
            do_basic_tokenize=True,
            never_split=None,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            tokenize_chinese_chars=True,
            strip_accents=None,
            **kwargs
    ):
        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = HCTTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)

        self.rules, self.rule_slot_cnts = load_rules(rule_file)
        self.idx2tag = self.load_tags()
        self.tag2idx = {tag: idx for idx, tag in enumerate(self.idx2tag)}

    @staticmethod
    def load_tags():
        return ["KEEP", "DELETE"]

    @staticmethod
    def upd_by_ptr(seq_width, curr_len_list, rule_seq, ptr):
        rem = sw = seq_width[ptr]
        cur_len, cur_rule = curr_len_list[ptr], rule_seq[ptr]
        return rem, sw, cur_len, cur_rule, ptr + 1

    def _split_to_wordpieces_span(self, tokens, label_action, label_start, label_end, seq_width, rule_seq):
        bert_tokens = []
        bert_label_action = []
        source_indices = []
        cum_num_list = []
        curr_len_list = []
        cum_num = 0
        src_start = orig_start = len(tokens)
        for i, token in enumerate(tokens):
            pieces = self.tokenizer.tokenize(token)
            if token == '|':
                src_start = len(bert_tokens) + 1
                orig_start = i + 1

            bert_label_action.extend([label_action[i]] * len(pieces))
            bert_tokens.extend(pieces)
            curr_len_list.append(len(pieces))
            cum_num_list.append(cum_num)
            cum_num += len(pieces) - 1

        if len(bert_tokens) > self.max_len:
            new_len = self.max_len - (len(bert_tokens) - src_start)
            source_indices = list(range(new_len, self.max_len))
            bert_tokens = bert_tokens[:new_len] + bert_tokens[src_start:]
        else:
            new_len = src_start
            source_indices = list(range(src_start, len(bert_tokens)))

        bert_label_start, bert_label_end = [], []
        bert_seq_width = []
        bert_rule = []
        cur_label_start, cur_label_end = [], []
        i = sum(seq_width[:orig_start])
        ptr = orig_start
        rem, sw, cur_len, cur_rule, ptr = self.upd_by_ptr(seq_width, curr_len_list, rule_seq, ptr)
        while i < len(label_start):
            if rem > 0:
                st, ed = label_start[i], label_end[i]
                i += 1
                start = st + cum_num_list[st] if st < len(cum_num_list) else st
                end = ed + cum_num_list[ed] + curr_len_list[ed] - 1 if ed < len(cum_num_list) else ed
                if start >= new_len or end >= new_len:
                    sw = max(1, sw - 1)
                    start, end = 0, 0
                zeros = [0] * (cur_len - 1)
                cur_label_start.append([start] + zeros)
                cur_label_end.append([end] + zeros)
                rem -= 1
            if rem == 0:
                bert_seq_width.extend([sw] * cur_len)
                bert_rule.extend([cur_rule] * cur_len)
                for tup_s, tup_e in zip(zip(*cur_label_start), zip(*cur_label_end)):
                    bert_label_start.append(tup_s)
                    bert_label_end.append(tup_e)
                cur_label_start.clear()
                cur_label_end.clear()
                if ptr < len(curr_len_list):
                    rem, sw, cur_len, cur_rule, ptr = self.upd_by_ptr(seq_width, curr_len_list, rule_seq, ptr)
                else:
                    assert (i == len(label_start))
        assert (len(bert_label_start) == len(bert_seq_width) == len(bert_rule))
        return bert_tokens, bert_label_action[
                            src_start:], bert_label_start, bert_label_end, bert_seq_width, bert_rule, source_indices

    def get_sens_tags(self, source, action_seq, span_seq, rule_seq):
        tokens = [self.tokenizer.cls_token] + source.strip().split(' ')
        start_seq, end_seq = zip(*[x.split('#') for x in span_seq])
        action_seq = [self.tag2idx.get(tag) for tag in ('DELETE',) + action_seq]
        start_seq, seq_width = self._split_multi_span(start_seq)
        end_seq, _ = self._split_multi_span(end_seq)
        rule_seq = [0] + list(map(self.to_int, rule_seq))

        bert_tokens, bert_label_action, bert_label_start, bert_label_end, bert_seq_width, bert_rule, src_indices = self._split_to_wordpieces_span(
            tokens, action_seq, start_seq, end_seq, seq_width, rule_seq)
        sentence = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        return sentence, bert_label_action, bert_label_start, bert_label_end, bert_seq_width, bert_rule, src_indices

    def _ids_to_sequence_list(self, max_len, vocab, char_vocab):
        ans = [[0] * max_len, [0] * max_len]  # PAD, UNK
        rpyvcab = {v: k for k, v in vocab.items()}
        for k in range(2, len(rpyvcab), 1):
            pystr = rpyvcab[k]
            seq = []
            for c in pystr:
                seq.append(char_vocab[c])
            seq = [0] * max_len + seq
            seq = seq[-max_len:]
            ans.append(seq)
        return ans

    def _convert_token_to_pinyin_id(self, token):
        py = self.zi_to_pinyin.get(token, None)
        if py is None:
            return self.pinyin_vocab['UNK']
        return self.pinyin_vocab.get(py, self.pinyin_vocab['UNK'])

    def _convert_token_id_to_pinyin_id(self, token_id):
        token = self._convert_id_to_token(token_id)
        return self._convert_token_to_pinyin_id(token)

    def _convert_token_id_to_pinyin_char_sequence(self, token_id):
        return self.pinyin_char_sequence_vocab[self._convert_token_id_to_pinyin_id(token_id)]

    def _convert_token_to_stroke_id(self, token):
        py = self.zi_to_stroke.get(token, None)
        if py is None:
            return self.stroke_vocab['UNK']
        return self.stroke_vocab.get(py, self.stroke_vocab['UNK'])

    def _convert_token_id_to_stroke_id(self, token_id):
        token = self._convert_id_to_token(token_id)
        return self._convert_token_to_stroke_id(token)

    def _convert_token_id_to_stroke_index_sequence(self, token_id):
        return self.stroke_sequence_vocab[self._convert_token_id_to_stroke_id(token_id)]

    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def prepare_for_model(
            self,
            ids: List[int],
            pair_ids: Optional[List[int]] = None,
            add_special_tokens: bool = True,
            padding: Union[bool, str, PaddingStrategy] = False,
            truncation: Union[bool, str, TruncationStrategy] = False,
            max_length: Optional[int] = None,
            stride: int = 0,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            prepend_batch_axis: bool = False,
            **kwargs
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens. Please Note, for *pair_ids*
        different than `None` and *truncation_strategy = longest_first* or `True`, it is not possible to return
        overflowing tokens. Such a combination of arguments will raise an error.

        Args:
            ids (`List[int]`):
                Tokenized input ids of the first sequence. Can be obtained from a string by chaining the `tokenize` and
                `convert_tokens_to_ids` methods.
            pair_ids (`List[int]`, *optional*):
                Tokenized input ids of the second sequence. Can be obtained from a string by chaining the `tokenize`
                and `convert_tokens_to_ids` methods.
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        if return_token_type_ids and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None."
            )

        if (
                return_overflowing_tokens
                and truncation_strategy == TruncationStrategy.LONGEST_FIRST
                and pair_ids is not None
        ):
            raise ValueError(
                "Not possible to return overflowing tokens for pair of sequences with the "
                "`longest_first`. Please select another truncation strategy than `longest_first`, "
                "for instance `only_second` or `only_first`."
            )

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        encoded_inputs = {}

        # Compute the total size of the returned encodings
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)

        # Truncation: Handle max sequence length
        overflowing_tokens = []
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )

        if return_overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])

        # Build output dictionary
        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

        # Check lengths
        self._eventual_warn_about_too_long_sequence(encoded_inputs["input_ids"], max_length, verbose)

        # Padding
        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding_strategy.value,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        pinyin_ids = []
        stroke_ids = []
        for token_id in encoded_inputs['input_ids']:
            pinyin_ids.append(self._convert_token_id_to_pinyin_char_sequence(token_id))
            stroke_ids.append(self._convert_token_id_to_stroke_index_sequence(token_id))

        encoded_inputs['pinyin_ids'] = pinyin_ids
        encoded_inputs['stroke_ids'] = stroke_ids

        batch_outputs = BatchEncoding(
            encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis
        )

        return batch_outputs

    def _batch_prepare_for_model(
            self,
            batch_ids_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int], None]]],
            add_special_tokens: bool = True,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
            max_length: Optional[int] = None,
            stride: int = 0,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[str] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_length: bool = False,
            verbose: bool = True,
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens

        Args:
            batch_ids_pairs: list of tokenized input ids or input ids pairs
        """

        batch_outputs = {}
        for first_ids, second_ids in batch_ids_pairs:
            outputs = self.prepare_for_model(
                first_ids,
                second_ids,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
                truncation=truncation_strategy.value,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=None,  # we pad in batch afterward
                return_attention_mask=False,  # we pad in batch afterward
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # We convert the whole batch to tensors at the end
                prepend_batch_axis=False,
                verbose=verbose,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        max_length = max([len(item) for item in batch_outputs['input_ids']])

        for i, item in enumerate(batch_outputs['pinyin_ids']):
            if len(item) < max_length:
                batch_outputs['pinyin_ids'][i] = [self._convert_token_id_to_pinyin_char_sequence(self.unk_token_id)] * (
                        max_length - len(item)) + item

        for i, item in enumerate(batch_outputs['stroke_ids']):
            if len(item) < max_length:
                batch_outputs['stroke_ids'][i] = [self._convert_token_id_to_stroke_index_sequence(
                    self.unk_token_id)] * (max_length - len(item)) + item

        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        return batch_outputs

    def _tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):

                # If the token is part of the never_split set
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    @staticmethod
    def gather_embed(inp_emb, inp_idx, inp_mask, dim=1):
        expand_dims = (-1,) * (dim + 1)
        ret = inp_emb.gather(dim, inp_idx.unsqueeze(2).expand(*expand_dims,
                                                              inp_emb.shape[-1])) * inp_mask.unsqueeze(2).to(
            inp_emb.dtype)
        return ret

    def embed_rules_slots(self, additional_special_tokens, fmask='{}'):
        # TODO: test whether slot embeddings are helpful
        ret, ret_ids = [], []
        mlen = 0
        for rule in self.rules:
            slot_id = 0
            ret.append([])
            ret_ids.append([])
            ret[-1].append(self.tokenizer.cls_token_id)
            j = len(ret[-1])
            for token in rule.split():
                if token == fmask:
                    ret_cur = [additional_special_tokens[slot_id]]
                    slot_id += 1
                    ret_ids[-1].append(j)
                else:
                    ret_cur = self.tokenizer.tokenize(token)
                ret[-1].extend(self.tokenizer.convert_tokens_to_ids(ret_cur))
                j += len(ret_cur)
            mlen = max(mlen, len(ret[-1]))
        for i, x in enumerate(ret):
            ret[i].extend([self.tokenizer.pad_token_id] * (mlen - len(x)))
            ret_ids[i].extend([0] * (self.max_sp_len - len(ret_ids[i])))
        ret = torch.tensor(ret, dtype=torch.long, device=self.device)
        ret_ids = torch.tensor(ret_ids, dtype=torch.long, device=self.device)

        ret = self.bert(ret, attention_mask=ret.ne(self.tokenizer.pad_token_id))[0]
        return Variable(ret[:, 0]), Variable(self.gather_embed(ret, ret_ids, ret_ids.ne(0)))

    def get_len_tokens(self, loss_mask, inp_idx):
        inp_len = loss_mask.long().sum()
        inp_tokens = self.tokenizer.convert_ids_to_tokens(inp_idx[:inp_len].tolist())
        return inp_len, inp_tokens

    def get_labels(self, label_action, label_rule, label_start, label_end, src_len, context_len, null_i=0):
        labels = []
        for idx in range(src_len):
            st, ed = null_i, null_i
            action = self.tags[label_action[idx]]
            rule = label_rule[idx]
            if rule > 0:
                st, ed = get_sp_strs(label_start[idx], label_end[idx], context_len)
                rule = label_rule[idx]
            labels.append(f'{action}|{st}#{ed}|{rule}')
        return labels

    def tags_to_string(self, source, labels, context=None, ignore_toks=set(['[SEP]', '[CLS]', '[UNK]', '|', '*'])):
        output_tokens = []
        for token, tag in zip(source, labels):
            action, added_phrase, rule_id = tag.split('|')
            rule_id = int(rule_id)
            slot_cnt = self.rule_slot_cnts[rule_id]
            starts, ends = added_phrase.split("#")
            starts, ends = map(lambda x: x.split(','), (starts, ends))
            sub_phrs = []
            for i, start in enumerate(starts):
                s_i, e_i = int(start), int(ends[i])
                add_phrase = ' '.join([s for s in context[s_i:e_i + 1] if s not in ignore_toks])
                if add_phrase:
                    sub_phrs.append(add_phrase)
                    if len(sub_phrs) == slot_cnt:
                        break
            sub_phrs.extend([''] * (slot_cnt - len(sub_phrs)))
            phr_toks = self.rules[rule_id].format(*sub_phrs).strip().split()
            output_tokens.extend(phr_toks)
            if action == 'KEEP':
                if token not in ignore_toks:
                    output_tokens.append(token)
            if len(output_tokens) > len(context):
                break

        if not output_tokens:
            output_tokens.append('*')
        elif len(output_tokens) > 1 and output_tokens[-1] == '*':
            output_tokens = output_tokens[:-1]
        return convert_tokens_to_string(output_tokens)

    def decode_into_string(self, source, label_action, label_rule, label_start, label_end, src_len, context=None,
                           context_len=0):
        if context is None:
            context = source
            context_len = src_len
        context = context[:context_len]
        labels = self.get_labels(label_action, label_rule, label_start, label_end, src_len, context_len)
        out_str = self.tags_to_string(source, labels, context=context)
        return out_str

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)
