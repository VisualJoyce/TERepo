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
"""Tokenization classes for Bert."""
import collections
from typing import Optional, List, Tuple

import opencc
from transformers import BertTokenizer
from transformers.utils import logging

from terepo.data.editors import Operations, LaserTaggerEditor
from terepo.data.unicode import convert_tokens_to_string, parse_to_segments

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.txt",
    "dtags_vocab_file": "dtags_vocab.txt",
    "labels_vocab_file": "labels_vocab.txt",
    "verb_form_vocab_file": "verb_form_vocab.txt",
}


class LaserTaggerTokenizer(BertTokenizer):
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
            labels_vocab_file,
            use_start_token=True,
            use_cls_at_first=True,
            use_sep_at_last=True,
            verb_form_vocab_file=None,
            do_lower_case=True,
            do_basic_tokenize=True,
            never_split=None,
            start_token="$START",
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            labels_keep="$KEEP",
            labels_delete="$DELETE",
            labels_unknown="@@UNKNOWN@@",
            tokenize_chinese_chars=True,
            strip_accents=None,
            chinese_converter_style=None,
            **kwargs
    ):
        super().__init__(
            vocab_file=vocab_file,
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

        self.use_start_token = use_start_token
        self.start_token = start_token

        self.use_cls_at_first = use_cls_at_first
        self.use_sep_at_last = use_sep_at_last

        self.editor = LaserTaggerEditor(
            labels_vocab_file,
            verb_form_vocab_file,
            labels_keep=labels_keep,
            labels_delete=labels_delete
        )
        # 纠错标签
        self.labels_vocab = self.editor.labels_vocab
        self.ids_to_labels = collections.OrderedDict([(ids, tok) for tok, ids in self.labels_vocab.items()])
        self.labels_keep = labels_keep
        self.labels_delete = labels_delete
        self.labels_unknown = labels_unknown

        if chinese_converter_style is not None:
            self.chinese_converter = opencc.OpenCC(f'{chinese_converter_style}.json')

    @property
    def labels_vocab_size(self):
        return len(self.labels_vocab)

    @property
    def labels_keep_token_id(self) -> Optional[int]:
        return self.labels_vocab[self.labels_keep]

    @property
    def labels_unknown_token_id(self) -> Optional[int]:
        return self.labels_vocab[self.labels_unknown]

    def convert_sequence_to_tokens(self, sequence):
        segments, zh_idx_list, candidates = parse_to_segments(sequence)
        tokens = []
        for span, text, is_zh in segments:
            if is_zh:
                if hasattr(self, 'chinese_converter'):
                    text = self.chinese_converter.convert(text)
                tokens.extend(list(text))
            else:
                tokens.extend(text.split())
        if self.use_start_token:
            tokens = [self.start_token] + tokens
        if self.use_cls_at_first:
            tokens = [self.cls_token] + tokens
        return tokens

    def convert_tokens_to_ids_with_offsets(self, string_tokens: List[str]):
        input_ids = []
        offsets: List[Optional[Tuple[int, int]]] = []
        for token_string in string_tokens:
            wordpieces = self.encode_plus(
                token_string,
                add_special_tokens=False,
                return_tensors=None,
                return_offsets_mapping=False,
                return_attention_mask=False,
            )
            wp_ids = wordpieces["input_ids"]
            if len(wp_ids) > 0:
                offsets.append((len(input_ids), len(input_ids) + len(wp_ids) - 1))
            else:
                offsets.append(None)
            input_ids.extend(wp_ids)

        if self.use_sep_at_last:
            input_ids = input_ids + [self.sep_token_id]
        # offsets = self._increment_offsets(offsets, 0)
        offsets = [x if x is not None else (-1, -1) for x in offsets]
        return input_ids, offsets

    def convert_labels_list_to_ids(self, labels_list: List[List[str]]):
        # if self.bert4gec_diff._tag_strategy == "keep_one":
        #     # get only first candidates for r_tags in right and the last for left
        #     label_ids = [self.bert4gec_diff.labels_vocab.get(x[0]) for x in labels]
        # elif self.bert4gec_diff._tag_strategy == "merge_all":
        #     # consider phrases as a words
        #     pass
        # else:
        #     raise Exception("Incorrect tag strategy")
        label_ids = [self.labels_vocab.get(x[0], self.labels_unknown_token_id) for x in labels_list]
        detect_tags = [self.dtags_correct if label == [self.labels_keep] else self.dtags_incorrect for
                       label in labels_list]
        detect_ids = [self.dtags_vocab.get(x, 0) for x in detect_tags]
        return label_ids, detect_ids

    def convert_ids_to_labels_list(self, label_ids: List[int]) -> List[List[str]]:
        labels = [[self.ids_to_labels.get(x, self.labels_keep)] for x in label_ids]
        return labels

    def convert_edits_into_labels_list(self, source_tokens, edits) -> List[List[str]]:
        # make sure that edits are flat
        flat_edits = []
        for edit in edits:
            for operation in edit.operations:
                flat_edits.append(Operations(start=edit.start, end=edit.end, operations=operation))

        edits = flat_edits[:]
        labels_list = []
        total_labels = len(source_tokens)
        if not edits:
            labels_list = [[self.labels_keep] for _ in range(total_labels)]
        else:
            for i in range(total_labels):
                edit_operations = [x.operations[0] for x in edits if x.start == i and x.end == i + 1]
                if not edit_operations:
                    labels_list.append([self.labels_keep])
                else:
                    labels_list.append(edit_operations)
        return labels_list

    def convert_labels_list_to_sentence(self, source_tokens, labels_list):
        relevant_edits = self.editor.convert_labels_list_into_edits(labels_list)
        target_tokens = source_tokens[:]
        if not relevant_edits:
            return target_tokens
        else:
            return self.editor.convert_edits_to_sentence(source_tokens, relevant_edits)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").replace(self.cls_token, "").replace(self.start_token,
                                                                                             "").strip()
        tokens = out_string.split()
        return convert_tokens_to_string(tokens) if tokens else ""
