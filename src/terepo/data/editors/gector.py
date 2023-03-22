import os
import random
from difflib import SequenceMatcher
from typing import List

import Levenshtein
import numpy as np
from transformers.utils import logging

from terepo.data.editors import Operations
from terepo.data.editors.base import Alignment, load_vocab

logger = logging.get_logger(__name__)


def load_verb_form_dicts(verb_form_vocab_file):
    encode, decode = {}, {}
    with open(verb_form_vocab_file, encoding="utf-8") as f:
        for line in f:
            words, tags = line.split(":")
            word1, word2 = words.split("_")
            tag1, tag2 = tags.split("_")
            decode_key = f"{word1}_{tag1}_{tag2.strip()}"
            if decode_key not in decode:
                encode[words] = tags
                decode[decode_key] = word2
    return encode, decode


class GECToREditor:

    def __init__(self,
                 labels_vocab_file,
                 verb_form_vocab_file=None,
                 extra_labels_vocab_file=None,
                 labels_keep="$KEEP",
                 labels_delete="$DELETE",
                 labels_unknown="@@UNKNOWN@@",
                 use_delete_then_append_prob=0
                 ):
        self.labels_vocab = load_vocab(labels_vocab_file)
        if extra_labels_vocab_file:
            assert os.path.isfile(extra_labels_vocab_file)
            self.extra_labels_vocab = load_vocab(extra_labels_vocab_file, index_offset=len(self.labels_vocab))
            self.labels_vocab.update(self.extra_labels_vocab)
        else:
            self.extra_labels_vocab = {}

        self.labels_keep = labels_keep
        self.labels_delete = labels_delete
        self.labels_unknown = labels_unknown

        self._tag_strategy = "keep_one"
        self.seq_delimiters = {
            "tokens": " ",
            "labels": "SEPL|||SEPR",
            "operations": "SEPL__SEPR"
        }

        if verb_form_vocab_file and os.path.isfile(verb_form_vocab_file):
            self.use_verb_form = True
            self.encode_verb_dict, self.decode_verb_dict = load_verb_form_dicts(verb_form_vocab_file)
        else:
            self.use_verb_form = False

        self.use_delete_then_append_prob = use_delete_then_append_prob

    @property
    def tokens_delimiter(self):
        return self.seq_delimiters['tokens']

    @property
    def labels_delimiter(self):
        return self.seq_delimiters['labels']

    @property
    def operations_delimiter(self):
        return self.seq_delimiters['operations']

    def check_equal(self, source_token, target_token):
        if source_token == target_token:
            return self.labels_keep
        else:
            return None

    @staticmethod
    def check_split(source_token, target_tokens):
        if source_token.split("-") == target_tokens:
            return "$TRANSFORM_SPLIT_HYPHEN"
        else:
            return None

    def encode_verb_form(self, original_word, corrected_word):
        decoding_request = original_word + "_" + corrected_word
        decoding_response = self.encode_verb_dict.get(decoding_request, "").strip()
        if original_word and decoding_response:
            answer = decoding_response
        else:
            answer = None
        return answer

    def check_verb(self, source_token, target_token):
        encoding = self.encode_verb_form(source_token, target_token)
        if encoding:
            return f"$TRANSFORM_VERB_{encoding}"
        else:
            return None

    @staticmethod
    def is_cased(source_token, target_token):
        return source_token != target_token and source_token.lower() == target_token.lower()

    def check_casetype(self, source_token, target_token):
        if self.is_cased(source_token, target_token):
            if source_token.lower() == target_token:
                return "$TRANSFORM_CASE_LOWER"
            elif source_token.capitalize() == target_token:
                return "$TRANSFORM_CASE_CAPITAL"
            elif source_token.upper() == target_token:
                return "$TRANSFORM_CASE_UPPER"
            elif source_token[1:].capitalize() == target_token[1:] and source_token[0] == target_token[0]:
                return "$TRANSFORM_CASE_CAPITAL_1"
            elif source_token[:-1].upper() == target_token[:-1] and source_token[-1] == target_token[-1]:
                return "$TRANSFORM_CASE_UPPER_-1"

    @staticmethod
    def check_plural(source_token, target_token):
        if source_token.endswith("s") and source_token[:-1] == target_token:
            return "$TRANSFORM_AGREEMENT_SINGULAR"
        elif target_token.endswith("s") and source_token == target_token[:-1]:
            return "$TRANSFORM_AGREEMENT_PLURAL"
        else:
            return None

    def convert_labels_list_into_edits(self, labels_list: List[List[str]]):
        all_edits = []
        for i, labels in enumerate(labels_list):
            if labels[0] == self.labels_keep:
                continue
            else:
                edit = Operations(start=i, end=i + 1, operations=labels)
                all_edits.append(edit)
        return all_edits

    def apply_transformation(self, source_token, target_token):
        target_tokens = target_token.split()
        if len(target_tokens) > 1:
            # check split
            transform = self.check_split(source_token, target_tokens)
            if transform:
                return transform

        if self.use_verb_form:
            checks = [self.check_equal, self.check_casetype, self.check_verb, self.check_plural]
        else:
            checks = [self.check_equal, self.check_casetype, self.check_plural]

        for check in checks:
            transform = check(source_token, target_token)
            if transform:
                return transform
        return None

    def convert_alignments_into_edits(self, alignment, shift_idx):
        action, target_tokens, new_idx = alignment.action, alignment.target, alignment.shift
        source_token = action.replace("REPLACE_", "")

        # check if delete
        if not target_tokens:
            edit = Operations(start=shift_idx, end=1 + shift_idx, operations=self.labels_delete)
            return [edit]

        # check splits
        edits = []
        for i in range(1, len(target_tokens)):
            target_token = " ".join(target_tokens[:i + 1])
            transform = self.apply_transformation(source_token, target_token)
            if transform:
                edit = Operations(start=shift_idx, end=shift_idx + 1, operations=transform)
                edits.append(edit)
                target_tokens = target_tokens[i + 1:]
                for target in target_tokens:
                    edits.append(
                        Operations(start=shift_idx, end=shift_idx + 1, operations=f"$APPEND_{target}"))
                return edits

        transform_costs = []
        transforms = []
        for target_token in target_tokens:
            transform = self.apply_transformation(source_token, target_token)
            if transform:
                cost = 0
                transforms.append(transform)
            else:
                cost = Levenshtein.distance(source_token, target_token)
                transforms.append(None)
            transform_costs.append(cost)

        min_cost_idx = transform_costs.index(min(transform_costs))
        # append to the previous word
        for i in range(0, min_cost_idx):
            target = target_tokens[i]
            edits.append(Operations(start=shift_idx - 1, end=shift_idx, operations=f"$APPEND_{target}"))

        if random.random() < self.use_delete_then_append_prob:
            # replace/transform target word
            transform = transforms[min_cost_idx]
            target = transform if transform is not None else f"$DELETE"
            edits.append(Operations(start=shift_idx, end=1 + shift_idx, operations=target))

            # append to this word
            for i in range(min_cost_idx, len(target_tokens)):
                target = target_tokens[i]
                edits.append(Operations(start=shift_idx, end=1 + shift_idx, operations=f"$APPEND_{target}"))
        else:
            # replace/transform target word
            transform = transforms[min_cost_idx]
            target = transform if transform is not None else f"$REPLACE_{target_tokens[min_cost_idx]}"
            edits.append(Operations(start=shift_idx, end=1 + shift_idx, operations=target))

            # append to this word
            for i in range(min_cost_idx + 1, len(target_tokens)):
                target = target_tokens[i]
                edits.append(Operations(start=shift_idx, end=1 + shift_idx, operations=f"$APPEND_{target}"))
        return edits

    @staticmethod
    def replace_merge_transforms(tokens):
        if all(not x.startswith("$MERGE_") for x in tokens):
            return tokens
        target_tokens = tokens[:]
        allowed_range = (1, len(tokens) - 1)
        for i in range(len(tokens)):
            target_token = tokens[i]
            if target_token.startswith("$MERGE"):
                if target_token.startswith("$MERGE_SWAP") and i in allowed_range:
                    target_tokens[i - 1] = tokens[i + 1]
                    target_tokens[i + 1] = tokens[i - 1]
                    target_tokens[i: i + 1] = []
        target_line = " ".join(target_tokens)
        target_line = target_line.replace(" $MERGE_HYPHEN ", "-")
        target_line = target_line.replace(" $MERGE_SPACE ", "")
        return target_line.split()

    def perfect_align(self, t, T, insertions_allowed=0,
                      cost_function=Levenshtein.distance):
        # dp[i, j, k] is a minimal cost of matching first `i` tokens of `t` with
        # first `j` tokens of `T`, after making `k` insertions after last match of
        # token from `t`. In other words t[:i] aligned with T[:j].

        # Initialize with INFINITY (unknown)
        shape = (len(t) + 1, len(T) + 1, insertions_allowed + 1)
        dp = np.ones(shape, dtype=int) * int(1e9)
        come_from = np.ones(shape, dtype=int) * int(1e9)
        come_from_ins = np.ones(shape, dtype=int) * int(1e9)

        dp[0, 0, 0] = 0  # The only known starting point. Nothing matched to nothing.
        for i in range(len(t) + 1):  # Go inclusive
            for j in range(len(T) + 1):  # Go inclusive
                for q in range(insertions_allowed + 1):  # Go inclusive
                    if i < len(t):
                        # Given matched sequence of t[:i] and T[:j], match token
                        # t[i] with following tokens T[j:k].
                        for k in range(j, len(T) + 1):
                            transform = \
                                self.apply_transformation(t[i], '   '.join(T[j:k]))
                            if transform:
                                cost = 0
                            else:
                                cost = cost_function(t[i], '   '.join(T[j:k]))
                            current = dp[i, j, q] + cost
                            if dp[i + 1, k, 0] > current:
                                dp[i + 1, k, 0] = current
                                come_from[i + 1, k, 0] = j
                                come_from_ins[i + 1, k, 0] = q
                    if q < insertions_allowed:
                        # Given matched sequence of t[:i] and T[:j], create
                        # insertion with following tokens T[j:k].
                        for k in range(j, len(T) + 1):
                            cost = len('   '.join(T[j:k]))
                            current = dp[i, j, q] + cost
                            if dp[i, k, q + 1] > current:
                                dp[i, k, q + 1] = current
                                come_from[i, k, q + 1] = j
                                come_from_ins[i, k, q + 1] = q

        # Solution is in the dp[len(t), len(T), *]. Backtracking from there.
        alignment = []
        i = len(t)
        j = len(T)
        q = dp[i, j, :].argmin()
        while i > 0 or q > 0:
            is_insert = (come_from_ins[i, j, q] != q) and (q != 0)
            j, k, q = come_from[i, j, q], j, come_from_ins[i, j, q]
            if not is_insert:
                i -= 1

            if is_insert:
                alignment.append(Alignment('INSERT', T[j:k], (i, i)))
            else:
                alignment.append(Alignment(f'REPLACE_{t[i]}', T[j:k], (i, i + 1)))

        assert j == 0
        return list(reversed(alignment))

    def convert_sequences_to_edits(self, source_tokens, target_tokens):
        matcher = SequenceMatcher(None, source_tokens, target_tokens)
        for diff in matcher.get_opcodes():
            tag, i1, i2, j1, j2 = diff
            if tag == 'delete':
                # delete all words separatly
                for j in range(i2 - i1):
                    yield Operations(start=i1 + j, end=i1 + j + 1, operations=self.labels_delete)
            elif tag == 'insert':
                # append to the previous word
                for target_token in target_tokens[j1:j2]:
                    yield Operations(start=i1 - 1, end=i1, operations=f"$APPEND_{target_token}")
            elif tag == 'replace':
                # When the opcode is "replace", it's necessary to look into the segment to check the following:
                # 1. check merge: if replacing is resulting in directly merge the tokens
                # 2. check agreements: if there are plural or singular pairs
                # 3. check verb forms
                # 4. Other are $REPLACE
                edits = self.apply_merge_transformation(source_tokens[i1:i2], target_tokens[j1:j2], shift_idx=i1)
                if edits:
                    for edit in edits:
                        yield edit
                else:
                    # normalize alignments if need (make them singleton)
                    for alignment in self.perfect_align(source_tokens[i1:i2],
                                                        target_tokens[j1:j2],
                                                        insertions_allowed=0):
                        new_shift = alignment.shift[0]
                        for edit in self.convert_alignments_into_edits(alignment, shift_idx=i1 + new_shift):
                            yield edit

    @staticmethod
    def apply_merge_transformation(source_tokens, target_tokens, shift_idx):

        def check_merge():
            if "".join(source_tokens) == "".join(target_tokens):
                return "$MERGE_SPACE"
            elif "-".join(source_tokens) == "-".join(target_tokens):
                return "$MERGE_HYPHEN"
            else:
                return None

        def check_swap():
            if source_tokens == [x for x in reversed(target_tokens)]:
                return "$MERGE_SWAP"
            else:
                return None

        edits = []
        if len(source_tokens) > 1 and len(target_tokens) == 1:
            # check merge
            transform = check_merge()
            if transform:
                for i in range(len(source_tokens) - 1):
                    edits.append(
                        Operations(start=shift_idx + i, end=shift_idx + i + 1, operations=transform))
                return edits

        if len(source_tokens) == len(target_tokens) == 2:
            # check swap
            transform = check_swap()
            if transform:
                edits.append(Operations(start=shift_idx, end=shift_idx + 1, operations=transform))
        return edits

    def apply_reverse_transformation(self, source_token, transform):

        def convert_using_case(token, smart_action):
            if not smart_action.startswith("$TRANSFORM_CASE_"):
                return token
            if smart_action.endswith("LOWER"):
                return token.lower()
            elif smart_action.endswith("UPPER"):
                return token.upper()
            elif smart_action.endswith("CAPITAL"):
                return token.capitalize()
            elif smart_action.endswith("CAPITAL_1"):
                return token[0] + token[1:].capitalize()
            elif smart_action.endswith("UPPER_-1"):
                return token[:-1].upper() + token[-1]
            else:
                return token

        def convert_using_verb(token, smart_action):
            key_word = "$TRANSFORM_VERB_"
            if not smart_action.startswith(key_word):
                raise Exception(f"Unknown action type {smart_action}")
            encoding_part = f"{token}_{smart_action[len(key_word):]}"
            if encoding_part in self.decode_verb_dict:
                decoded_target_word = self.decode_verb_dict[encoding_part]
            else:
                logger.warning(f"{encoding_part} does not exist!")
                decoded_target_word = token
            return decoded_target_word

        def convert_using_split(token, smart_action):
            key_word = "$TRANSFORM_SPLIT"
            if not smart_action.startswith(key_word):
                raise Exception(f"Unknown action type {smart_action}")
            target_words = token.split("-")
            return " ".join(target_words)

        def convert_using_plural(token, smart_action):
            if smart_action.endswith("PLURAL"):
                return token + "s"
            elif smart_action.endswith("SINGULAR"):
                return token[:-1]
            else:
                raise Exception(f"Unknown action type {smart_action}")

        if transform.startswith("$TRANSFORM"):
            # deal with equal
            if transform == self.labels_keep:
                return source_token
            # deal with case
            if transform.startswith("$TRANSFORM_CASE"):
                return convert_using_case(source_token, transform)
            # deal with verb
            if transform.startswith("$TRANSFORM_VERB"):
                return convert_using_verb(source_token, transform)
            # deal with split
            if transform.startswith("$TRANSFORM_SPLIT"):
                return convert_using_split(source_token, transform)
            # deal with single/plural
            if transform.startswith("$TRANSFORM_AGREEMENT"):
                return convert_using_plural(source_token, transform)
            # raise exception if not find correct type
            raise Exception(f"Unknown action type {transform}")
        else:
            return source_token

    def convert_edits_to_sentence(self, source_tokens, edits):
        target_tokens = source_tokens[:]
        leveled_target_tokens = {}
        max_level = max([len(x.operations) for x in edits])
        for level in range(max_level):
            rest_edits = []
            shift_idx = 0
            for ops in edits:
                # try:
                # (start, end), label_list = edits
                label = ops.operations[0]
                target_pos = ops.start + shift_idx
                source_token = target_tokens[target_pos]
                if label == self.labels_delete:
                    del target_tokens[target_pos]
                    shift_idx -= 1
                elif label.startswith("$APPEND_"):
                    word = label.replace("$APPEND_", "")
                    target_tokens[target_pos + 1: target_pos + 1] = [word]
                    shift_idx += 1
                elif label.startswith("$REPLACE_"):
                    word = label.replace("$REPLACE_", "")
                    target_tokens[target_pos] = word
                elif label.startswith("$TRANSFORM"):
                    word = self.apply_reverse_transformation(source_token, label)
                    if word is None:
                        word = source_token
                    target_tokens[target_pos] = word
                elif label.startswith("$MERGE_"):
                    # apply merge only on last stage
                    if level == (max_level - 1):
                        target_tokens[target_pos + 1: target_pos + 1] = [label]
                        shift_idx += 1
                    else:
                        rest_edits.append(Operations(start=ops.start + shift_idx,
                                                     end=ops.end + shift_idx,
                                                     operations=label))

                rest_labels = ops.operations[1:]
                if rest_labels:
                    rest_edits.append(Operations(start=ops.start + shift_idx,
                                                 end=ops.end + shift_idx,
                                                 operations=rest_labels))
            # except IndexError as e:
            #     logger.warning(e)
            #     logger.warning(source_tokens)
            #     logger.warning(edits)

            leveled_tokens = target_tokens[:]
            # update next step
            edits = rest_edits[:]
            if level == (max_level - 1):
                leveled_tokens = self.replace_merge_transforms(leveled_tokens)
            leveled_target_tokens[level + 1] = leveled_tokens

        return leveled_target_tokens[max_level]
