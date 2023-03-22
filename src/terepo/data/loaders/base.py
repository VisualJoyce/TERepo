"""
Copyright (c) VisualJoyce.
Licensed under the MIT license.
"""
import abc
import csv
import glob
import json
import logging
import random
from dataclasses import dataclass
from typing import Optional, Any, Tuple, Union

from transformers import PretrainedConfig

from terepo.arguments import TERepoModelArguments, TERepoTrainingArguments, TERepoDataArguments

# from typing import Tuple, Any, OrderedDict, Optional, List, Union

logger = logging.getLogger(__name__)


class TERepoBaseDataLoader(metaclass=abc.ABCMeta):

    def __init__(self, tokenizer,
                 feature_extractor,
                 data_files,
                 model_args: TERepoModelArguments,
                 training_args: TERepoTrainingArguments,
                 data_args: Union[TERepoDataArguments, TERepoDataArguments],
                 config: PretrainedConfig):
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args
        self.config = config
        self.feature_extractor = feature_extractor
        self.data_files = data_files

    def get_shards(self, train_files):
        shards = []
        [shards.extend(glob.glob(f"{f}/**/*.tar", recursive=True)) for f in train_files.split(":")]
        if self.training_args.train_without_webdataset_sorted:
            random.shuffle(shards)
            return shards  # 所有tar文件路径
        else:
            return list(sorted(shards))

    def do_mask_and_convert(self, sent: list, tokenizer=None):
        # sent list of strings (tokens)
        # make masks & convert to word-ids
        tokenizer = self.tokenizer if tokenizer is None else tokenizer

        # TODO: make word-inputs to char ids
        # TODO:
        masked_sent = []
        truth = []

        # tokens = tokenizer.tokenize(sent)  # learned tokens
        tokens = sent
        # char-to-char; word-2-word (no extending)
        for token in tokens:
            if random.random() < self.data_args.mlm_probability:
                # do mask
                masked_sent.append(tokenizer._convert_token_to_id(tokenizer.mask_token))
                truth.append(tokenizer._convert_token_to_id(token))
            else:
                masked_sent.append(tokenizer._convert_token_to_id(token))
                truth.append(-100)
        return masked_sent, truth

    def truncate_seq_pair(self, tokens_a, tokens_b, max_num_tokens):
        """Truncates a pair of sequences to a maximum sequence length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_num_tokens:
                break

            trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
            assert len(trunc_tokens) >= 1

            # We want to sometimes truncate from the front and sometimes from the
            # back to add more randomness and avoid biases.
            if random.random() < 0.5:
                del trunc_tokens[0]
            else:
                trunc_tokens.pop()

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


# for text down stream
@dataclass
class InputExample:
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

    # def to_json_string(self):
    #     """Serializes this instance to a JSON string."""
    #     return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, process_index):
        self.process_index = process_index

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_jsonl(cls, input_file):
        """Reads a json list file."""
        with open(input_file, "r") as f:
            reader = f.readlines()
            lines = []
            for line in reader:
                lines.append(json.loads(line.strip()))
            return lines

    @classmethod
    def _read_json(cls, input_file):
        """Reads a json list file."""
        with open(input_file, "r") as f:
            return json.load(f)
