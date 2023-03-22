import json
import logging
import random

import torch
import webdataset as wds
from more_itertools import unzip
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import PretrainedConfig

from terepo.arguments import TERepoModelArguments, TERepoTrainingArguments
from terepo.arguments.gec import TERepoDataArguments
from terepo.data.loaders import register_loader, register_eval_loader
from terepo.data.loaders.tagging.base import TERepoBaseDataLoader
from terepo.utils.misc import pad_tensors

logger = logging.getLogger(__name__)


class LaserTaggerBaseDataLoader(TERepoBaseDataLoader):
    def __init__(self, tokenizer, feature_extractor, data_files,
                 model_args: TERepoModelArguments, training_args: TERepoTrainingArguments,
                 data_args: TERepoDataArguments, config: PretrainedConfig):
        super().__init__(tokenizer, feature_extractor, data_files, model_args, training_args, data_args, config)
        self.n_ctx = self.data_args.block_size
        self._tag_strategy = "keep_one"

    def _build_sample(self, source_sequence, target_sequence):
        source_tokens = self.tokenizer.convert_sequence_to_tokens(source_sequence)[:self.n_ctx]
        target_tokens = self.tokenizer.convert_sequence_to_tokens(target_sequence)[:self.n_ctx]

        input_ids, offsets = self.tokenizer.convert_tokens_to_ids_with_offsets(source_tokens)

        attention_mask = [1] * len(input_ids)
        if len(input_ids) > 512:
            raise ValueError(f"Too long! {source_tokens}")
        if len(input_ids) > 2 * self.n_ctx:
            raise ValueError(f"Careful! Cuda out of memory! {source_tokens}")

        edits = list(self.tokenizer.editor.convert_sequences_to_edits(source_tokens, target_tokens))
        labels_list = self.tokenizer.convert_edits_into_labels_list(source_tokens, edits)
        label_ids, detect_tags = self.tokenizer.convert_labels_list_to_ids(labels_list)
        original_mask = [1] * len(label_ids)

        input_ids = torch.tensor(input_ids)
        label_ids = torch.tensor(label_ids)
        attention_mask = torch.tensor(attention_mask)
        detect_tags = torch.tensor(detect_tags)
        offsets = torch.tensor(offsets)
        original_mask = torch.tensor(original_mask)
        return source_tokens, target_tokens, input_ids, attention_mask, offsets, original_mask, label_ids, detect_tags


@register_loader("tagging", "gector")
class LaserTaggerDataLoader(LaserTaggerBaseDataLoader):
    def __init__(self, tokenizer, feature_extractor, data_files,
                 model_args: TERepoModelArguments, training_args: TERepoTrainingArguments,
                 data_args: TERepoDataArguments, config: PretrainedConfig):
        super().__init__(tokenizer, feature_extractor, data_files, model_args, training_args, data_args, config)
        self.shards = self.get_shards(self.data_files)
        self.batch_size = self.training_args.per_device_train_batch_size

    def build_sample(self, example):
        example = example['json']

        source_sequence = example['source']
        target_sequence = example['target']

        _, _, input_ids, attention_mask, offsets, original_mask, label_ids, detect_tags = self._build_sample(
            source_sequence, target_sequence)

        return input_ids, attention_mask, offsets, original_mask, label_ids, detect_tags

    def wrap_build_sample(self, example):
        # return self.build_sample(example)
        try:
            return self.build_sample(example)
        except Exception as e:
            logger.warning([e, example])
            return [None] * 7

    @staticmethod
    def collate_fn(inputs):
        (input_ids, attention_mask, offsets, original_mask, label_ids, d_tags) = map(list,
                                                                                                   unzip(
                                                                                                       [item for item in
                                                                                                        inputs if
                                                                                                        None not in item]))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attn_masks = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        original_masks = pad_sequence(original_mask, batch_first=True, padding_value=0)

        label_ids = pad_sequence(label_ids, batch_first=True, padding_value=-100)
        d_tags = pad_sequence(d_tags, batch_first=True, padding_value=-100)

        num_offsets = [f.size(0) for f in offsets]
        offsets = pad_tensors(offsets, num_offsets, pad=-1)

        batch = {'input_ids': input_ids,
                 'attention_mask': attn_masks,
                 'offsets': offsets,
                 'original_mask': original_masks,
                 'd_tags': d_tags,
                 'labels': label_ids}
        return batch

    def __iter__(self):

        def check(item):
            data = json.loads(item['json'])
            source_sequence = data['source']
            target_sequence = data['target']

            if isinstance(source_sequence, list):
                return True

            if not source_sequence.strip() or not target_sequence.strip():
                return False

            if source_sequence.strip() == target_sequence.strip():
                if random.random() < self.data_args.use_correct_lines_prob:
                    return False
            return True

        assert len(self.shards) >= self.training_args.world_size  # guarantee at least one shard for each device
        logging.info(f"Constructing data loader for text editing: {len(self.shards)}")
        dataset = wds.WebDataset(self.shards, nodesplitter=wds.split_by_node).shuffle(1000).select(check).decode().map(
            self.wrap_build_sample)
        for batch in DataLoader(dataset, num_workers=self.training_args.train_num_workers, batch_size=self.batch_size,
                                collate_fn=self.collate_fn):
            yield batch
