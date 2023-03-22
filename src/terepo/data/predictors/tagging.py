import json
import logging
import os
import re
from abc import abstractmethod
from dataclasses import dataclass
from time import time
from typing import Optional

import torch
from more_itertools import unzip, chunked
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from terepo.data.loaders import move_to_cuda
from terepo.data.loaders.base import DataProcessor
from terepo.data.predictors import register_predictor
from terepo.utils.misc import pad_tensors

logger = logging.getLogger(__name__)


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
    source: str
    target: Optional[str] = None


class Conll2014Processor(DataProcessor):
    """Processor for the TNEWS data set (CLUE version)."""

    def get_test_examples(self, data_dir):
        """See base class."""
        input_file = os.path.join(data_dir, "official-2014.combined.m2")
        with open(input_file) as fd:
            source_sentences, target_sentences = convert_m2_to_para(fd)
        return self._create_examples(source_sentences, "test")

    def dump(self, data, output_file):
        with open(output_file, 'w') as f:
            for item in data:
                f.write(item['inference'] + '\n')

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i % self.process_index != 0:
                continue
            guid = i
            examples.append(InputExample(guid=guid, source=line.strip()))
        return examples


class BEA2019Processor(DataProcessor):
    """Processor for the TNEWS data set (CLUE version)."""

    def get_test_examples(self, data_dir):
        """See base class."""
        input_file = os.path.join(data_dir, "ABCN.test.bea19.orig")
        with open(input_file) as fd:
            source_sentences = fd.readlines()
        return self._create_examples(source_sentences, "test")

    def dump(self, data, output_file):
        with open(output_file, 'w') as f:
            for item in data:
                f.write(item['inference'] + '\n')

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i % self.process_index != 0:
                continue
            guid = i
            examples.append(InputExample(guid=guid, source=line.strip()))
        return examples


class MiduCTCProcessor(DataProcessor):
    """Processor for the TNEWS data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "preliminary_train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "preliminary_dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "preliminary_b_test_source.json")), "test")

    def dump(self, data, output_file):
        with open(output_file, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i % self.process_index != 0:
                continue
            guid = line['id'] if 'id' in line else '"%s-%s" % (set_type, i)'
            examples.append(
                InputExample(guid=guid, source=line['source']))
        return examples


class MuCGECProcessor(DataProcessor):
    """Processor for the TNEWS data set (CLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "MuCGEC_dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        if os.path.isdir(data_dir):
            test_file = os.path.join(data_dir, "MuCGEC_test.txt")
        else:
            test_file = data_dir
        return self._create_examples(self._read_tsv(test_file), "test")

    @staticmethod
    def dump_intermediate(data, output_file):
        with open(output_file, 'w') as f:
            for item in data:
                assert item['inference'].strip()
                f.write(f"{item['id']}\t{item['inference']}\n")

    @staticmethod
    def dump(data, output_file):
        with open(output_file, 'w') as f:
            latency = []
            for item in data:
                assert item['inference'].strip()
                f.write(f"{item['id']}\t{item['source']}\t{item['inference']}\n")
                latency.append(item['latency'])
            logger.info(f"Average Latency: {sum(latency) / len(latency)}")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i % self.process_index != 0:
                continue
            if len(line) == 2:
                guid, source = line
            else:
                guid, source, target = line
            examples.append(
                InputExample(guid=guid, source=source))
        return examples


class FCGECProcessor(DataProcessor):
    """
    Processor for the TNEWS data set (CLUE version).
    {
        "id": ,  # The global id of the instance
        "sentence":, # The original sentence
        "error_flag": , # Whether sentence contains errors
        "error_type": , The error types of sentence
        "operation": , # [{The operation of the first reference}, {The operation of the second reference}, {...}],
        "external": Additional information(e.g., version)
    }
    """

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "FCGEC_valid.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        if os.path.isdir(data_dir):
            test_file = os.path.join(data_dir, "FCGEC_test.json")
        else:
            test_file = data_dir
        return self._create_examples(self._read_json(test_file), "test")

    @staticmethod
    def dump_intermediate(data, output_file):
        with open(output_file, 'w') as f:
            for item in data:
                assert item['inference'].strip()
                f.write(f"{item['id']}\t{item['inference']}\n")

    @staticmethod
    def dump(data, output_file):
        with open(output_file, 'w') as f:
            output = {}
            latency = []
            for item in data:
                assert item['inference'].strip()
                output[item['id']] = {
                    "error_flag": int(item['source'] != item['inference']),
                    "error_type": 'IWO',
                    "correction": item['inference']
                }
                latency.append(item['latency'])
            logger.info(f"Average Latency: {sum(latency) / len(latency)}")
            json.dump(output, f, ensure_ascii=False, indent=2)

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, (guid, v) in enumerate(data.items()):
            if i % self.process_index != 0:
                continue
            source = v["sentence"]
            examples.append(
                InputExample(guid=guid, source=source))
        return examples


class MCSCSetProcessor(DataProcessor):

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid_gold.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        if os.path.isdir(data_dir):
            test_file = os.path.join(data_dir, "test_gold.txt")
        else:
            test_file = data_dir
        return self._create_examples(self._read_tsv(test_file), "test")

    @staticmethod
    def dump_intermediate(data, output_file):
        with open(output_file, 'w') as f:
            for item in data:
                assert item['inference'].strip()
                f.write(f"{item['id']}\t{item['inference']}\n")

    @staticmethod
    def dump(data, output_file):
        with open(output_file, 'w') as f:
            latency = []
            for item in data:
                assert item['inference'].strip()
                f.write(f"{item['id']}\t{item['source']}\t{item['inference']}\n")
                latency.append(item['latency'])
            logger.info(f"Average Latency: {sum(latency) / len(latency)}")

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, (guid, source, target) in enumerate(data):
            if i % self.process_index != 0:
                continue
            examples.append(
                InputExample(guid=guid, source=source))
        return examples


class CLTCTrack1Processor(DataProcessor):
    """Processor for the TNEWS data set (CLUE version)."""

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "yaclc-csc_test.src")), "test")

    def dump(self, data, output_file):
        with open(output_file, 'w') as f:
            for item in data:
                line = item['id'] + ", "
                no_error = True
                for idx, (token, labels) in enumerate(zip(item['source'], item['labels'])):
                    idx = idx + 1
                    label = labels[0]
                    if label.startswith("$REPLACE_"):
                        no_error = False
                        line += (str(idx) + ", " + label.replace("$REPLACE_", "") + ", ")
                if no_error:
                    line += '0'
                line = line.strip(", ")
                f.write(line + '\n')

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i % self.process_index != 0:
                continue
            if len(line) == 2:
                guid, source = line
            else:
                guid, source, target = line
            examples.append(
                InputExample(guid=guid, source=source))
        return examples


class CLTCTrack2Processor(DataProcessor):
    """Processor for the TNEWS data set (CLUE version)."""

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "cged_test.txt")), "test")

    def dump(self, data, output_file):
        with open(output_file, 'w') as f:
            for line in pair2edits_char([item['id'] for item in data],
                                        [item['source'] for item in data],
                                        [item['inference'] for item in data]):
                f.write(line + '\n')

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i % self.process_index != 0:
                continue
            if len(line) == 2:
                guid, source = line
            else:
                guid, source, target = line
            examples.append(
                InputExample(guid=guid, source=source))
        return examples


processors = {
    "conll2014": Conll2014Processor,
    "bea2019": BEA2019Processor,
    "miductc": MiduCTCProcessor,
    'mucgec': MuCGECProcessor,
    'fcgec': FCGECProcessor,
    'mcscset': MCSCSetProcessor,
    'cltc-track1': CLTCTrack1Processor,
    'cltc-track2': CLTCTrack2Processor,
}


class TERepoBasePredictor:
    def __init__(self, tokenizer, feature_extractor, model, batch_size, task):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.model = model
        self.batch_size = batch_size
        self.task = task
        self.corr_index = self.feature_extractor.dtags_correct_id
        self.incorr_index = self.feature_extractor.dtags_incorrect_id
        self.num_detect_classes = self.feature_extractor.dtags_vocab_size
        self.num_labels_classes = self.feature_extractor.label_vocab_size

    @staticmethod
    def split_sentence(document: str, flag: str = "all", limit: int = 510):
        """
        Args:
            document:
            flag: Type:str, "all" 中英文标点分句，"zh" 中文标点分句，"en" 英文标点分句
            limit: 默认单句最大长度为510个字符
        Returns: Type:list
        """
        sent_list = []
        try:
            if flag == "zh":
                document = re.sub('(?P<quotation_mark>([。？！](?![”’"\'])))', r'\g<quotation_mark>\n', document)  # 单字符断句符
                document = re.sub('(?P<quotation_mark>([。？！])[”’"\'])', r'\g<quotation_mark>\n', document)  # 特殊引号
            elif flag == "en":
                document = re.sub('(?P<quotation_mark>([.?!](?![”’"\'])))', r'\g<quotation_mark>\n',
                                  document)  # 英文单字符断句符
                document = re.sub('(?P<quotation_mark>([?!.]["\']))', r'\g<quotation_mark>\n', document)  # 特殊引号
            else:
                document = re.sub('(?P<quotation_mark>([。？！….?!](?![”’"\'])))', r'\g<quotation_mark>\n',
                                  document)  # 单字符断句符
                document = re.sub('(?P<quotation_mark>(([。？！.!?]|…{1,2})[”’"\']))', r'\g<quotation_mark>\n',
                                  document)  # 特殊引号

            sent_list_ori = document.splitlines()
            for sent in sent_list_ori:
                sent = sent.strip()
                if not sent:
                    continue
                else:
                    while len(sent) > limit:
                        temp = sent[0:limit]
                        sent_list.append(temp)
                        sent = sent[limit:]
                    sent_list.append(sent)
        except:
            sent_list.clear()
            sent_list.append(document)
        return sent_list

    def _build_sample(self, source_sequence):
        source_tokens = self.feature_extractor.convert_sequence_to_tokens(source_sequence, self.tokenizer)
        input_ids, offsets = self.feature_extractor.convert_tokens_to_ids_with_offsets(source_tokens, self.tokenizer)

        attention_mask = [1] * len(input_ids)
        if len(input_ids) > 512:
            raise ValueError(f"Too long! {source_tokens}")

        original_mask = [1] * len(source_tokens)
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        offsets = torch.tensor(offsets)
        original_mask = torch.tensor(original_mask)
        return source_tokens, input_ids, attention_mask, offsets, original_mask

    def build_sample(self, example):
        rets = self._build_sample(example.source)
        return (example.guid, example.source) + rets

    def wrap_build_sample(self, example):
        try:
            return self.build_sample(example)
        except Exception as e:
            logger.warning([e, example])
            return [None] * 9

    def collate_fn(self, inputs):
        (idx, sources, source_tokens, input_ids, attention_mask, offsets, original_mask) = map(list,
                                                                                               unzip(
                                                                                                   [
                                                                                                       item
                                                                                                       for
                                                                                                       item
                                                                                                       in
                                                                                                       inputs
                                                                                                       if
                                                                                                       None not in item]))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attn_masks = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        num_offsets = [f.size(0) for f in offsets]
        offsets = pad_tensors(offsets, num_offsets, pad=-1)
        original_masks = pad_sequence(original_mask, batch_first=True, padding_value=0)

        batch = {
            'meta': (idx, sources, source_tokens),
            'input_ids': input_ids,
            'attention_mask': attn_masks,
            'offsets': offsets,
            'original_mask': original_masks,
        }
        return batch

    @abstractmethod
    def probabilities(self, intermediate_batch, intermediate_outputs, confidence_bias):
        raise NotImplementedError

    @abstractmethod
    def convert(self, ids, sources, input_words,
                class_probabilities_d, class_probabilities_labels, incorrect_probability, original_mask,
                min_error_probability):
        raise NotImplementedError

    def handle_examples(self, examples, iterations, min_error_probability, confidence_bias):

        ids, original_sources = None, None
        prediction_history = {}
        intermediate_examples = examples
        halted_gids = set()
        halted_predictions = {}
        for i in range(iterations):
            intermediate_batch = self.collate_fn(map(self.wrap_build_sample, intermediate_examples))
            intermediate_ids, intermediate_sources, intermediate_tokens = intermediate_batch.pop('meta')
            if i == 0:
                ids = intermediate_ids
                original_sources = intermediate_sources
            intermediate_batch = move_to_cuda(intermediate_batch, device=self.model.device)
            intermediate_outputs = self.model(**intermediate_batch, return_dict=True)

            class_probabilities_d, class_probabilities_labels, incorr_prob = self.probabilities(intermediate_batch,
                                                                                                intermediate_outputs,
                                                                                                confidence_bias)

            intermediate_examples = []
            for gid, source, prediction in self.convert(intermediate_ids,
                                                        intermediate_sources,
                                                        intermediate_tokens,
                                                        class_probabilities_d, class_probabilities_labels, incorr_prob,
                                                        intermediate_batch['original_mask'],
                                                        min_error_probability):
                if gid not in halted_gids:
                    prediction_history.setdefault(gid, [source])
                    if prediction in prediction_history[gid]:
                        halted_gids.add(gid)
                        halted_predictions[gid] = prediction
                    else:
                        intermediate_examples.append(InputExample(guid=gid, source=prediction))
                    prediction_history[gid].append(prediction)

            if len(intermediate_examples) == 0:
                break

        for i, (idx, source) in enumerate(zip(ids, original_sources)):
            if idx in halted_predictions:
                yield idx, source, halted_predictions[idx]
            else:
                yield idx, source, prediction_history[idx][-1]

    def forwards_with_iterations(self, examples, iterations, min_error_probability, confidence_bias):
        for examples in chunked(examples, self.batch_size):
            start = time()
            for idx, source, prediction in self.handle_examples(examples, iterations,
                                                                min_error_probability,
                                                                confidence_bias):
                yield idx, source, prediction, time() - start


class TERepoGECToRBasePredictor(TERepoBasePredictor):

    def probabilities(self, intermediate_batch, intermediate_outputs, confidence_bias):
        convert = self.convert
        class_probabilities_d = torch.softmax(intermediate_outputs.logits_d, dim=-1)
        class_probabilities_labels = torch.softmax(intermediate_outputs.logits_labels, dim=-1)
        error_probs = class_probabilities_d[:, :, self.incorr_index] * intermediate_batch['original_mask']
        incorr_prob = torch.max(error_probs, dim=-1)[0]
        class_probabilities_labels[:, :, self.feature_extractor.labels_keep_token_id] += confidence_bias
        return class_probabilities_d, class_probabilities_labels, incorr_prob

    def convert(self, ids, sources, input_words,
                class_probabilities_d, class_probabilities_labels, incorrect_probability, original_mask,
                min_error_probability):
        for gid, source, s_tokens, tag_prob, label_probs, err_prob, mask in zip(ids, sources, input_words,
                                                                                class_probabilities_d,
                                                                                class_probabilities_labels,
                                                                                incorrect_probability,
                                                                                original_mask):
            if err_prob < min_error_probability:
                yield gid, source, source
            else:
                label_ids = []
                for i, lp in enumerate(label_probs):
                    prob, idx = torch.max(lp, dim=0)
                    if prob < min_error_probability:
                        label_ids.append([self.feature_extractor.labels_keep_token_id])
                    else:
                        label_ids.append([idx.item()])

                labels_list = self.feature_extractor.convert_ids_to_labels_list(label_ids)
                p = self.feature_extractor.convert_labels_list_to_sentence(s_tokens, labels_list)
                p_sent = self.feature_extractor.convert_tokens_to_string(p, self.tokenizer)
                yield gid, source, p_sent


class TERepoIGECToRBasePredictor(TERepoBasePredictor):

    def probabilities(self, intermediate_batch, intermediate_outputs, confidence_bias):
        class_probabilities_d = [torch.softmax(logits_d, dim=-1) for logits_d in intermediate_outputs.logits_d]
        class_probabilities_labels = [torch.softmax(logits_labels, dim=-1) for logits_labels in
                                      intermediate_outputs.logits_labels]
        error_probs = [(1 - cp[:, :, self.corr_index]) * intermediate_batch['original_mask'] for cp in
                       class_probabilities_d]
        # incorr_prob = [torch.max(ep) for ep in error_probs]
        incorr_prob = torch.max(error_probs[0], dim=-1)[0]
        class_probabilities_labels[0][:, :, self.tokenizer.labels_keep_token_id] += confidence_bias
        return class_probabilities_d, class_probabilities_labels, incorr_prob

    def convert(self, ids, sources, input_words,
                class_probabilities_d, class_probabilities_labels, incorrect_probability, original_mask,
                min_error_probability):
        for i_ex, (gid, source, s_tokens, err_prob) in enumerate(zip(ids, sources, input_words, incorrect_probability)):
            if err_prob < min_error_probability:
                yield gid, source, source
            else:
                label_ids = [[] for _ in s_tokens]
                l_labels_list = [item[i_ex] for item in class_probabilities_labels]
                l_tags_list = [item[i_ex] for item in class_probabilities_d]
                for i_step, (logits_labels, logits_tags) in enumerate(zip(l_labels_list, l_tags_list)):
                    for i, token in enumerate(s_tokens):
                        logits_label = logits_labels[i]
                        idx = torch.argmax(logits_label).item()
                        label_ids[i].append(idx)

                        # logits_tag = logits_tags[i]
                        # d_idx = torch.argmax(logits_tag).item()
                        # if d_idx != self.tokenizer.dtags_correct_id:
                        #     logits_label = logits_labels[i]
                        #     idx = torch.argmax(logits_label).item()
                        #     label_ids[i].append(idx)
                        # else:
                        #     label_ids[i].append(self.tokenizer.labels_keep_token_id)
                labels_list = self.tokenizer.convert_ids_to_labels_list(label_ids)
                p = self.tokenizer.convert_labels_list_to_sentence(s_tokens, labels_list)
                p_sent = self.tokenizer.convert_tokens_to_string(p)
                yield gid, source, p_sent


@register_predictor("tagging", "gector")
class TERepoGECToRPredictor(TERepoGECToRBasePredictor):

    def __init__(self, tokenizer, feature_extractor, model, batch_size, task):
        super().__init__(tokenizer, feature_extractor, model, batch_size, task)

    def predict(self, input_file, output_file, iterations, min_error_probability, confidence_bias):
        print([iterations, min_error_probability, confidence_bias])
        processor = processors[self.task](1)
        pred_texts = []
        examples = processor.get_test_examples(input_file)
        for gid, source, prediction, latency in tqdm(self.forwards_with_iterations(examples,
                                                                                   iterations,
                                                                                   min_error_probability,
                                                                                   confidence_bias),
                                                     total=len(examples),
                                                     desc=input_file[-30:]):
            pred_texts.append({
                "id": gid,
                "source": source,
                "inference": prediction,
                "latency": latency
            })
        processor.dump(pred_texts, output_file)


@register_predictor("tagging", "igector")
class TERepoIGECToRPredictor(TERepoIGECToRBasePredictor):

    def __init__(self, tokenizer, model, batch_size, task):
        super().__init__(tokenizer, model, batch_size, task)

    def predict(self, input_file, output_file, iterations, min_error_probability, confidence_bias):
        print([iterations, min_error_probability, confidence_bias])
        processor = processors[self.task](1)
        pred_texts = []
        examples = processor.get_test_examples(input_file)
        for gid, source, prediction, latency in tqdm(self.forwards_with_iterations(examples,
                                                                                   iterations,
                                                                                   min_error_probability,
                                                                                   confidence_bias),
                                                     total=len(examples),
                                                     desc=input_file[-30:]):
            pred_texts.append({
                "id": gid,
                "source": source,
                "inference": prediction,
                "latency": latency
            })
        processor.dump(pred_texts, output_file)
