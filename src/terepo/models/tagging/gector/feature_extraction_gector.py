# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for ViT."""
import collections
import copy
import json
import os
from typing import Optional, List, Tuple, Union, Dict, Any

import numpy as np
import opencc
from nltk import word_tokenize
from transformers.dynamic_module_utils import custom_object_save
from transformers.feature_extraction_utils import FeatureExtractionMixin, PreTrainedFeatureExtractor
from transformers.utils import logging, FEATURE_EXTRACTOR_NAME, is_offline_mode, is_remote_url, download_url, \
    cached_file, extract_commit_hash

from terepo.data.editors import GECToREditor
from terepo.data.editors.base import load_vocab, Operations
from terepo.data.unicode import parse_to_segments, convert_tokens_to_string

logger = logging.get_logger(__name__)

CHINESE_CONVERTERS = {
    't2s': opencc.OpenCC(f't2s.json'),
    's2t': opencc.OpenCC(f's2t.json'),
}

VOCAB_FILES_NAMES = {
    "dtags_vocab_file": "dtags_vocab.txt",
    "labels_vocab_file": "labels_vocab.txt",
    "verb_form_vocab_file": "verb_form_vocab.txt",
}


class GECToRFeatureExtractor(FeatureExtractionMixin):
    model_input_names = ["pixel_values"]
    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
            self,
            dtags_vocab_file=VOCAB_FILES_NAMES['dtags_vocab_file'],
            labels_vocab_file=VOCAB_FILES_NAMES['labels_vocab_file'],
            extra_labels_vocab_file=None,
            use_start_token=True,
            use_cls_at_first=True,
            use_sep_at_last=True,
            verb_form_vocab_file=None,
            start_token="$START",
            start_token_id=None,
            dtags_correct="CORRECT",
            dtags_incorrect="INCORRECT",
            dtags_unknown="@@UNKNOWN@@",
            labels_keep="$KEEP",
            labels_delete="$DELETE",
            labels_unknown="@@UNKNOWN@@",
            tokenize_chinese_chars=True,
            strip_accents=None,
            chinese_converter_style=None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.use_start_token = use_start_token
        self.start_token = start_token
        self.start_token_id = start_token_id

        self.use_cls_at_first = use_cls_at_first
        self.use_sep_at_last = use_sep_at_last

        self._dtags_correct = dtags_correct
        self._dtags_incorrect = dtags_incorrect
        self._dtags_unknown = dtags_unknown
        self.dtags_vocab = load_vocab(dtags_vocab_file)

        self.editor = GECToREditor(
            labels_vocab_file,
            verb_form_vocab_file,
            extra_labels_vocab_file,
            labels_keep=labels_keep,
            labels_delete=labels_delete,
            labels_unknown=labels_unknown
        )
        self.ids_to_labels = collections.OrderedDict([(ids, tok) for tok, ids in self.editor.labels_vocab.items()])
        self.chinese_converter_style = chinese_converter_style

    @property
    def dtags_vocab_size(self):
        return len(self.dtags_vocab)

    @property
    def dtags_correct(self) -> Optional[str]:
        """
        `str`: Unknown token. Log an error if used while not having been set.
        """
        if self._dtags_correct is None and self.verbose:
            logger.error("Using unk_token, but it is not set yet.")
            return None
        return str(self._dtags_correct)

    @property
    def dtags_correct_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the beginning of sentence token in the vocabulary. Returns `None` if the token has not
        been set.
        """
        if self._dtags_correct is None:
            return None
        return self.dtags_vocab.get(self._dtags_correct)

    @property
    def dtags_incorrect(self) -> Optional[str]:
        """
        `str`: Unknown token. Log an error if used while not having been set.
        """
        if self._dtags_incorrect is None and self.verbose:
            logger.error("Using unk_token, but it is not set yet.")
            return None
        return str(self._dtags_incorrect)

    @property
    def dtags_incorrect_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the beginning of sentence token in the vocabulary. Returns `None` if the token has not
        been set.
        """
        if self._dtags_incorrect is None:
            return None
        return self.dtags_vocab.get(self._dtags_incorrect)

    @property
    def dtags_unknown(self) -> Optional[str]:
        """
        `str`: Unknown token. Log an error if used while not having been set.
        """
        if self._dtags_unknown is None and self.verbose:
            logger.error("Using unk_token, but it is not set yet.")
            return None
        return str(self._dtags_unknown)

    @property
    def dtags_unknown_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the beginning of sentence token in the vocabulary. Returns `None` if the token has not
        been set.
        """
        if self._dtags_unknown is None:
            return None
        return self.dtags_vocab.get(self._dtags_unknown)

    @property
    def label_vocab_size(self):
        return len(self.editor.labels_vocab)

    @property
    def extra_label_vocab_size(self):
        return len(self.editor.extra_labels_vocab)

    @property
    def labels_keep_token_id(self) -> Optional[int]:
        return self.editor.labels_vocab[self.editor.labels_keep]

    @property
    def labels_unknown_token_id(self) -> Optional[int]:
        return self.editor.labels_vocab[self.editor.labels_unknown]

    def convert_sequence_to_tokens(self, sequence, tokenizer):
        if not isinstance(sequence, list):
            sequence = [sequence]

        tokens = []
        for i, seq in enumerate(sequence):
            if i > 0:
                tokens.append(tokenizer.sep_token)

            segments, zh_idx_list, candidates = parse_to_segments(seq)
            for span, text, is_zh in segments:
                if is_zh:
                    if hasattr(self, 'chinese_converter_style') and self.chinese_converter_style in CHINESE_CONVERTERS:
                        text = CHINESE_CONVERTERS[self.chinese_converter_style].convert(text)
                    tokens.extend(list(text))
                else:
                    # tokens.extend(text.split())
                    tokens.extend(word_tokenize(text))

            if i == 0:
                if self.use_start_token:
                    tokens = [self.start_token] + tokens
                if self.use_cls_at_first:
                    tokens = [tokenizer.cls_token] + tokens
        return tokens

    def convert_tokens_to_ids_with_offsets(self, string_tokens: List[str], tokenizer):
        max_bpe_pieces = 5
        input_ids = []
        offsets: List[Optional[Tuple[int, int]]] = []
        for token_string in string_tokens:
            if token_string == self.start_token and self.start_token_id is not None:
                wp_ids = [self.start_token_id]
            else:
                wordpieces = tokenizer.encode_plus(
                    f' {token_string}', # add space to token to fix roberta tokenization issue
                    add_special_tokens=False,
                    return_tensors=None,
                    return_offsets_mapping=False,
                    return_attention_mask=False,
                )
                wp_ids = wordpieces["input_ids"]

            if len(wp_ids) > 0:
                if len(wp_ids) > max_bpe_pieces:
                    wp_ids = wp_ids[:max_bpe_pieces]
                offsets.append((len(input_ids), len(input_ids) + len(wp_ids) - 1))
            else:
                offsets.append(None)
            input_ids.extend(wp_ids)

        if self.use_sep_at_last:
            input_ids = input_ids + [tokenizer.sep_token_id]
        # offsets = self._increment_offsets(offsets, 0)
        offsets = [x if x is not None else (-1, -1) for x in offsets]
        return input_ids, offsets

    def convert_labels_list_to_ids(self, labels_list: List[List[str]]):
        label_ids = [self.editor.labels_vocab.get(x[0], self.labels_unknown_token_id) for x in labels_list]
        detect_tags = [self.dtags_correct if label == [self.editor.labels_keep] else self.dtags_incorrect for
                       label in labels_list]
        detect_ids = [self.dtags_vocab.get(x, 0) for x in detect_tags]
        return label_ids, detect_ids

    def convert_ids_to_labels_list(self, label_ids: List[List[int]]) -> List[List[str]]:
        labels = [[self.ids_to_labels.get(xx, self.editor.labels_keep) for xx in x] for x in label_ids]
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
            labels_list = [[self.editor.labels_keep] for _ in range(total_labels)]
        else:
            for i in range(total_labels):
                edit_operations = [x.operations[0] for x in edits if x.start == i and x.end == i + 1]
                if not edit_operations:
                    labels_list.append([self.editor.labels_keep])
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

    def convert_tokens_to_string(self, tokens, tokenizer):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").replace(tokenizer.cls_token, "").replace(self.start_token,
                                                                                                  "").strip()
        if tokenizer.sep_token in out_string:
            tokens = []
            for i, seq in enumerate(out_string.split(tokenizer.sep_token)):
                if i > 0:
                    tokens.append(tokenizer.sep_token)
                tokens.extend(seq.split())
        else:
            tokens = out_string.split()
        return convert_tokens_to_string(tokens) if tokens else ""

    @classmethod
    def from_pretrained(
            cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> PreTrainedFeatureExtractor:
        r"""
        Instantiate a type of [`~feature_extraction_utils.FeatureExtractionMixin`] from a feature extractor, *e.g.* a
        derived class of [`SequenceFeatureExtractor`].

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or
                  namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
                - a path to a *directory* containing a feature extractor file saved using the
                  [`~feature_extraction_utils.FeatureExtractionMixin.save_pretrained`] method, e.g.,
                  `./my_model_directory/`.
                - a path or url to a saved feature extractor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model feature extractor should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the feature extractor files and override the cached versions
                if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received file. Attempts to resume the download if such a file
                exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            use_auth_token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.


                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>".

                </Tip>

            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final feature extractor object. If `True`, then this
                functions returns a `Tuple(feature_extractor, unused_kwargs)` where *unused_kwargs* is a dictionary
                consisting of the key/value pairs whose keys are not feature extractor attributes: i.e., the part of
                `kwargs` which has not been used to update `feature_extractor` and is otherwise ignored.
            kwargs (`Dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are feature extractor attributes will be used to override the
                loaded values. Behavior concerning key/value pairs whose keys are *not* feature extractor attributes is
                controlled by the `return_unused_kwargs` keyword parameter.

        Returns:
            A feature extractor of type [`~feature_extraction_utils.FeatureExtractionMixin`].

        Examples:

        ```python
        # We can't instantiate directly the base class *FeatureExtractionMixin* nor *SequenceFeatureExtractor* so let's show the examples on a
        # derived class: *Wav2Vec2FeatureExtractor*
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )  # Download feature_extraction_config from huggingface.co and cache.
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "./test/saved_model/"
        )  # E.g. feature_extractor (or model) was saved using *save_pretrained('./test/saved_model/')*
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("./test/saved_model/preprocessor_config.json")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base-960h", return_attention_mask=False, foo=False
        )
        assert feature_extractor.return_attention_mask is False
        feature_extractor, unused_kwargs = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base-960h", return_attention_mask=False, foo=False, return_unused_kwargs=True
        )
        assert feature_extractor.return_attention_mask is False
        assert unused_kwargs == {"foo": False}
        ```"""
        feature_extractor_dict, kwargs = cls.get_feature_extractor_dict(pretrained_model_name_or_path, **kwargs)

        return cls.from_dict(feature_extractor_dict, **kwargs)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save a feature_extractor object to the directory `save_directory`, so that it can be re-loaded using the
        [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the feature extractor JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs:
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id, token = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_feature_extractor_file = os.path.join(save_directory, FEATURE_EXTRACTOR_NAME)

        self.to_json_file(output_feature_extractor_file)
        logger.info(f"Feature extractor saved in {output_feature_extractor_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory, repo_id, files_timestamps, commit_message=commit_message, token=token
            )

        return [output_feature_extractor_file]

    @classmethod
    def get_feature_extractor_dict(
            cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        feature extractor of type [`~feature_extraction_utils.FeatureExtractionMixin`] using `from_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the feature extractor object.
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)

        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        commit_hash = kwargs.pop("_commit_hash", None)

        user_agent = {"file_type": "feature extractor", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        single_file_id = None
        if os.path.isdir(pretrained_model_name_or_path):
            feature_extractor_file = os.path.join(pretrained_model_name_or_path, FEATURE_EXTRACTOR_NAME)
        if os.path.isfile(pretrained_model_name_or_path):
            resolved_feature_extractor_file = pretrained_model_name_or_path
            is_local = True
        elif is_remote_url(pretrained_model_name_or_path):
            feature_extractor_file = pretrained_model_name_or_path
            resolved_feature_extractor_file = download_url(pretrained_model_name_or_path)
        else:
            feature_extractor_file = FEATURE_EXTRACTOR_NAME
            try:
                # Load from local folder or from cache or download from model Hub and cache
                resolved_feature_extractor_file = cached_file(
                    pretrained_model_name_or_path,
                    feature_extractor_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    user_agent=user_agent,
                    revision=revision,
                )
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load feature extractor for '{pretrained_model_name_or_path}'. If you were trying to load"
                    " it from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a {FEATURE_EXTRACTOR_NAME} file"
                )

        try:
            # Load feature_extractor dict
            with open(resolved_feature_extractor_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            feature_extractor_dict = json.loads(text)

        except json.JSONDecodeError:
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_feature_extractor_file}' is not a valid JSON file."
            )

        if is_local:
            logger.info(f"loading configuration file {resolved_feature_extractor_file}")
        else:
            logger.info(
                f"loading configuration file {feature_extractor_file} from cache at {resolved_feature_extractor_file}"
            )

        vocab_files = {**cls.vocab_files_names}
        # Get files from url, cache, or disk depending on the case
        resolved_vocab_files = {}
        unresolved_files = []
        for file_id, file_path in vocab_files.items():
            if file_path is None:
                resolved_vocab_files[file_id] = None
            elif single_file_id == file_id:
                if os.path.isfile(file_path):
                    resolved_vocab_files[file_id] = file_path
                elif is_remote_url(file_path):
                    resolved_vocab_files[file_id] = download_url(file_path, proxies=proxies)
            else:
                resolved_vocab_files[file_id] = cached_file(
                    pretrained_model_name_or_path,
                    file_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                    _commit_hash=commit_hash,
                )
                commit_hash = extract_commit_hash(resolved_vocab_files[file_id], commit_hash)

        feature_extractor_dict.update(resolved_vocab_files)
        return feature_extractor_dict, kwargs

    @classmethod
    def from_dict(cls, feature_extractor_dict: Dict[str, Any], **kwargs) -> PreTrainedFeatureExtractor:
        """
        Instantiates a type of [`~feature_extraction_utils.FeatureExtractionMixin`] from a Python dictionary of
        parameters.

        Args:
            feature_extractor_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the feature extractor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~feature_extraction_utils.FeatureExtractionMixin.to_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the feature extractor object.

        Returns:
            [`~feature_extraction_utils.FeatureExtractionMixin`]: The feature extractor object instantiated from those
            parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        feature_extractor = cls(**feature_extractor_dict)

        # Update feature_extractor with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(feature_extractor, key):
                setattr(feature_extractor, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        # logger.info(f"Feature extractor {feature_extractor}")
        if return_unused_kwargs:
            return feature_extractor, kwargs
        else:
            return feature_extractor

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this feature extractor instance.
        """
        output = copy.deepcopy(self.__dict__)
        output["feature_extractor_type"] = self.__class__.__name__

        return output

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]) -> PreTrainedFeatureExtractor:
        """
        Instantiates a feature extractor of type [`~feature_extraction_utils.FeatureExtractionMixin`] from the path to
        a JSON file of parameters.

        Args:
            json_file (`str` or `os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            A feature extractor of type [`~feature_extraction_utils.FeatureExtractionMixin`]: The feature_extractor
            object instantiated from that JSON file.
        """
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        feature_extractor_dict = json.loads(text)
        return cls(**feature_extractor_dict)

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this feature_extractor instance in JSON format.
        """
        dictionary = self.to_dict()

        for key, value in dictionary.items():
            if isinstance(value, np.ndarray):
                dictionary[key] = value.tolist()

        # make sure private name "_processor_class" is correctly
        # saved as "processor_class"
        _processor_class = dictionary.pop("_processor_class", None)
        if _processor_class is not None:
            dictionary["processor_class"] = _processor_class

        dictionary.pop('editor')

        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this feature_extractor instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())
