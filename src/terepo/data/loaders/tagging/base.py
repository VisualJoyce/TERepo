import logging
from abc import abstractmethod

from transformers import PretrainedConfig

from terepo.arguments import TERepoModelArguments, TERepoTrainingArguments, TERepoDataArguments
from terepo.data.loaders.base import TERepoBaseDataLoader

logger = logging.getLogger(__name__)


class TERepoBaseDataLoader(TERepoBaseDataLoader):
    def __init__(self, tokenizer, feature_extractor, data_files,
                 model_args: TERepoModelArguments, training_args: TERepoTrainingArguments,
                 data_args: TERepoDataArguments, config: PretrainedConfig):
        super().__init__(tokenizer, feature_extractor, data_files, model_args, training_args, data_args, config)
        self.n_ctx = self.data_args.block_size
        self._tag_strategy = "keep_one"

    @abstractmethod
    def build_sample(self, example):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def collate_fn(inputs):
        raise NotImplementedError
