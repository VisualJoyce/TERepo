"""
Copyright (c) VisualJoyce.
Licensed under the MIT license.
"""
import logging

import numpy as np
from transformers import PretrainedConfig

from terepo.arguments import TERepoModelArguments, TERepoTrainingArguments, TERepoDataArguments

logger = logging.getLogger(__name__)


class TERepoBaseEvaluator:
    def __init__(self, tokenizer, feature_extractor,
                 model_args: TERepoModelArguments,
                 training_args: TERepoTrainingArguments,
                 data_args: TERepoDataArguments,
                 config: PretrainedConfig):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args
        self.config = config

    @staticmethod
    def f1(precision, recall):
        if precision + recall == 0:
            return 0
        return round(2 * precision * recall / (precision + recall), 4)

    @staticmethod
    def retrieval(query_output_list, value_output_list, targets):
        query_num, value_num = len(query_output_list), len(value_output_list)
        score_matrix = np.zeros((query_num, value_num))
        target_matrix = np.zeros((query_num, value_num))

        for value_idx, v in enumerate(value_output_list):
            for query_idx, c in enumerate(query_output_list):
                score_matrix[query_idx, value_idx] = np.inner(v, c)

        for (value_idx, query_idx), target in targets.items():
            target_matrix[query_idx, value_idx] = target

        rank_matrix = np.where(np.argsort(-score_matrix,
                                          axis=-1) == np.expand_dims(np.where(target_matrix == 1)[1],
                                                                     axis=-1))[1]

        r1 = 100.0 * np.sum(rank_matrix < 1) / len(rank_matrix)
        r5 = 100.0 * np.sum(rank_matrix < 5) / len(rank_matrix)
        r10 = 100.0 * np.sum(rank_matrix < 10) / len(rank_matrix)

        medr = np.floor(np.median(rank_matrix) + 1)
        meanr = np.mean(rank_matrix) + 1
        logger.info(
            "Final r1:%.3f, r5:%.3f, r10:%.3f, mder:%.3f, meanr:%.3f"
            % (r1, r5, r10, medr, meanr)
        )
        return {
            f'r1': r1,
            f'r5': r5,
            f'r10': r10}
