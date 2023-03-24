"""
Copyright (c) Anonymized.
Licensed under the MIT license.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import HfArgumentParser, AutoTokenizer

from terepo.arguments.base import TERepoModelArguments, TERepoTrainingArguments, TERepoDataArguments
from terepo.data.evaluators import TERepoEvaluator
from terepo.models import MODEL_REGISTRY
from terepo.models.tagging import GECToRConfig, GECToRFeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class TERepoGECDataArguments(TERepoDataArguments):
    task_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
        },
    )

    eval_gold_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
        },
    )

    use_delete_then_append_prob: Optional[float] = field(default=0, metadata={"help": "Whether to run training."})
    use_correct_lines_prob: Optional[float] = field(default=0, metadata={"help": "Whether to run training."})
    focal_gamma: Optional[float] = field(default=2, metadata={"help": "Whether to run training."})
    hop_loss_rate: Optional[float] = field(default=8, metadata={"help": "Whether to run training."})
    hop_num: Optional[int] = field(default=5, metadata={"help": "Whether to run training."})
    use_matching_dropout: Optional[bool] = field(default=False, metadata={"help": "Whether to run training."})
    use_matching_layernorm: Optional[bool] = field(default=False, metadata={"help": "Whether to run training."})


def main():
    parser = HfArgumentParser((TERepoModelArguments, TERepoGECDataArguments, TERepoTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config = GECToRConfig.from_pretrained(model_args.model_name_or_path)
    logger.info("Model configurations %s", config)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, config=config.encoder)
    feature_extractor = GECToRFeatureExtractor.from_pretrained(model_args.model_name_or_path)
    model = MODEL_REGISTRY[model_args.model_cls].from_pretrained(model_args.model_name_or_path)
    logger.info(model)
    logger.info("Model configurations %s", config)

    best_pt = model_args.best_pt
    if best_pt:
        logger.info(f"Loading best checkpoint from: {best_pt}")
        model.load_state_dict(torch.load(best_pt, map_location=model.device), strict=True)

    model = model.cuda()

    evaluator = TERepoEvaluator(tokenizer, feature_extractor, model_args, training_args, data_args, config)
    metrics = evaluator.evaluate(model, training_args.logging_dir)
    print(metrics)
    print(sorted(metrics.items(), key=lambda x: x[1]['f1']))


if __name__ == "__main__":
    main()
