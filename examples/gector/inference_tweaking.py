"""
Copyright (c) Anonymized.
Licensed under the MIT license.
"""
import logging
import os

import torch
from transformers import HfArgumentParser

from terepo.arguments import WenxinModelArguments, WenxinTrainingArguments
from terepo.arguments.gec import WenxinTextEditingDataArguments
from terepo.data.evaluators import WenxinEvaluator
from terepo.models import MODEL_REGISTRY
from terepo.models.tagging import GECToRTokenizer, GECToRConfig

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((WenxinModelArguments, WenxinTextEditingDataArguments, WenxinTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config = GECToRConfig.from_pretrained(model_args.model_name_or_path)
    logger.info("Model configurations %s", config)
    tokenizer = GECToRTokenizer.from_pretrained(model_args.model_name_or_path)

    best_pt = model_args.best_pt
    if best_pt:
        logger.info(f"Loading best checkpoint from: {best_pt}")
        config = GECToRConfig.from_pretrained(os.path.dirname(best_pt))
        logger.info("Model configurations %s", config)
        model = MODEL_REGISTRY[model_args.model_cls].from_pretrained(os.path.dirname(best_pt), config=config)
        logger.info(model)
        model.load_state_dict(torch.load(best_pt, map_location=model.device), strict=True)
    else:
        model = MODEL_REGISTRY[model_args.model_cls].from_pretrained(model_args.model_name_or_path)
        logger.info(model)
        config = GECToRConfig.from_pretrained(model_args.model_name_or_path)
        logger.info("Model configurations %s", config)

    model = model.cuda()

    evaluator = WenxinEvaluator(tokenizer, None, model_args, training_args, data_args, config)
    metrics = evaluator.evaluate(model, training_args.logging_dir)
    print(metrics)
    print(sorted(metrics.items(), key=lambda x: x[1]['f1']))
    # for k, v in metrics.items():
    #     TB_LOGGER.add_scalar(f'eval/{k}', v, global_step or 0)

    # predictor = WenxinPredictor(tokenizer, model, args.predictor, args.predictor_subname, args.task)
    # predictor.predict(args.input_file, args.output_file,
    #                   iterations=args.iteration_count,
    #                   min_error_probability=min_error_probability,
    #                   confidence_bias=confidence_bias)


if __name__ == "__main__":
    main()
