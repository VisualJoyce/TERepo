"""
Copyright (c) Anonymized.
Licensed under the MIT license.
"""
import argparse
import logging
import os

import torch
from transformers import AutoTokenizer

from terepo.data.predictors import TERepoPredictor
from terepo.models import MODEL_REGISTRY
from terepo.models.tagging import GECToRFeatureExtractor, GECToRConfig

logger = logging.getLogger(__name__)


def main(args):
    # model.resize_token_embeddings(len(tokenizer))
    model = MODEL_REGISTRY[args.model_cls].from_pretrained(args.model_name_or_path)
    logger.info(model)
    config = model.config
    logger.info("Model configurations %s", config)

    best_pt = args.best_pt
    if best_pt:
        logger.info(f"Loading best checkpoint from: {best_pt}")
        model.load_state_dict(torch.load(best_pt, map_location=model.device), strict=True)

    model = model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, config=config.encoder)
    feature_extractor = GECToRFeatureExtractor.from_pretrained(args.model_name_or_path)
    predictor = TERepoPredictor(tokenizer, feature_extractor,
                                model, args.predictor, args.predictor_subname, args.task,
                                batch_size=args.batch_size)
    predictor.predict(args.input_file, args.output_file,
                      iterations=args.iteration_count,
                      min_error_probability=args.min_error_probability,
                      confidence_bias=args.confidence_bias)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', required=True, help='annotation JSON')
    parser.add_argument('--model_cls', required=True, help='annotation JSON')
    parser.add_argument('--best_pt', type=str, default=None, help='JSON config files')
    parser.add_argument('--predictor', type=str, default=None, help='JSON config files')
    parser.add_argument('--predictor_subname', type=str, default=None, help='JSON config files')
    parser.add_argument('--task', type=str, default=None, help='JSON config files')
    parser.add_argument('--input_file', type=str, default=None, help='JSON config files')
    parser.add_argument('--output_file', type=str, default=None, help='JSON config files')
    parser.add_argument('--iteration_count',
                        type=int,
                        help='The number of iterations of the model',
                        default=1)  # 迭代修改轮数
    parser.add_argument('--confidence_bias',
                        type=float,
                        help='How many probability to add to $KEEP token.',
                        default=0)
    parser.add_argument('--min_error_probability',
                        type=float,
                        help='Minimum probability for each action to apply. '
                             'Also, minimum error probability, as described in the paper.',
                        default=0.0)
    parser.add_argument('--batch_size',
                        type=int,
                        help='The number of iterations of the model',
                        default=1)
    args = parser.parse_args()
    main(args)
