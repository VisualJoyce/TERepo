"""
Copyright (c) VisualJoyce.
Licensed under the MIT license.
"""
import importlib
import logging
import os

import torch
import wandb
from transformers import PretrainedConfig, BertForSequenceClassification

from terepo.arguments import TERepoModelArguments, TERepoTrainingArguments, TERepoDataArguments
from terepo.data.loaders import EVAL_LOADER_REGISTRY

logger = logging.getLogger(__name__)

EVALUATOR_REGISTRY = {}


def register_evaluator(name, subname=None):
    """
    New model types can be added to fairseq with the :func:`register_model`
    function decorator.

    For example::

        @register_model('lstm')
        class LSTM(FairseqEncoderDecoderModel):
            (...)

    .. note:: All models must implement the :class:`BaseFairseqModel` interface.
        Typically you will extend :class:`FairseqEncoderDecoderModel` for
        sequence-to-sequence tasks or :class:`FairseqLanguageModel` for
        language modeling tasks.

    Args:
        name (str): the name of the model
        :param name:
        :param subname:
    """

    def register_evaluator_cls(cls):
        if subname is None:
            if name in EVALUATOR_REGISTRY:
                raise ValueError('Cannot register duplicate model ({})'.format(name))
            EVALUATOR_REGISTRY[name] = cls
        else:
            if name in EVALUATOR_REGISTRY and subname in EVALUATOR_REGISTRY[name]:
                raise ValueError('Cannot register duplicate model ({}/{})'.format(name, subname))
            EVALUATOR_REGISTRY.setdefault(name, {})
            EVALUATOR_REGISTRY[name][subname] = cls
        return cls

    return register_evaluator_cls


# automatically import any Python files in the models/ directory
datasets_dir = os.path.dirname(__file__)
for file in os.listdir(datasets_dir):
    path = os.path.join(datasets_dir, file)
    if (
            not file.startswith('_')
            and not file.startswith('.')
            and (file.endswith('.py') or os.path.isdir(path))
    ):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module(f'terepo.data.evaluators.{model_name}')


class TERepoEvaluator(object):

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
        # self.config.num_labels = 2 if 'entity_sentiment_prompt' in self.modalities else 5
        self.delimiter = "@"

    def construct_model(self, modality):
        device = self.training_args.device
        state_dict = None
        if self.model_args.best_pt:
            logger.info(f"Loading best checkpoint from: {self.model_args.best_pt}")
            state_dict = torch.load(self.model_args.best_pt, map_location=device)

        model_class = BertForSequenceClassification
        try:
            model = model_class.from_pretrained(self.model_args.model_name_or_path,
                                                state_dict=state_dict,
                                                config=self.config)
        except OSError:
            model = model_class(self.config)
            if state_dict is not None:
                model.load_state_dict(state_dict, strict=False)
        model = model.to(device)

        return model

    @staticmethod
    def eval_loader_cls(m, e_subname=None):
        return EVAL_LOADER_REGISTRY[m] if e_subname is None else EVAL_LOADER_REGISTRY[m][e_subname]

    @staticmethod
    def evaluator_cls(m, m_subname=None):
        return EVALUATOR_REGISTRY[m] if m_subname is None else EVALUATOR_REGISTRY[m][m_subname]

    def parse_eval_paras(self):
        eval_files_list = None
        if hasattr(self.data_args, f'eval_files'):
            eval_files = getattr(self.data_args, f'eval_files')
            if eval_files not in (None, ''):
                eval_files_list = eval_files.split("@")

        n = len(eval_files_list)

        eval_loader_classes = self.data_args.eval_loader_names.split(self.delimiter)
        if len(eval_loader_classes) == 1:
            eval_loader_classes = [eval_loader_classes[0]] * n

        evaluator_subname_list = None
        if hasattr(self.data_args, f'evaluator_subnames'):
            evaluator_subnames = getattr(self.data_args, f'evaluator_subnames')
            if evaluator_subnames:
                evaluator_subname_list = evaluator_subnames.split('@')
                if len(evaluator_subname_list) == 1:
                    evaluator_subname_list = [evaluator_subname_list[0]] * n

        if hasattr(self.data_args, f'eval_loader_subnames') and getattr(self.data_args, f'eval_loader_subnames'):
            eval_loader_subnames = self.data_args.eval_loader_subnames.split(self.delimiter)
            if len(eval_loader_subnames) == 1:
                eval_loader_subnames = [eval_loader_subnames[0]] * n
        else:
            eval_loader_subnames = [None] * n

        for ef, ec, es, ev in zip(eval_files_list, eval_loader_classes, eval_loader_subnames, evaluator_subname_list):
            yield ef, ec, es, ev

    @property
    def loaders(self):
        d = [[ef, ec, ev, ep] for ef, ec, ev, ep in self.parse_eval_paras()]
        return d

    @torch.no_grad()
    def evaluate(self, model=None, output_dir=None):
        logging.info(f"Modalities: {self.loaders}")

        metrics = {}
        for ef, ec, es, ev in self.loaders:
            logger.info(f"Evaluating {ec}")
            loader = self.eval_loader_cls(ec, e_subname=es)(self.tokenizer, self.feature_extractor, ef,
                                                            self.model_args,
                                                            self.training_args,
                                                            self.data_args,
                                                            self.config)

            evaluator = self.evaluator_cls(ec, m_subname=ev)(self.tokenizer, self.feature_extractor,
                                                             self.model_args,
                                                             self.training_args,
                                                             self.data_args,
                                                             self.config)

            split = os.path.basename(ef)

            if model is None:
                model = self.construct_model(ec)

            model.eval()

            eval_name = f'{ev or ""}_{"-".join(ef.split(os.path.sep)[-2:])}'
            evaluator(model, loader, output_dir)
            for k, v in evaluator.metrics.items():
                if k == 'submission':
                    with open(os.path.join(output_dir, f'submission-{m}-{split}.txt'), 'w') as f:
                        f.write(v.getvalue())
                else:
                    logger.info(f"{eval_name}_{k} : {v}")
                    metrics[f'{eval_name}_{k}'] = v
        logger.info(metrics)
        model.train()
        return metrics
