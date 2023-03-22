"""
Copyright (c) VisualJoyce.
Licensed under the MIT license.
"""
import importlib
import logging
import os

import torch

logger = logging.getLogger(__name__)

PREDICTOR_REGISTRY = {}


def register_predictor(name, subname=None):
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

    def register_predictor_cls(cls):
        if subname is None:
            if name in PREDICTOR_REGISTRY:
                raise ValueError('Cannot register duplicate model ({})'.format(name))
            PREDICTOR_REGISTRY[name] = cls
        else:
            if name in PREDICTOR_REGISTRY and subname in PREDICTOR_REGISTRY[name]:
                raise ValueError('Cannot register duplicate model ({}/{})'.format(name, subname))
            PREDICTOR_REGISTRY.setdefault(name, {})
            PREDICTOR_REGISTRY[name][subname] = cls
        return cls

    return register_predictor_cls


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
        module = importlib.import_module(f'terepo.data.predictors.{model_name}')


class TERepoPredictor(object):

    def __init__(self, tokenizer, feature_extractor, model, predictor_cls, predictor_subname, task, batch_size=8):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.model = model
        self.predictor_cls = predictor_cls
        self.predictor_subname = predictor_subname
        self.task = task
        self.batch_size = batch_size

    @torch.no_grad()
    def predict(self, input_file, output_file, **kwargs):
        self.model.eval()
        predictor = PREDICTOR_REGISTRY[self.predictor_cls][self.predictor_subname](self.tokenizer,
                                                                                   self.feature_extractor,
                                                                                   self.model,
                                                                                   self.batch_size, self.task)
        predictor.predict(input_file, output_file, **kwargs)
