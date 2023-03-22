"""
Copyright (c) VisualJoyce.
Licensed under the MIT license.
"""
import importlib
import logging
import os
from abc import abstractmethod, ABCMeta
from collections import defaultdict
from itertools import chain, islice

import torch
from transformers import BatchFeature, PretrainedConfig

from terepo.arguments.base import TERepoModelArguments, TERepoTrainingArguments, TERepoDataArguments

logger = logging.getLogger(__name__)


def move_to_cuda(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.cuda(non_blocking=True, device=device)
    elif isinstance(batch, list):
        new_batch = [move_to_cuda(t, device=device) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(move_to_cuda(t, device=device) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: move_to_cuda(t, device=device) for n, t in batch.items()}
    elif isinstance(batch, BatchFeature):
        new_batch = {n: move_to_cuda(t, device=device) for n, t in batch.items()}
    else:
        return batch
    return new_batch


def record_cuda_stream(batch, device):
    if isinstance(batch, torch.Tensor):
        batch.record_stream(torch.cuda.current_stream(device=device))
    elif isinstance(batch, list) or isinstance(batch, tuple):
        for t in batch:
            record_cuda_stream(t, device)
    elif isinstance(batch, dict):
        for t in batch.values():
            record_cuda_stream(t, device)
    else:
        pass


class PrefetchLoader(object):
    """
    overlap compute and cuda data transfer
    (copied and then modified from nvidia apex)
    """

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)

    def __iter__(self):
        loader_it = iter(self.loader)
        self.preload(loader_it)
        batch = self.next(loader_it)
        while batch is not None:
            yield batch
            batch = self.next(loader_it)

    def __len__(self):
        return len(self.loader)

    def preload(self, it):
        try:
            self.batch = next(it)
        except StopIteration:
            self.batch = None
            return
        # if record_stream() doesn't work, another option is to make sure
        # device inputs are created on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input,
        #                                        device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target,
        #                                         device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use
        # by the main stream at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.batch = move_to_cuda(self.batch, device=self.device)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this
            # side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

    def next(self, it):
        torch.cuda.current_stream(device=self.device).wait_stream(self.stream)
        batch = self.batch
        if batch is not None:
            record_cuda_stream(batch, self.device)
        self.preload(it)
        return batch

    def __getattr__(self, name):
        method = self.loader.__getattribute__(name)
        return method


LOADER_REGISTRY = {}


def register_loader(name, subname=None):
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

    def register_loader_cls(cls):
        if subname is None:
            if name in LOADER_REGISTRY:
                raise ValueError('Cannot register duplicate model ({})'.format(name))
            LOADER_REGISTRY[name] = cls
        else:
            if name in LOADER_REGISTRY and subname in LOADER_REGISTRY[name]:
                raise ValueError('Cannot register duplicate model ({}/{})'.format(name, subname))
            LOADER_REGISTRY.setdefault(name, {})
            LOADER_REGISTRY[name][subname] = cls
        return cls

    return register_loader_cls


EVAL_LOADER_REGISTRY = {}


def register_eval_loader(name, subname=None):
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
    """

    def register_eval_loader_cls(cls):
        if subname is None:
            if name in EVAL_LOADER_REGISTRY:
                raise ValueError('Cannot register duplicate model ({})'.format(name))
            EVAL_LOADER_REGISTRY[name] = cls
        else:
            if name in EVAL_LOADER_REGISTRY and subname in EVAL_LOADER_REGISTRY[name]:
                raise ValueError('Cannot register duplicate model ({}/{})'.format(name, subname))
            EVAL_LOADER_REGISTRY.setdefault(name, {})
            EVAL_LOADER_REGISTRY[name][subname] = cls
        return cls

    return register_eval_loader_cls


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
        module = importlib.import_module(f'terepo.data.loaders.{model_name}')


class MMLoader(metaclass=ABCMeta):

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

        self.all_epochs = defaultdict(int)
        self.step = 0  # how many batches have been fired

        self.delimiter = "@"

    def parse_train_paras(self):
        train_files_list = None
        if hasattr(self.data_args, f'train_files'):
            train_files = getattr(self.data_args, f'train_files')
            if train_files not in (None, ''):
                train_files_list = train_files.split(self.delimiter)

        n = len(train_files_list)

        train_loader_classes = self.data_args.train_loader_names.split(self.delimiter)
        if len(train_loader_classes) == 1:
            train_loader_classes = [train_loader_classes[0]] * n

        if hasattr(self.data_args, f'train_loader_subnames') and getattr(self.data_args, f'train_loader_subnames'):
            train_loader_subnames = self.data_args.train_loader_subnames.split(self.delimiter)
            if len(train_loader_subnames) == 1:
                train_loader_subnames = [train_loader_subnames[0]] * n
        else:
            train_loader_subnames = [None] * n

        if hasattr(self.data_args, f'train_proportions') and getattr(self.data_args, f'train_proportions'):
            train_proportions = list(map(int, self.data_args.train_proportions.split(self.delimiter)))
        else:
            train_proportions = [1] * n

        for ef, ec, ev, ep in zip(train_files_list, train_loader_classes, train_loader_subnames, train_proportions):
            yield ef, ec, ev, ep

    @property
    def loaders(self):
        d = []
        for ef, ec, ev, ep in self.parse_train_paras():
            d.extend([[ef, ec, ev]] * ep)
        return d

    def _use_at_most_k_wrapper(self, loader):
        if self.training_args.use_at_most_k:
            k = self.training_args.use_at_most_k
            return islice(loader, k)
        return loader

    @abstractmethod
    def _interchange_iter(self):
        raise NotImplementedError

    def __iter__(self):
        logging.info(f"Loaders: {self.loaders}")
        while True:
            self.all_epochs['epoch'] += 1
            for batch in PrefetchLoader(self._interchange_iter(), self.training_args.device):
                yield batch
                self.step += 1


class MMLoaderWiseLoader(MMLoader):

    def __init__(self, tokenizer, feature_extractor,
                 model_args: TERepoModelArguments, training_args: TERepoTrainingArguments,
                 data_args: TERepoDataArguments, config: PretrainedConfig):
        super().__init__(tokenizer, feature_extractor, model_args, training_args, data_args, config)

    # def _roll_modalities(self):
    #     idx = np.arange(len(self.loaders))
    #     return [self.loaders[i] for i in np.roll(idx, self.training_args.local_rank)]

    @staticmethod
    def loader_cls(m, e_subname=None):
        return LOADER_REGISTRY[m] if e_subname is None else LOADER_REGISTRY[m][e_subname]

    def _interchange_iter(self):
        # modalities = self._roll_modalities() if self.training_args.roll_modalities else self.loaders
        loaders = map(self._use_at_most_k_wrapper, [self.loader_cls(ec, ev)(self.tokenizer,
                                                                            self.feature_extractor,
                                                                            ef,
                                                                            self.model_args,
                                                                            self.training_args,
                                                                            self.data_args,
                                                                            self.config) for
                                                    ef, ec, ev in self.loaders])

        for batch in chain(*loaders):
            cur_step = self.step + 1
            _check_ = (cur_step + 1) % self.training_args.gradient_accumulation_steps == 0
            batch['should_grad_sync_and_apply'] = True if _check_ else False
            batch['gradient_accumulation_steps'] = self.training_args.gradient_accumulation_steps
            yield batch


class MMStepWiseLoader(MMLoader):

    def __init__(self, tokenizer, feature_extractor,
                 model_args: TERepoModelArguments, training_args: TERepoTrainingArguments,
                 data_args: TERepoDataArguments, config: PretrainedConfig):
        super().__init__(tokenizer, feature_extractor, model_args, training_args, data_args, config)
        assert self.training_args.roll_modalities is False

        self.all_gradient_accumulation_steps = {
            m: getattr(self.training_args,
                       f'{m}_gradient_accumulation_steps') or self.training_args.gradient_accumulation_steps
            for m in self.loaders
        }
        logger.info(f"all_gradient_accumulation_steps: {self.all_gradient_accumulation_steps}")

        self.interchange_steps = sum(self.all_gradient_accumulation_steps.values())
        logger.info(f"interchange_steps: {self.interchange_steps}")

    def _get_modality_from_step(self):
        step_idx = self.step % self.interchange_steps
        start = 0
        for m in self.loaders:
            gradient_accumulation_steps = self.all_gradient_accumulation_steps[m]
            end = start + gradient_accumulation_steps
            if start <= step_idx < end:
                if step_idx + 1 == end:
                    return m, True, gradient_accumulation_steps
                else:
                    return m, False, gradient_accumulation_steps
            start = end

    def _interchange_iter(self):

        loaders = {m: iter(self._use_at_most_k_wrapper(LOADER_REGISTRY[m](self.tokenizer,
                                                                          self.feature_extractor,
                                                                          self.model_args,
                                                                          self.training_args,
                                                                          self.data_args,
                                                                          self.config))) for m in self.loaders}

        while True:
            m, should_grad_sync_and_apply, gradient_accumulation_steps = self._get_modality_from_step()
            try:
                batch = next(loaders[m])
                batch['should_grad_sync_and_apply'] = should_grad_sync_and_apply
                batch['gradient_accumulation_steps'] = gradient_accumulation_steps
                yield batch
            except StopIteration:
                self.all_epochs[m] += 1
                loaders[m] = iter(self._use_at_most_k_wrapper(LOADER_REGISTRY[m](self.tokenizer,
                                                                                 self.feature_extractor,
                                                                                 self.model_args,
                                                                                 self.training_args,
                                                                                 self.data_args,
                                                                                 self.config)))


MM_LOADERS = {
    'step_wise': MMStepWiseLoader,
    'modality_wise': MMLoaderWiseLoader
}
