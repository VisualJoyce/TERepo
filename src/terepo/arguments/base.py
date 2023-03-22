"""
Copyright (c) VisualJoyce.
Licensed under the MIT license.
"""
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class TERepoModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )

    model_cls: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )

    extra_labels_vocab_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )

    best_pt: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )


@dataclass
class TERepoDataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_files: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
                    "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
        },
    )

    train_loader_names: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
                    "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
        },
    )

    train_loader_subnames: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
        },
    )

    train_proportions: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
                    "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
        },
    )

    eval_files: Optional[str] = field(
        default=None,
        metadata={
            "help": "for eval text down stream"
        },
    )

    eval_loader_names: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
                    "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
        },
    )

    eval_loader_subnames: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
        },
    )

    evaluator_subnames: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
        },
    )

    block_size: int = field(
        default=512,
        metadata={
        },
    )


@dataclass
class TERepoTrainingArguments(TrainingArguments):
    use_at_most_k: int = field(
        default=None, metadata={"help": "use at most k training examples for each modality"}
    )

    roll_modalities: bool = field(
        default=False, metadata={"help": "use at most k training examples for each modality"}
    )

    interchange_mode: str = field(
        default='modality_wise', metadata={"help": "use at most k training examples for each modality"}
    )

    evaluators: str = field(
        default=None, metadata={"help": "use at most k training examples for each modality"}
    )

    train_num_workers: int = field(
        default=8,
        metadata={
        },
    )

    train_without_webdataset_sorted: bool = field(
        default=False, metadata={"help": "use at most k training examples for each modality"}
    )


@dataclass
class TERepoPredictorArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )

    best_pt: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
