from dataclasses import field, dataclass
from typing import Optional

from terepo.arguments.base import TERepoDataArguments


@dataclass
class TERepoDataArguments(TERepoDataArguments):
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
