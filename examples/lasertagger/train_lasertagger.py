"""
Copyright (c) VisualJoyce.
Licensed under the MIT license.
"""
import logging
import os
from collections import Counter

import torch
from torch.cuda.amp import autocast
from torch.optim import AdamW
from tqdm import tqdm
from transformers import HfArgumentParser, set_seed, get_linear_schedule_with_warmup

from terepo.arguments.base import WenxinModelArguments, WenxinTrainingArguments
from terepo.arguments.gec import WenxinTextEditingDataArguments
from terepo.data.evaluators import WenxinEvaluator
from terepo.data.loaders import MM_LOADERS
from terepo.models import MODEL_REGISTRY
from terepo.models.tagging import LaserTaggerTokenizer
from terepo.utils.misc import NoOp

logger = logging.getLogger(__name__)
BUFSIZE = 40960000


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def do_train(model, scaler, train_data, training_args):
    loss_counter = Counter()
    for step, batch in enumerate(train_data):
        should_grad_sync_and_apply = batch.pop('should_grad_sync_and_apply')
        gradient_accumulation_steps = batch.pop('gradient_accumulation_steps')

        if not should_grad_sync_and_apply:
            with autocast():
                if training_args.local_rank != -1:
                    with model.no_sync():
                        outputs = model(**batch, return_dict=True)
                        batch_loss = outputs.loss / gradient_accumulation_steps
                else:
                    outputs = model(**batch, return_dict=True)
                    batch_loss = outputs.loss / gradient_accumulation_steps

            scaler.scale(batch_loss).backward()
            loss_counter.update({
                "loss": batch_loss.item(),
                "loss_d": outputs.loss_d.item() / gradient_accumulation_steps,
                "loss_labels": outputs.loss_labels.item() / gradient_accumulation_steps
            })
        else:
            with autocast():
                outputs = model(**batch, return_dict=True)
                batch_loss = outputs.loss / gradient_accumulation_steps

            scaler.scale(batch_loss).backward()
            loss_counter.update({
                "loss": batch_loss.item(),
                "loss_d": outputs.loss_d.item() / gradient_accumulation_steps,
                "loss_labels": outputs.loss_labels.item() / gradient_accumulation_steps
            })
            yield loss_counter

            loss_counter = Counter()


# light
# @light_init(params={"training_framework": "pytorch_ddp"})
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((WenxinModelArguments, WenxinTextEditingDataArguments, WenxinTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.local_rank in [-1, 0]:
        from terepo.utils.logger import TensorboardLogger
        TB_LOGGER = TensorboardLogger()
        TB_LOGGER.create(training_args.logging_dir)
        pbar = tqdm(total=training_args.max_steps, desc=model_args.model_cls)
    else:
        pbar = NoOp()
        TB_LOGGER = NoOp()
    # training_args.local_rank = 0  # for debug

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Data parameters %s", data_args)

    # set_seed(training_args.seed)
    set_seed(training_args.seed + training_args.process_index)

    tokenizer = LaserTaggerTokenizer.from_pretrained(model_args.model_name_or_path,
                                                     extra_labels_vocab_file=model_args.extra_labels_vocab_file)

    device = training_args.device
    global_step = 0

    best_pt = model_args.best_pt
    if best_pt:
        logger.info(f"Loading best checkpoint from: {best_pt}")
        config = GECToRConfig.from_pretrained(os.path.dirname(best_pt))
        config.label_vocab_size += tokenizer.extra_label_vocab_size
        config.focal_gamma = data_args.focal_gamma
        model = MODEL_REGISTRY[model_args.model_cls].from_pretrained(model_args.model_name_or_path, config=config)
        model.load_state_dict(torch.load(best_pt, map_location=device), strict=True)
    else:
        config = GECToRConfig.from_pretrained(model_args.model_name_or_path)
        config.label_vocab_size += tokenizer.extra_label_vocab_size
        config.focal_gamma = data_args.focal_gamma
        config.use_matching_dropout = data_args.use_matching_dropout
        config.use_matching_layernorm = data_args.use_matching_layernorm
        logger.info("Model configurations %s", config)

        model = MODEL_REGISTRY[model_args.model_cls].from_pretrained(model_args.model_name_or_path, config=config)
        if tokenizer.extra_label_vocab_size > 0:
            model.resize_label_embeddings(config.label_vocab_size)

    logger.info(model)
    model = model.to(device)

    evaluator = WenxinEvaluator(tokenizer, None, model_args, training_args, data_args, config)

    # Training
    if training_args.do_train:
        logger.info("getting data")
        train_data = MM_LOADERS[training_args.interchange_mode](tokenizer, None,
                                                                model_args, training_args, data_args,
                                                                config)  # infinite data generator

        logger.info("init trainer")
        # Initialize our Trainer

        logger.info("start training")

        # do train:
        model.train()

        scaler = torch.cuda.amp.GradScaler()

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=training_args.learning_rate,
            betas=(training_args.adam_beta1, training_args.adam_beta2),
            eps=training_args.adam_epsilon,
        )
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=training_args.warmup_steps,
                                                       num_training_steps=training_args.max_steps)

        if os.path.isfile(os.path.join(model_args.model_name_or_path, "scheduler.pt")):
            optimizer.load_state_dict(model_args.model_name_or_path + '')

        # os.environ['MASTER_ADDR'] = 'localhost'  # for debug
        # os.environ['MASTER_PORT'] = '8888'
        # torch.distributed.init_process_group(backend='nccl',init_method='env://',
        # world_size=1, rank=training_args.local_rank)  # for debug
        if training_args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[training_args.local_rank],
                output_device=training_args.local_rank,
                find_unused_parameters=True)
        model.zero_grad()

        logger.info("start iterate")
        # do train
        for global_step, loss_counter in enumerate(
                do_train(model, scaler, train_data, training_args), start=1):
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            model.zero_grad()
            pbar.update(1)

            for k, v in loss_counter.items():
                TB_LOGGER.add_scalar(f'train/{k}', v, global_step)
            TB_LOGGER.add_scalar('train/grad_norm', total_norm, global_step)
            TB_LOGGER.add_scalar('train/focal_gamma', config.focal_gamma, global_step)
            for k, v in train_data.all_epochs.items():
                TB_LOGGER.add_scalar(f'train/{k}', v, global_step)
            for gid, group in enumerate(optimizer.param_groups):
                TB_LOGGER.add_scalar(f'train/lr_{gid}', group['lr'], global_step)
            TB_LOGGER.step()

            if global_step % training_args.save_steps == 0:
                ckpt_output_dir = os.path.join(training_args.output_dir, 'ckpt' + str(global_step))
                tokenizer.save_pretrained(ckpt_output_dir)
                config.save_pretrained(ckpt_output_dir)
                # processor.save_pretrained(ckpt_output_dir)  # for speech
                if training_args.local_rank != -1:
                    model.module.save_pretrained(ckpt_output_dir)
                else:
                    model.save_pretrained(ckpt_output_dir)

                if training_args.local_rank in [-1, 0]:
                    metrics = evaluator.evaluate(model, ckpt_output_dir)
                    for k, v in metrics.items():
                        TB_LOGGER.add_scalar(f'eval/{k}', v, global_step)

            if global_step > training_args.max_steps:
                break

    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        metrics = evaluator.evaluate(model, training_args.logging_dir)
        for k, v in metrics.items():
            TB_LOGGER.add_scalar(f'eval/{k}', v, global_step or 0)


if __name__ == "__main__":
    main()
