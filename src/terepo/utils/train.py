from collections import Counter

from torch import autocast


def do_train(model, scaler, train_data, training_args, device_type='cuda'):
    loss_counter = Counter()
    for step, batch in enumerate(train_data):
        should_grad_sync_and_apply = batch.pop('should_grad_sync_and_apply')
        gradient_accumulation_steps = batch.pop('gradient_accumulation_steps')

        if not should_grad_sync_and_apply:
            with autocast(device_type):
                if training_args.local_rank != -1:
                    with model.no_sync():
                        outputs = model(**batch, return_dict=True)
                        batch_loss = outputs.loss / gradient_accumulation_steps
                else:
                    outputs = model(**batch, return_dict=True)
                    batch_loss = outputs.loss / gradient_accumulation_steps

            scaler.scale(batch_loss).backward()
            loss_counter.update({"loss": batch_loss.item()})
        else:
            with autocast(device_type):
                outputs = model(**batch, return_dict=True)
                batch_loss = outputs.loss / gradient_accumulation_steps

            scaler.scale(batch_loss).backward()
            loss_counter.update({"loss": batch_loss.item()})
            yield loss_counter

            loss_counter = Counter()
