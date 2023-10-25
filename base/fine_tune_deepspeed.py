from dataclasses import field, dataclass
import json
import logging
import os
import numpy as np
from pathlib import Path
import torch
from typing import Optional, Union, Tuple, Sequence, Dict
import copy
from datasets import Dataset, load_dataset, load_from_disk
from fine_tune_base import HFTrainingArguments, get_tokenizer, get_model, DEFAULT_SEED, load_training_dataset, \
    _tokenize_fn
import transformers

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    IntervalStrategy,
    PreTrainedTokenizer,
    SchedulerType,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)

ROOT_PATH = Path(__file__).parent.parent


def train(args: HFTrainingArguments):
    set_seed(DEFAULT_SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = get_tokenizer(args.tokenizer)
    train_dataset, eval_dataset = load_training_dataset(
        tokenizer, path_or_dataset=args.dataset, max_seq_len=args.max_seq_len, args=args,
    )
    model = get_model(pretrained_name_or_path=args.model)

    if args.deepspeed_config:
        with open(args.deepspeed_config) as json_data:
            deepspeed_config_dict = json.load(json_data)
    else:
        deepspeed_config_dict = None

    training_args = Seq2SeqTrainingArguments(
        local_rank=args.local_rank,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        optim=args.optim,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_strategy=args.logging_strategy,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        fp16=args.fp16,
        bf16=args.bf16,
        deepspeed=deepspeed_config_dict,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        push_to_hub=False,
        disable_tqdm=True,
        report_to=["tensorboard"],
        # group_by_length=True,
        ddp_find_unused_parameters=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    logger.info("Training the model")
    trainer.train()

    logger.info(f"Saving Model to {args.final_model_output_path}")
    trainer.save_model(output_dir=args.final_model_output_path)
    tokenizer.save_pretrained(args.final_model_output_path)

    logger.info("Training finished.")


def main():
    parser = HfArgumentParser(HFTrainingArguments)

    parsed = parser.parse_args_into_dataclasses()
    args: HFTrainingArguments = parsed[0]

    train(args)


if __name__ == "__main__":
    cache_root = os.getenv("CACHE_ROOT", "/dbfs/Users/leoleehaoli")
    cache_hf_dir = f"{cache_root}/hf"
    cache_model_dir = f"{cache_root}/hf/models/"

    os.environ["HF_HOME"] = cache_hf_dir
    os.environ["HF_DATASETS_CACHE"] = cache_hf_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_hf_dir

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        main()
    except Exception:
        logger.exception("main failed")
        raise
