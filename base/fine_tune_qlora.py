import logging
import os
import torch
from typing import Optional, Union, Tuple, Sequence, Dict
from fine_tune_base import HFTrainingArguments, get_peft_config, get_tokenizer, get_model, DEFAULT_SEED, \
    load_training_dataset
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

# from trl import SFTTrainer
logger = logging.getLogger(__name__)


def train(args: HFTrainingArguments):
    set_seed(DEFAULT_SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = get_tokenizer(args.tokenizer)
    train_dataset, eval_dataset = load_training_dataset(
        tokenizer, path_or_dataset=args.dataset, max_seq_len=args.max_seq_len, args=args,
    )
    peft_config = get_peft_config()
    model = get_model(pretrained_name_or_path=args.model, peft_config=peft_config)

    max_grad_norm = 0.3
    warmup_ratio = 0.03

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
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        push_to_hub=False,
        disable_tqdm=True,
        report_to=["tensorboard"],
        # group_by_length=True,
        ddp_find_unused_parameters=False,
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,

    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

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

