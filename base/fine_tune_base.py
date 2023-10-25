from dataclasses import field, dataclass
from typing import Optional, Union, Tuple, Sequence, Dict
from pathlib import Path
import os
import logging
import torch
import copy
from datasets import Dataset, load_dataset, load_from_disk
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
    BitsAndBytesConfig,
)
from accelerate import Accelerator
from huggingface_hub import login
# Login to Huggingface to get access to the model
login(token='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')

logger = logging.getLogger(__name__)
import importlib
def is_peft_available():
    return importlib.util.find_spec("peft") is not None

if is_peft_available():
    print(f"[Load peft]")
    from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_int8_training

ROOT_PATH = Path(__file__).parent.parent
MODEL_PATH = 'meta-llama/Llama-2-7b-hf'
TOKENIZER_PATH = 'meta-llama/Llama-2-7b-hf'
DEFAULT_TRAINING_DATASET = "/dbfs/Users/leoleehaoli/dataset/HTT en-es.hf"
CONFIG_PATH = "../../config/a10_config.json"
LOCAL_OUTPUT_DIR = "/dbfs/Users/leoleehaoli/models/llama2-7b-en-es"
DEFAULT_SEED = 68
IGNORE_INDEX = -100

@dataclass
class HFTrainingArguments:
    local_rank: Optional[str] = field(default="-1")
    dataset: Optional[str] = field(default=DEFAULT_TRAINING_DATASET)
    model: Optional[str] = field(default=MODEL_PATH)
    tokenizer: Optional[str] = field(default=TOKENIZER_PATH)
    max_seq_len: Optional[int] = field(default=256)

    final_model_output_path: Optional[str] = field(default=LOCAL_OUTPUT_DIR)

    deepspeed_config: Optional[str] = field(default=CONFIG_PATH)

    output_dir: Optional[str] = field(default=None)
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_checkpointing: Optional[bool] = field(default=True)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=1e-6)
    optim: Optional[str] = field(default="adamw_hf")
    num_train_epochs: Optional[int] = field(default=None)
    max_steps: Optional[int] = field(default=-1)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)
    lr_scheduler_type: Union[SchedulerType, str] = field(
        default="cosine",
    )
    warmup_steps: int = field(default=0)
    weight_decay: Optional[float] = field(default=1)
    logging_strategy: Optional[Union[str, IntervalStrategy]] = field(
        default=IntervalStrategy.STEPS
    )
    evaluation_strategy: Optional[Union[str, IntervalStrategy]] = field(
        default=IntervalStrategy.STEPS
    )
    save_strategy: Optional[Union[str, IntervalStrategy]] = field(
        default=IntervalStrategy.STEPS
    )
    fp16: Optional[bool] = field(default=False)
    bf16: Optional[bool] = field(default=True)
    save_steps: Optional[int] = field(default=100)
    logging_steps: Optional[int] = field(default=10)
    is_quantize: Optional[bool] = field(default=False)
    prompt_template: Optional[str] = field(default=None)


def get_peft_config(lora_args:dict={}):
    from peft import LoraConfig
    lora_alpha = lora_args.get("lora_alpha", 16)
    lora_dropout = lora_args.get("lora_dropout", 0.1)
    lora_r = lora_args.get("lora_r", 64)

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'k_proj', 'v_proj']
        # Choose all linear layers from the model
    )

    return peft_config


def get_tokenizer(
    pretrained_name_or_path: str,
    use_auth_token = "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",
    padding_side = 'left',
) -> PreTrainedTokenizer:
    if "BigTranslate" in pretrained_name_or_path:
        from transformers import LlamaForCausalLM, LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(pretrained_name_or_path,trust_remote_code="true", padding_side=padding_side, use_auth_token=use_auth_token)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_name_or_path, trust_remote_code="true", padding_side=padding_side, use_auth_token=use_auth_token,
        )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_model(
    pretrained_name_or_path: str = MODEL_PATH,
    device_map = "auto",
    use_auth_token = "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",
    peft_config: dict = None,
    cache_dir: str = None,
    local_files_only = False,
) -> AutoModelForCausalLM:
    logger.info(f"Loading model: {pretrained_name_or_path} from cache_dir: {cache_dir} & peft_config={peft_config}")

    if peft_config:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_name_or_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            use_auth_token=use_auth_token,
            # device_map=device_map,
            device_map = {"": Accelerator().process_index},
            cache_dir = cache_dir,
            local_files_only=local_files_only,
        )
        model = prepare_model_for_int8_training(model)
        model = get_peft_model(model, peft_config)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained_name_or_path,
            trust_remote_code="true",
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            use_auth_token=use_auth_token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

    model.config.use_cache = False

    return model

def _tokenize_fn(texts: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, max_seq_len: int) -> Dict:
    """Tokenize a list of strings."""
    outputs = tokenizer(
        texts,
        # return_tensors="pt",
        padding="longest",
        max_length=max_seq_len - 1,
        truncation=True,
        return_length=True,
    )

    input_batch = []
    attention_masks = []
    input_lengths = []

    for length, input_ids, attention_mask in zip(
        outputs["length"], outputs["input_ids"], outputs["attention_mask"]
    ):
        length = length + 1
        input_ids = input_ids + [tokenizer.eos_token_id]
        attention_mask = attention_mask + [1]
        if length <= max_seq_len:
            input_batch.append(input_ids)
            attention_masks.append(attention_mask)
            input_lengths.append(length)

    input_batch = torch.tensor(input_batch, dtype=torch.int64)
    input_lengths = torch.tensor(input_lengths, dtype=torch.int64)

    label_batch = input_batch.clone()
    labels_lens = input_lengths.clone()

    return dict(
        input_ids=input_batch,
        labels=label_batch,
        # attention_mask=attention_masks,
        input_ids_lens=input_lengths,
        labels_lens=labels_lens,
    )


def load_training_dataset(
        tokenizer,
        path_or_dataset: str = DEFAULT_TRAINING_DATASET,
        max_seq_len: int = 256,
        seed: int = DEFAULT_SEED,
        args: HFTrainingArguments = None,
) -> Dataset:
    logger.info(f"Loading dataset from {path_or_dataset}")

    if os.path.exists(path_or_dataset):
        print(f"Loading dataset locally from {path_or_dataset}")
        dataset = load_from_disk(path_or_dataset)
    else:
        print(f"Loading dataset remotely from {path_or_dataset}")
        dataset = load_dataset(path_or_dataset)

    logger.info(f"Training: found {dataset['train'].num_rows} rows")
    logger.info(f"Eval: found {dataset['test'].num_rows} rows")

    prompt_template = '''{instruction}\n### Text:{context}\n### Translation:'''
    if args and args.prompt_template:
        prompt_template = args.prompt_template

    msg = f"prompt_template:{prompt_template}"
    print(msg)


    def _reformat_data_sft(batch):
        # return batch['prompt']
        sources = []
        targets = []
        texts = []
        for instruction, context, response in zip(batch["instruction"], batch["context"], batch["response"]):
            prompt_dict = {
                "instruction": instruction,
                "context": context,
            }
            source = prompt_template.format_map(prompt_dict) #f'''{instruction}\n### Text:{context}\n### Translation:'''
            target = response
            text = source + target
            sources.append(source)
            targets.append(target)
            texts.append(text)

        return sources, targets, texts

    def _tokenize_sft(element):
        sources, targets, examples = _reformat_data_sft(element)
        examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer, max_seq_len) for strings in
                                                 (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX

        result_dict = dict(input_ids=input_ids, labels=labels)
        return result_dict


    tokenize = _tokenize_sft

    train_tokenized_dataset = dataset["train"].map(
        tokenize, batched=True, remove_columns=dataset["train"].column_names, #batch_size=args.per_device_train_batch_size,
    )
    eval_tokenized_dataset = dataset["test"].map(
        tokenize, batched=True, remove_columns=dataset["test"].column_names, #batch_size=args.per_device_eval_batch_size,
    )

    return train_tokenized_dataset, eval_tokenized_dataset