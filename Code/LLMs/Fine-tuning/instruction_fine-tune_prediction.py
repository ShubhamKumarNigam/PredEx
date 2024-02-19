#!pip install -q -U bitsandbytes
## !pip install -q -U git+https://github.com/huggingface/transformers.git
#!pip install transformers==4.31
#!pip install -q -U git+https://github.com/huggingface/peft.git
#!pip install -q -U git+https://github.com/huggingface/accelerate.git
#!pip install -q datasets
#!pip install evaluate
#!pip install -qqq trl==0.7.1
import time
import torch
#import evaluate
import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset
import random


df_train = pd.read_csv("train_ft.csv")
# df_train = df_train.head(50)
df_val = pd.read_csv("val_ft.csv")
# df_val = df_val.head(10)
#df = df.drop(columns = ["Unnamed: 0"])
train_data = Dataset.from_pandas(df_train)
val_data = Dataset.from_pandas(df_val)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "meta-llama/Llama-2-7b-chat-hf"

bnb_config = BitsAndBytesConfig(
     load_in_4bit=True,
     bnb_4bit_use_double_quant=True,
     bnb_4bit_quant_type="nf4",
     bnb_4bit_compute_dtype=torch.bfloat16)


model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", token="...........")
#model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", token="..........")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():

        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

print(model)

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    # target_modules=["query_key_value"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], #specific to Llama models.
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print_trainable_parameters(model)

OUTPUT_DIR = "ckpts"

from transformers import TrainingArguments

training_arguments = TrainingArguments(
    per_device_train_batch_size=16,    
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    logging_steps=1,
    learning_rate=1e-4,
    fp16=True,
    max_grad_norm=0.3,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    eval_steps=0.2,
    warmup_ratio=0.05,
    save_strategy="epoch",
    logging_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    group_by_length=True,
    output_dir=OUTPUT_DIR,
    save_safetensors=True,
    lr_scheduler_type="cosine",
    seed=42,
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

from trl import SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=1060,
    tokenizer=tokenizer,
    args=training_arguments,
)

trainer.train()

peft_model_path="./llama_pred"

trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)
