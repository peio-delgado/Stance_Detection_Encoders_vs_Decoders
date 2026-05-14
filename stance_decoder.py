import os
os.environ["HF_HOME"] = "/home/pdelgado010/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/home/pdelgado010/.cache/huggingface"
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import re
import string
import seaborn as sns
import torch
import sys
from sklearn.metrics import classification_report
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig



#Functions 
def load_data(fnames):
    """
    Loads the files in the paths stored in the fnames list that comes as an input.
    Once loaded, it concatenates them to form a single pd.Dataframe. Finally it finds the
    unique values in 'Target' to return all the possible label values as a list with the
    formed dataframe.
    """
    data = []
    for fname in fnames:
        data.append(pd.read_csv(fname, sep=',', encoding='utf-8'))
    data = pd.concat(data)
    return data

def format_sample(row):
    return f"Tweet: {row['Tweet']}\nTarget: {row['Target']}\nStance: {row['Stance']}"

def parse_label(generated_text):
    text = generated_text.upper().translate(str.maketrans('', '', string.punctuation)).strip()
    if "FAVOR" in text:
        return "FAVOR"
    else:
        return "AGAINST"

def evaluate(model, tokenizer, test_dataset, device="cuda"):
    model.eval()
    predictions = []
    ground_truth = []

    for _, row in test_dataset.iterrows():
        prompt = (f"<s>[INST] You are a stance classification system. "
    f"Classify the stance as EXACTLY one of: FAVOR, AGAINST.\n\n" # Eliminado NONE
    f"Tweet: {row['Tweet']}\n"
    f"Target: {row['Target']}\n"
    f"Answer with only one word: FAVOR or AGAINST. [/INST]"
    )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        predictions.append(parse_label(generated))
        ground_truth.append(row["Stance"])

    print(classification_report(ground_truth, predictions, zero_division=0))
    return predictions, ground_truth


#Loading all the data for training

train_file_trump = "PStance data/raw_train_trump.csv"
val_file_trump = "PStance data/raw_val_trump.csv"

training_trump = load_data([train_file_trump])
val_trump = load_data([val_file_trump])

train_file_biden  = "PStance data/raw_train_biden.csv"
train_file_bernie = "PStance data/raw_train_bernie.csv"
training_biden  = load_data([train_file_biden])
training_bernie = load_data([train_file_bernie])

#All the testing files, now with the three targets

test_file_trump = "PStance data/raw_test_trump.csv"
testing_trump = load_data([test_file_trump])
#print(testing_trump[0:5])

test_file_biden = "PStance data/raw_test_biden.csv"
testing_biden = load_data([test_file_biden])
#print(testing_biden[0:5])

test_file_bernie = "PStance data/raw_test_bernie.csv"
testing_bernie = load_data([test_file_bernie])
#print(testing_bernie[0:5])

#Converting to HuggingFace Dataset
train_dataset = Dataset.from_pandas(training_trump)
train_dataset = train_dataset.map(lambda x: {"text": format_sample(x)})

val_dataset = Dataset.from_pandas(val_trump)
val_dataset = val_dataset.map(lambda x: {"text": format_sample(x)})

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

#Zero-shot 

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    device_map="auto",
    torch_dtype=torch.float16,
)

print("=== Zero-shot | Trump ===")
evaluate(model, tokenizer, testing_trump)

print("=== Zero-shot | Biden ===")
evaluate(model, tokenizer, testing_biden)

print("=== Zero-shot | Bernie ===")
evaluate(model, tokenizer, testing_bernie)


#Few-shot 

def sample_few_shot_examples(train_data, k=6):
    # k ejemplos balanceados entre las tres clases
    return train_data.groupby("Stance").apply(
        lambda x: x.sample(k // 3, random_state=42)
    ).reset_index(drop=True)

def few_shot_prompt(row, examples):
    prompt = "<s>[INST] You are a stance detection system. Given a tweet and a target, classify the stance as FAVOR or AGAINST. Answer with only one word.\n\n"
    for _, ex in examples.iterrows():
        prompt += f"Tweet: {ex['Tweet']}\nTarget: {ex['Target']}\nStance: {ex['Stance']}\n\n"
    prompt += f"Tweet: {row['Tweet']}\nTarget: {row['Target']}\nStance: [/INST]"
    return prompt

def evaluate_few_shot(model, tokenizer, test_dataset, examples, device="cuda"):
    model.eval()
    predictions = []
    ground_truth = []

    for _, row in test_dataset.iterrows():
        prompt = few_shot_prompt(row, examples)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        predictions.append(parse_label(generated))
        ground_truth.append(row["Stance"])

    print(classification_report(ground_truth, predictions, zero_division=0))
    return predictions, ground_truth


few_shot_examples_trump  = sample_few_shot_examples(
    pd.concat([training_biden, training_bernie]), k=6
)


few_shot_examples_biden  = sample_few_shot_examples(
    pd.concat([training_trump, training_bernie]), k=6
)


few_shot_examples_bernie = sample_few_shot_examples(
    pd.concat([training_trump, training_biden]), k=6
)

#few shot with 3 examples
print("=== Few-shot k=3 | Trump ===")
few_shot_examples_trump_k3 = sample_few_shot_examples(
    pd.concat([training_biden, training_bernie]), k=3
)
evaluate_few_shot(model, tokenizer, testing_trump, few_shot_examples_trump_k3)

print("=== Few-shot k=3 | Biden ===")
few_shot_examples_biden_k3 = sample_few_shot_examples(
    pd.concat([training_trump, training_bernie]), k=3
)
evaluate_few_shot(model, tokenizer, testing_biden, few_shot_examples_biden_k3)

print("=== Few-shot k=3 | Bernie ===")
few_shot_examples_bernie_k3 = sample_few_shot_examples(
    pd.concat([training_trump, training_biden]), k=3
)
evaluate_few_shot(model, tokenizer, testing_bernie, few_shot_examples_bernie_k3)

#Few-shot with 6 examples
print("=== Few-shot k=6 | Trump ===")
evaluate_few_shot(model, tokenizer, testing_trump, few_shot_examples_trump)

print("=== Few-shot k=6 | Biden ===")
evaluate_few_shot(model, tokenizer, testing_biden, few_shot_examples_biden)

print("=== Few-shot k=6 | Bernie ===")
evaluate_few_shot(model, tokenizer, testing_bernie, few_shot_examples_bernie)

#Few shot with 9 examples

print("=== Few-shot k=9 | Trump ===")
few_shot_examples_trump_k9 = sample_few_shot_examples(
    pd.concat([training_biden, training_bernie]), k=9
)
evaluate_few_shot(model, tokenizer, testing_trump, few_shot_examples_trump_k9)

print("=== Few-shot k=9 | Biden ===")
few_shot_examples_biden_k9 = sample_few_shot_examples(
    pd.concat([training_trump, training_bernie]), k=9
)
evaluate_few_shot(model, tokenizer, testing_biden, few_shot_examples_biden_k9)

print("=== Few-shot k=9 | Bernie ===")
few_shot_examples_bernie_k9 = sample_few_shot_examples(
    pd.concat([training_trump, training_biden]), k=9
)
evaluate_few_shot(model, tokenizer, testing_bernie, few_shot_examples_bernie_k9)


#cleaning the memory before loading the quantized model
del model
torch.cuda.empty_cache()
import gc
gc.collect()
import time
time.sleep(5) 

#Fine-tuning 

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    quantization_config=bnb_config,
    device_map="auto",
)

# Prepare for QLoRA
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
tokenizer.padding_side = "right"

# Training
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    args=SFTConfig(
        output_dir="./qlora-stance",
        dataset_text_field="text",    
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        fp16=True,

        # checkpoints
        eval_strategy="epoch",
        save_strategy="epoch", 
        load_best_model_at_end=True,  
        metric_for_best_model="eval_loss",         
        save_total_limit=2,             
                                        
        
        # logging
        logging_dir="./logs",
        logging_steps=50,               
        report_to="none",               
        
        # reproducibility
        seed=42,
        
        # learning rate
        learning_rate=2e-4,             
        warmup_ratio=0.03,              
        lr_scheduler_type="cosine",
    ),
)

trainer.train()

trainer.save_model("./qlora-stance-trump")
tokenizer.save_pretrained("./qlora-stance-trump")

tokenizer.padding_side = "left"

print("=== Fine-tuned | Trump (in-target) ===")
evaluate(model, tokenizer, testing_trump)

print("=== Fine-tuned | Biden (cross-target) ===")
evaluate(model, tokenizer, testing_biden)

print("=== Fine-tuned | Bernie (cross-target) ===")
evaluate(model, tokenizer, testing_bernie)