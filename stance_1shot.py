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

#One-shot 

def sample_one_shot_example(train_data):
    return train_data.sample(1, random_state=42)

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


few_shot_examples_trump  = sample_one_shot_example(
    pd.concat([training_biden, training_bernie])
)


few_shot_examples_biden  = sample_one_shot_example(
    pd.concat([training_trump, training_bernie])
)


few_shot_examples_bernie = sample_one_shot_example(
    pd.concat([training_trump, training_biden])
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    device_map="auto",
    torch_dtype=torch.float16
)

print("Trump Evaluation")
evaluate_few_shot(model, tokenizer, testing_trump, few_shot_examples_trump)

print("Biden Evaluation")
evaluate_few_shot(model, tokenizer, testing_biden, few_shot_examples_biden)

print("Bernie Evaluation")
evaluate_few_shot(model, tokenizer, testing_bernie, few_shot_examples_bernie)