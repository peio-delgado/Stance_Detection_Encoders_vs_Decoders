# Stance Detection with Mistral-7B + QLoRA

This project fine-tunes and evaluates Mistral-7B-Instruct-v0.3 for stance detection on the PStance dataset. It supports:

- Zero-shot evaluation
- Few-shot prompting
- QLoRA fine-tuning
- Cross-target evaluation on Trump, Biden, and Bernie

The task is binary stance classification:

- `FAVOR`
- `AGAINST`

---

# Project Structure

```text
stance_project/
├── stance_encoder.py
├── stance_1shot.py
├── AppsII_code.py
├── run_stance_encoder.sh
├── run_stance_one.sh
├── run_appsii.sh
├── PStance data/
│   ├── raw_train_trump.csv
│   ├── raw_val_trump.csv
│   ├── raw_test_trump.csv
│   ├── raw_train_biden.csv
│   ├── raw_test_biden.csv
│   ├── raw_train_bernie.csv
│   └── raw_test_bernie.csv
├── qlora-stance/
├── qlora-stance-trump/
├── logs/
└── checkpoints/
```

---

# Requirements

Install the required libraries:

```bash
pip install torch transformers datasets peft trl bitsandbytes scikit-learn pandas numpy matplotlib seaborn
```

A CUDA-compatible GPU is strongly recommended for inference and fine-tuning.

---

# Environment Setup

The scripts expect a Hugging Face cache location and a Python virtual environment.

```bash
export HF_HOME="/home/pdelgado010/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/pdelgado010/.cache/huggingface"

source /home/pdelgado010/envs/my_venv/bin/activate
```

---

# Data Format

Each CSV file should contain the following columns:

- `Tweet`
- `Target`
- `Stance`

Example:

```csv
Tweet,Target,Stance
"I support this policy",Trump,FAVOR
"This is terrible",Biden,AGAINST
```

---

# Methodology

## 1. Data Loading

The scripts load train, validation, and test splits for:

- Trump
- Biden
- Bernie

using pandas and Hugging Face datasets.

---

## 2. Prompt Formatting

Each sample is converted into a prompt format like:

```text
Tweet: ...
Target: ...
Stance: ...
```

For inference, the model is instructed to answer with only one label.

---

## 3. Label Parsing

Generated text is normalized and mapped as follows:

- `FAVOR` → if the generated text contains the word `FAVOR`
- `AGAINST` → otherwise

---

# Evaluation Modes

## Zero-shot Evaluation

The model is evaluated directly without any in-context examples.

---

## Few-shot Evaluation

The model receives a small number of labeled examples before predicting the target example.

The experiments evaluate:

- `k = 3`
- `k = 6`
- `k = 9`

Few-shot examples are sampled from different political targets to test cross-target generalization.

---

## QLoRA Fine-tuning

The project fine-tunes Mistral-7B using:

- 4-bit quantization
- LoRA adapters
- `trl.SFTTrainer`

### Training Configuration

```python
num_train_epochs=3
per_device_train_batch_size=4
gradient_accumulation_steps=4
learning_rate=2e-4
warmup_ratio=0.03
lr_scheduler_type="cosine"
fp16=True
```

---

# Running the Project

## Run the Main Script

```bash
python stance_encoder.py
```

---

# SLURM Execution

The repository includes SLURM batch scripts for cluster execution.

## Submit Zero-shot / Fine-tuning Job

```bash
sbatch run_stance_encoder.sh
```

## Submit One-shot Evaluation

```bash
sbatch run_stance_one.sh
```

## Submit Additional Experiment

```bash
sbatch run_appsii.sh
```

---

# Outputs

The scripts generate:

- Classification reports
- Training logs
- Fine-tuned checkpoints

Saved outputs include:

```text
./qlora-stance
./qlora-stance-trump
./logs
```

---

# Experimental Results

# Zero-shot Results

| Target | Accuracy |
|---|---|
| Trump | ~0.70 |
| Biden | ~0.83 |
| Bernie | ~0.78 |

---

# Few-shot Results

Few-shot prompting improved performance in several cases, especially for the `AGAINST` class recall, though performance varied depending on the number of examples and target domain.

---

# Fine-tuned QLoRA Results

| Evaluation | Accuracy |
|---|---|
| Trump (in-target) | ~0.85 |
| Biden (cross-target) | ~0.79 |
| Bernie (cross-target) | ~0.75 |

---

# Notes

- The setup performs binary stance classification only.
- `parse_label()` defaults to `AGAINST` unless the output explicitly contains `FAVOR`.
- Training is performed on Trump data while Biden and Bernie are used for cross-target evaluation.
- The tokenizer uses:
  - left padding for inference
  - right padding during training

---

# Reproducibility

The experiments use:

```python
seed = 42
```

to improve reproducibility.

---

# License

Add your preferred license here.

---

# Author

Add author name and contact information here.
