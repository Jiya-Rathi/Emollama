import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import login
import os

# === Authentication
print("🔐 Logging into Hugging Face...")
login()

# === Step 1: Load and format datasets
def format_counsel_chat(example):
    return {
        "instruction": example['questionText'],
        "output": example["answerText"]
    }

def format_amod_dataset(example):
    return {
        "instruction": example['Context'],
        "output": example['Response']
    }

print("📥 Loading nbertagnolli/counsel-chat...")
counsel_chat = load_dataset("nbertagnolli/counsel-chat")["train"].map(format_counsel_chat)

print("📥 Loading Amod/mental_health_counseling_conversations...")
amod = load_dataset("Amod/mental_health_counseling_conversations")["train"].map(format_amod_dataset)

# === Step 2: Combine and shuffle
print("🔗 Combining and shuffling datasets...")
combined_dataset = concatenate_datasets([counsel_chat, amod]).shuffle(seed=42)

# === Step 3: Save combined dataset to JSONL
save_path = "combined_mental_health_dataset.jsonl"
print(f"💾 Saving combined dataset to: {save_path}")
combined_dataset.to_json(save_path, orient="records", lines=True)

# === Step 4: Load LLaMA-2 with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_name = "meta-llama/Llama-2-7b-hf"
print(f"📦 Loading {model_name} in 4-bit...")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # Prevent tokenizer issues during training

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# === Step 5: Apply LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, peft_config)

# === Summary Output
print("\n✅ Setup complete!")
print(f"Total combined samples: {len(combined_dataset)}")
print(f"- CounselChat samples: {len(counsel_chat)}")
print(f"- Amod samples: {len(amod)}")

# Sample Preview
print("\n🔍 Sample entry:")
print("Instruction:", combined_dataset[0]["instruction"])
print("Response:", combined_dataset[0]["output"])
