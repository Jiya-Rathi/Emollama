import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

# === Step 1: Load CounselChat from Hugging Face (Cloud Download)
def format_example(example):
    return {
        "instruction": example["context"],
        "output": example["response"]
    }

print("📥 Loading CounselChat dataset...")
dataset = load_dataset("counsel-chat")
formatted_dataset = dataset["train"].map(format_example)

# === Step 2: Setup 4-bit Quantization (BitsAndBytes)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# === Step 3: Load LLaMA-2-7B model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
print(f"📦 Loading {model_name} in 4-bit...")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# === Step 4: LoRA Configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, peft_config)

# === Confirm Everything Works
print("\n✅ Model loaded and LoRA applied.")
print(f"Total samples in CounselChat: {len(formatted_dataset)}")
print("🔍 Sample:")
print("Instruction:", formatted_dataset[0]["instruction"])
print("Response:", formatted_dataset[0]["output"])
