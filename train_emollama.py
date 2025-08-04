import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import login

# === Step 0: Hugging Face login (optional)
print("🔐 Logging into Hugging Face...")
login()  # Replace if needed

# === Step 1: Load cleaned JSONL dataset
print("📥 Loading dataset...")
dataset = load_dataset("json", data_files="data1_cleaned.jsonl", split="train")

# === Step 2: Load tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Fix tokenizer padding setup
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# === Step 3: Improved tokenization function
def tokenize(examples):
    """Tokenize examples with proper formatting and consistent padding"""
    # Handle both single examples and batches
    if isinstance(examples['instruction'], list):
        # Batch processing
        prompts = [
            f"### Instruction:\n{inst}\n\n### Response:\n{out}"
            for inst, out in zip(examples['instruction'], examples['output'])
        ]
    else:
        # Single example
        prompts = f"### Instruction:\n{examples['instruction']}\n\n### Response:\n{examples['output']}"
    
    # Tokenize with consistent parameters
    tokenized = tokenizer(
        prompts,
        padding=True,  # Changed from "max_length" to True for dynamic padding
        truncation=True,
        max_length=512,
        return_tensors=None  # Keep as lists for dataset processing
    )
    
    # Ensure labels are set (copy input_ids for causal LM)
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

print("🔄 Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize, 
    batched=True, 
    remove_columns=dataset.column_names,
    desc="Tokenizing"
)

# === Step 4: Load model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

print("🤖 Loading LLaMA-2 7B model in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# === Step 5: Apply LoRA BEFORE gradient checkpointing
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

model = get_peft_model(model, peft_config)

# Enable gradient checkpointing AFTER applying LoRA
if hasattr(model, 'enable_input_require_grads'):
    model.enable_input_require_grads()
else:
    # Fallback for older versions
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Ensure LoRA parameters require gradients
for name, param in model.named_parameters():
    if 'lora' in name.lower():
        param.requires_grad = True

model.print_trainable_parameters()

# === Step 6: Prepare TrainingArguments with better settings
training_args = TrainingArguments(
    output_dir="emollama-lora-checkpoints",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    report_to="none",
    dataloader_drop_last=True,
    group_by_length=True,
    warmup_steps=50,
    max_grad_norm=1.0,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    # Additional stability settings
    dataloader_num_workers=0,  # Disable multiprocessing
    save_safetensors=True,
)

# === Step 7: Data collator with proper padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False,
    pad_to_multiple_of=8  # Pad to multiple of 8 for efficiency
)

# === Step 8: Standard trainer (custom trainer not needed)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    processing_class=tokenizer,
    data_collator=data_collator
)

# === Step 9: Verify model setup before training
print("🔍 Verifying model setup...")
print(f"Model device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")

# Check trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

# Verify some parameters require gradients
if trainable_params == 0:
    raise ValueError("No trainable parameters found! LoRA setup failed.")

print("🚀 Starting training...")
trainer.train()

# === Step 10: Save adapter
print("💾 Saving LoRA adapter...")
model.save_pretrained("emollama-lora-adapter")
tokenizer.save_pretrained("emollama-lora-adapter")

print("✅ Training complete and adapter saved!")