# 🧠 Emollama: Emotionally Intelligent LLaMA-2 Chatbot

**Emollama** is a fine-tuned version of `LLaMA-2-7B-Chat` that aims to generate emotionally aware, supportive, and conversationally intelligent responses. The model is trained in two phases using curriculum learning principles on:

- 🩺 [`CounselChat`](https://huggingface.co/datasets/counsel-chat) — to instill therapeutic tone and emotional understanding
- 🤖 [`nart-100k-synthetic-buddy-mixed-names`](https://huggingface.co/datasets/victunes/nart-100k-synthetic-buddy-mixed-names) — to enhance conversational fluency and persona-based replies

---

## 🚀 Project Goals

- Build an emotionally intelligent chatbot using `LLaMA-2-7B-Chat`
- Incorporate empathy and emotional support in conversations
- Leverage LoRA and BitsAndBytes for efficient fine-tuning
- Apply **curriculum learning**: start with emotional grounding, then train for conversational diversity

---

## 🧰 Tech Stack

- 🤗 Transformers + PEFT + Datasets
- 🦙 `LLaMA-2-7B-Chat` (Meta)
- 🔍 LoRA (`r=8`, `alpha=16`, `dropout=0.05–0.1`)
- ⚖️ BitsAndBytes (4-bit quantization for memory efficiency)
- 🔧 Accelerate, WandB, and PyTorch Lightning (optional)
- 🐳 Docker / EC2 / Vast.ai (deployment options)

---

## 🏗️ Training Pipeline

```text
1. Prepare environment (conda/venv, CUDA, HuggingFace cache, etc.)
2. Load datasets:
   - CounselChat → Instruction-Output pairs
   - NART-100k → Synthetic buddy dialog pairs
3. Preprocess:
   - Tokenization (HF tokenizer)
   - Instruction tuning format
4. Phase 1: Train on CounselChat (LoRA + 4bit)
5. Phase 2: Train on NART-100k for conversational diversity
6. Evaluate + Save + Push to Hub (optional)
