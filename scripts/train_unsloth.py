import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# --- ПАРАМЕТРЫ V6 (РЕЦЕПТ ОТ OVERFITTING) ---
max_seq_length = 512  ### Увеличим до 512, чтобы модель видела более длинные связи
model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
lora_rank = 16 
output_dir = "./lora_fns_adapter_3b_v6" ### НОВАЯ ПАПКА ДЛЯ V6

print(f"🚀 Загружаем Llama 3.2 3B...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    device_map = "auto",
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32, ### УВЕЛИЧИЛИ ДО 32 (Rank * 2) — это сделает адаптер сильнее
    lora_dropout = 0.05, ### ДОБАВИЛИ DROPOUT 5% — это заставит модель НЕ зубрить (защита от циклов)
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

print("📚 Загружаем и готовим датасет...")
dataset = load_dataset('json', data_files='fns_100_examples.jsonl', split='train')

def format_func(example):
    text = tokenizer.apply_chat_template(example['messages'], tokenize=False, add_generation_prompt=False)
    return {"text": text}

dataset = dataset.map(format_func)

# --- НАСТРОЙКИ С ЖЕСТКОЙ РЕГУЛЯРИЗАЦИЕЙ ---
training_args = TrainingArguments(
    output_dir = output_dir,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_steps = 20, ### УВЕЛИЧИЛИ РАЗОГРЕВ ДО 20 — чтобы веса не "шокировались" в начале
    num_train_epochs = 3, ### 3 ЭПОХИ — золотая середина для 322 примеров
    learning_rate = 3e-5, ### СНИЗИЛИ LR ДО 3e-5 — ювелирная точность обучения
    fp16 = True,
    logging_steps = 1,
    save_strategy = "no",
    optim = "adamw_8bit",
    weight_decay = 0.1, ### УВЕЛИЧИЛИ В 10 РАЗ! — главная защита от галлюцинаций (санитарного надзора)
    lr_scheduler_type = "cosine",
    seed = 3407,
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = training_args,
    packing = False, 
)

print("🏋️ Начинаем финальное обучение v6...")
torch.cuda.empty_cache()

trainer.train()

print("💾 Сохраняем финальный адаптер v6...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("✅ ГОТОВО! Модель v6 готова к мерджу.")
