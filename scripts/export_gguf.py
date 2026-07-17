import os
# ОБМАНЫВАЕМ ПРОВЕРКУ SUDO И ИНТЕРНЕТА
os.environ["UNSLOTH_IS_OFFLINE"] = "1"
# ПРОБРАСЫВАЕМ ПУТЬ К ТВОЕМУ CMAKE
venv_bin = os.path.join(os.getcwd(), "venv", "bin")
os.environ["PATH"] = venv_bin + os.pathsep + os.environ["PATH"]

from unsloth import FastLanguageModel
import torch

# Используем полную версию для склейки, иначе будет 6ГБ мусора
base_model = "unsloth/Llama-3.2-3B-Instruct"  
# ПРАВИЛЬНЫЕ ПУТИ ДЛЯ V6
adapter_path = "./lora_fns_adapter_3b_v6"     # ← ИЗМЕНИЛ НА V6!
output_dir = "./model_for_ollama_3b_v6_gguf"   # ← ИЗМЕНИЛ ПАПКУ

print(f"🚀 Загружаем модель для СКЛЕЙКИ (в 16-bit)...")

# ЗАГРУЖАЕМ БЕЗ 4-BIT, чтобы склейка прошла честно!
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = adapter_path, # Загружаем сразу адаптер, Unsloth сам подтянет базу
    max_seq_length = 1024,
    load_in_4bit = False, # ВАЖНО: False для экспорта!
    torch_dtype = torch.float16,
    device_map = "cpu", # Экономим VRAM, склейка идет в оперативной памяти
)

print(f"✅ Модель и адаптер загружены")

print(f"🚀 Начинаем Merge и Конвертацию в GGUF (q4_k_m)...")
print(f"⏱️ Это займет 10-15 минут. Пей кофе...")

# Сохраняем со сжатием
model.save_pretrained_gguf(
    output_dir,
    tokenizer,
    quantization_method = "q4_k_m"  # 2.2 GB
)

print(f"✅ Готово! Модель сохранена в {output_dir}")
os.system(f"du -sh {output_dir}")