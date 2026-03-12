from unsloth import FastLanguageModel
import torch

# КРИТИЧЕСКИ ВАЖНО: Мы учим 3.2-3B, а не 8B!
model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
adapter_path = "./lora_fns_adapter_3b_v4" # Твой свежий адаптер

print("🚀 Загружаем экспертную модель v4 (3.2-3B)...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 256,
    load_in_4bit = True,
)

# Подключаем твой свежий адаптер v4
model.load_adapter(adapter_path)

# ВКЛЮЧАЕМ РЕЖИМ ИНФЕРЕНСА (ускоряет генерацию в 2 раза)
FastLanguageModel.for_inference(model)

def ask(question):
    messages = [{"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True, # Обязательно, чтобы модель поняла, что пора отвечать
        return_tensors="pt"
    ).to("cuda")
    
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=100,
        temperature=0.2,            # Делаем еще строже
        repetition_penalty=1.2,    # ВОТ ОНО! Запрещаем повторять слова
        top_p=0.9,                 # Ограничиваем выбор только логичными словами
        do_sample=True,            # Разрешаем чуть-чуть думать
    )
    
    # Декодируем только новые токены (ответ), чтобы не дублировать вопрос
    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return response.strip()

print("\n🔍 Тестовый прогон v4:")
test_questions = [
    "Какие документы регулируют награждение в ФНС?",
    "Кто подписывает наградные материалы?",
    "наргады" # Специально с опечаткой, чтобы проверить "мозги" модели
]

for q in test_questions:
    print(f"\n❓ {q}")
    print(f"🤖 {ask(q)}")
