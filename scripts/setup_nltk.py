from nltk.stem import SnowballStemmer

print("🔎 Проверяю Russian SnowballStemmer...")

stemmer = SnowballStemmer("russian")
result = stemmer.stem("государственный")

print(f"Тест: {result}")
print("✅ Russian SnowballStemmer работает")
