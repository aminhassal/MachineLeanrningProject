import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from joblib import dump

# 1. تحميل القاموس من ملف Excel
dictionary_file_path = "C:\\py\\Last\\split_words.xlsx"  # مسار ملف القاموس
dictionary_data = pd.read_excel(dictionary_file_path)

# 2. استخراج الكلمات والتصنيفات
terms = dictionary_data['term'].astype(str).tolist()
categories = dictionary_data['classification'].astype(str).tolist()

# 3. تحويل التصنيفات إلى أرقام
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(categories)
n_classes = len(label_encoder.classes_)

# 4. تحويل النصوص إلى تمثيلات عددية باستخدام Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(terms)
X_sequences = tokenizer.texts_to_sequences(terms)
vocab_size = len(tokenizer.word_index) + 1  # حجم المفردات

# 5. ضبط طول التسلسلات (Padding)
max_length = max(len(seq) for seq in X_sequences)
X_padded = pad_sequences(X_sequences, maxlen=max_length, padding='post')

# 6. بناء نموذج LSTM
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(n_classes, activation='softmax')  # إخراج مصنف متعدد الفئات
])

# 7. تجميع النموذج
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 8. تدريب النموذج
model.fit(X_padded, y, epochs=10, batch_size=16, validation_split=0.2)

# 9. حفظ النموذج والأدوات المساعدة
model_file_path = "C:\\py\\Last\\lstm_model.h5"
tokenizer_file_path = "C:\\py\\Last\\tokenizer.joblib"
label_encoder_file_path = "C:\\py\\Last\\label_encoder.joblib"

model.save(model_file_path)
dump(tokenizer, tokenizer_file_path)
dump(label_encoder, label_encoder_file_path)

print(f"تم حفظ النموذج في: {model_file_path}")
print(f"تم حفظ الـ Tokenizer في: {tokenizer_file_path}")
print(f"تم حفظ الـ Label Encoder في: {label_encoder_file_path}")
