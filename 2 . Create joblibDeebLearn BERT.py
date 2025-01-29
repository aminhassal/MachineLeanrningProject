import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers.legacy import Adam
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

# 4. تحميل Tokenizer الخاص بـ BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
X_encoded = tokenizer(terms, padding=True, truncation=True, return_tensors="tf", max_length=128)

# 5. تحميل نموذج BERT للتصنيف
model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=n_classes)

# 6. تجميع النموذج
optimizer = Adam(learning_rate=2e-5)
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# 7. تدريب النموذج
model.fit(dict(X_encoded), y, epochs=3, batch_size=16, validation_split=0.2)

# 8. حفظ النموذج والأدوات المساعدة
model_file_path = "C:\\py\\Last\\bert_model"
tokenizer_file_path = "C:\\py\\Last\\bert_tokenizer.joblib"
label_encoder_file_path = "C:\\py\\Last\\label_encoder.joblib"

model.save_pretrained(model_file_path)
dump(tokenizer, tokenizer_file_path)
dump(label_encoder, label_encoder_file_path)

print(f"تم حفظ النموذج في: {model_file_path}")
print(f"تم حفظ الـ Tokenizer في: {tokenizer_file_path}")
print(f"تم حفظ الـ Label Encoder في: {label_encoder_file_path}")
