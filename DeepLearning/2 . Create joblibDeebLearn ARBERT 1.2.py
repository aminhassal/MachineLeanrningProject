import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers.legacy import Adam # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore
from joblib import dump, load
from sklearn.metrics import classification_report
import os

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

# 4. تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(terms, y, test_size=0.2, random_state=42)

# 5. تحميل Tokenizer الخاص بـ AraBERT
model_name = "aubmindlab/bert-base-arabertv02"  # نموذج AraBERT
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 6. تحويل النصوص إلى تنسيق AraBERT
def encode_texts(texts, tokenizer, max_length=128):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="tf", max_length=max_length)

X_train_encoded = encode_texts(X_train, tokenizer)
X_test_encoded = encode_texts(X_test, tokenizer)

# 7. تحميل النموذج الحالي إذا كان موجودًا، أو إنشاء نموذج جديد
model_path = "C:\\py\\Last\\arabert_model"
if os.path.exists(model_path):
    print("تم العثور على نموذج موجود. يتم تحميله...")
    model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
    # تحميل Label Encoder الحالي
    label_encoder = load("C:\\py\\Last\\label_encoder.joblib")
    n_classes = len(label_encoder.classes_)
else:
    print("لم يتم العثور على نموذج موجود. يتم إنشاء نموذج جديد...")
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_classes)

# 8. تجميع النموذج
optimizer = Adam(learning_rate=2e-5)
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# 9. إضافة Callbacks
checkpoint = ModelCheckpoint(
    "C:\\py\\Last\\best_arabert_model",  # مسار لحفظ أفضل نموذج
    monitor="val_accuracy",  # مراقبة دقة التحقق
    save_best_only=True,  # حفظ أفضل نموذج فقط
    mode="max",  # تعظيم الدقة
)

early_stopping = EarlyStopping(
    monitor="val_accuracy",  # مراقبة دقة التحقق
    patience=2,  # عدد المرات المسموح بها بعدم التحسن
    mode="max",  # تعظيم الدقة
)

# 10. تدريب النموذج
history = model.fit(
    dict(X_train_encoded), y_train,
    epochs=5,  # زيادة عدد الـ epochs
    batch_size=16,
    validation_data=(dict(X_test_encoded), y_test),
    callbacks=[checkpoint, early_stopping]  # استخدام Callbacks
)

# 11. تقييم النموذج
y_pred = model.predict(dict(X_test_encoded))
y_pred_labels = np.argmax(y_pred.logits, axis=1)

print("تقرير التصنيف:")
print(classification_report(y_test, y_pred_labels, target_names=label_encoder.classes_))

# 12. حفظ النموذج والأدوات المساعدة
model_file_path = "C:\\py\\Last\\arabert_model"
tokenizer_file_path = "C:\\py\\Last\\arabert_tokenizer.joblib"
label_encoder_file_path = "C:\\py\\Last\\label_encoder.joblib"

model.save_pretrained(model_file_path)
dump(tokenizer, tokenizer_file_path)
dump(label_encoder, label_encoder_file_path)

print(f"تم حفظ النموذج في: {model_file_path}")
print(f"تم حفظ الـ Tokenizer في: {tokenizer_file_path}")
print(f"تم حفظ الـ Label Encoder في: {label_encoder_file_path}")