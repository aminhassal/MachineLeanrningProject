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

# 4. تحميل Tokenizer الخاص بـ ARBERT
tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/ARBERT")

# 5. تحويل النصوص إلى تنسيق يمكن للنموذج فهمه
X_encoded = tokenizer(terms, padding=True, truncation=True, return_tensors="tf", max_length=128)

# 6. تحميل نموذج ARBERT للتصنيف
model = TFAutoModelForSequenceClassification.from_pretrained("UBC-NLP/ARBERT", num_labels=n_classes)

# 7. تجميع النموذج
optimizer = Adam(learning_rate=2e-5)
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# 8. تدريب النموذج
model.fit(dict(X_encoded), y, epochs=3, batch_size=16, validation_split=0.2)

# 9. حفظ النموذج والأدوات المساعدة
model_file_path = "C:\\py\\Last\\ARBERT_model"
tokenizer_file_path = "C:\\py\\Last\\ARBERT_tokenizer.joblib"
label_encoder_file_path = "C:\\py\\Last\\label_encoder.joblib"

model.save_pretrained(model_file_path)
dump(tokenizer, tokenizer_file_path)
dump(label_encoder, label_encoder_file_path)

print(f"تم حفظ النموذج في: {model_file_path}")
print(f"تم حفظ الـ Tokenizer في: {tokenizer_file_path}")
print(f"تم حفظ الـ Label Encoder في: {label_encoder_file_path}")

# هذا الكود يستخدم نموذج ARBERT بدلاً من BERT. لاحظ أنك تحتاج إلى تحميل نموذج ARBERT من خلال مكتبة Hugging Face.

# عم، يمكن تصنيف الكود السابق على أنه تدريب عميق (Deep Learning) لعدة أسباب:
# أسباب تصنيف الكود على أنه تدريب عميق
# استخدام نموذج BERT/ARBERT: نموذج BERT/ARBERT هو نموذج تعلم عميق يعتمد على الشبكات العصبية الاصطناعية.
# استخدام الشبكات العصبية الاصطناعية: الكود يستخدم الشبكات العصبية الاصطناعية لتحليل النصوص وتصنيفها.
# استخدام تقنيات التعلم الآلي: الكود يستخدم تقنيات التعلم الآلي مثل التعلم الإشرافي (Supervised Learning) لتدريب النموذج.
# استخدام المكتبات الخاصة بالتعلم العميق: الكود يستخدم مكتبات خاصة بالتعلم العميق مثل TensorFlow وTransformers.
# خصائص التدريب العميق في الكود
# التعلم الإشرافي: الكود يستخدم التعلم الإشرافي لتدريب النموذج على البيانات المسمى.
# الشبكات العصبية الاصطناعية: الكود يستخدم الشبكات العصبية الاصطناعية لتحليل النصوص وتصنيفها.
# التعلم العميق: الكود يستخدم التعلم العميق لتحليل النصوص وتصنيفها.
# المكتبات الخاصة بالتعلم العميق: الكود يستخدم مكتبات خاصة بالتعلم العميق مثل TensorFlow وTransformers.
# فوائد التدريب العميق في الكود
# تحسين دقة التصنيف: التدريب العميق يمكن أن يحسن دقة التصنيف للنصوص.
# تحليل النصوص بشكل أفضل: التدريب العميق يمكن أن يحلل النصوص بشكل أفضل وتحديد المعاني والسياقات بشكل أفضل.
# تطوير نماذج أكثر دقة: التدريب العميق يمكن أن يطور نماذج أكثر دقة وتحسين أداء النموذج.
