import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from joblib import dump

#----------------------------------------------------------------------------

#لإنشاء نموذج خاص بك باستخدام قاموس يدوي 
#  (كما هو الحال مع ملف اكسل الذي يحتوي على الكلمات وتصنيفاتها)


#----------------------------------------------------------------------------

# 1. تحميل القاموس من ملف Excel
dictionary_file_path = "C:\\py\\Last\\split_words.xlsx"  # مسار ملف القاموس
dictionary_data = pd.read_excel(dictionary_file_path)

# 2. استخراج الكلمات والتصنيفات
terms = dictionary_data['term']
categories = dictionary_data['classification']

# 3. تحويل النصوص إلى تمثيلات عددية باستخدام CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(terms)

# 4. ترميز التصنيفات إلى أرقام
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(categories)

# 5. تدريب نموذج Naive Bayes
model = MultinomialNB()
model.fit(X, y)

# 6. حفظ النموذج والـ Vectorizer والـ Label Encoder لاستخدامها لاحقًا
model_file_path = "C:\\py\\Last\\naive_bayes_model.joblib"
vectorizer_file_path = "C:\\py\\Last\\vectorizer.joblib"
label_encoder_file_path = "C:\\py\\Last\\label_encoder.joblib"

dump(model, model_file_path)
dump(vectorizer, vectorizer_file_path)
dump(label_encoder, label_encoder_file_path)

print(f"تم حفظ النموذج في: {model_file_path}")
print(f"تم حفظ الـ Vectorizer في: {vectorizer_file_path}")
print(f"تم حفظ الـ Label Encoder في: {label_encoder_file_path}")
