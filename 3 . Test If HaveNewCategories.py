import pandas as pd
from joblib import load

# 1. تحميل النموذج، الـ Vectorizer، والـ Label Encoder
model = load("naive_bayes_model.joblib")
vectorizer = load("vectorizer.joblib")
label_encoder = load("label_encoder.joblib")

# 2. تحميل الكلمات الجديدة من ملف Excel
new_words_file_path = "new_words.xlsx"  # مسار ملف Excel للكلمات الجديدة
new_words_data = pd.read_excel(new_words_file_path)

# افتراض أن العمود يحتوي على الكلمات الجديدة اسمه 'term'
new_terms = new_words_data['term']

# 3. تحويل النصوص إلى تمثيل عددي باستخدام الـ Vectorizer
X_new = vectorizer.transform(new_terms)

# 4. تصنيف الكلمات
predictions = model.predict(X_new)

# 5. تحويل التصنيفات من أرقام إلى نصوص
predicted_categories = label_encoder.inverse_transform(predictions)

# 6. إضافة التصنيفات إلى ملف Excel
new_words_data['classification'] = predicted_categories

# 7. حفظ النتائج إلى ملف جديد
output_file_path = "classified_words.xlsx"
new_words_data.to_excel(output_file_path, index=False)

print(f"تم حفظ الكلمات المصنفة في: {output_file_path}")


# تفاصيل الخطوات:
# تحميل الكلمات الجديدة من Excel:

# يتم افتراض وجود عمود باسم term يحتوي على الكلمات التي ترغب في تصنيفها.
# تصنيف الكلمات:

# يتم استخدام النموذج المدرب لتحويل الكلمات الجديدة إلى تصنيفات.
# إضافة التصنيفات إلى ملف Excel:

# يتم إضافة التصنيفات كعمود جديد في ملف الكلمات.
# حفظ النتائج:

# يتم حفظ ملف Excel جديد يحتوي على الكلمات مع التصنيفات الخاصة بها.

# -------------------------------------

# متطلبات:
# ملف Excel يحتوي على عمود term (الكلمات الجديدة للتصنيف).
# ملفات النموذج (naive_bayes_model.joblib, vectorizer.joblib, label_encoder.joblib) محفوظة مسبقًا.
# الناتج:
# ملف Excel جديد (classified_words.xlsx) يحتوي على الكلمات مع التصنيفات التي تم استخراجها.
