import spacy
from spacy.pipeline import EntityRecognizer
import os
import pandas as pd

# تحميل نموذج التعرف على الكيانات المسماة باللهجة الليبية
libyanlang_model.xlsx = spacy.load('libyanlang_model.xlsx')

# جملة للتحليل
text = "أنا من مدينة طرابلس وأعمل في شركة النفط الليبية."

# تحليل الجملة وتحديد الكيانات المسماة
doc = libyanlang_model.xlsx(text)

# طباعة الكيانات المسماة
for entity in doc.ents:
    print(f"{entity.text} ({entity.label_})")

# النتيجة:
# طرابلس (LOC)
# شركة النفط الليبية (ORG)
