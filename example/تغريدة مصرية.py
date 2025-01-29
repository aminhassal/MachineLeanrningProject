import spacy

# تحميل اللغة العربية في spaCy
nlp = spacy.load('ar_core_news_sm')

# جملة للتحليل
sentence = 'جاءت الشركة شركة ليبيا للنفط من ليبيا.'

# تحليل الجملة باستخدام spaCy
doc = nlp(sentence)

# طباعة الكيانات المعترف بها
for entity in doc.ents:
    print(entity.text, entity.label_)

