# استيراد المكتبات الضرورية
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# تحميل مجموعة البيانات من ملف إكسل
data = pd.read_excel('path_to_your_excel_file.xlsx')

# مثال على بنية البيانات في ملف إكسل:
# العمود الأول: 'Word' - الكلمات في النص
# العمود الثاني: 'Tag' - العلامات المرتبطة بالكيانات المسماة (مثل B-PER, I-PER, O)

# تحويل الكلمات إلى قائمة وإنشاء قاموس للعلمات (التاغات)
words = list(set(data['Word'].values))
tags = list(set(data['Tag'].values))

# إضافة كلمات padding لقائمة الكلمات والعلامات
words.append("PAD")
tags.append("PAD")

# إنشاء قواميس لتخزين الكلمات والعلامات بالأرقام
word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
idx2tag = {i: t for t, i in tag2idx.items()}

# تحضير البيانات لتكون مناسبة للإدخال في LSTM
X = [[word2idx[w] for w in s] for s in data.groupby(['Sentence #'])['Word'].apply(list)]
y = [[tag2idx[t] for t in s] for s in data.groupby(['Sentence #'])['Tag'].apply(list)]

# تطبيق التجزئة وتوسيع التسلسلات لتكون بنفس الطول
X = pad_sequences(X, maxlen=50, padding='post')
y = pad_sequences(y, maxlen=50, padding='post')

# تقسيم البيانات إلى مجموعة تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# تحميل النموذج والمُرمِّز الجاهز لـ BERT
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')

# تعريف معمارية النموذج باستخدام BERT و BiLSTM
input_ids = Input(shape=(50,), dtype=tf.int32)
attention_mask = Input(shape=(50,), dtype=tf.int32)

# استخدام مخرجات BERT كمدخلات
bert_output = bert_model(input_ids, attention_mask=attention_mask)[0]

# استخدام LSTM ثنائي الاتجاه لتحليل التسلسل
lstm_layer = Bidirectional(LSTM(units=64, return_sequences=True))(bert_output)
output = TimeDistributed(Dense(len(tags), activation="softmax"))(lstm_layer)

# بناء النموذج وتحديد وظيفة الفقد والمقياس المستخدم
model = Model(inputs=[input_ids, attention_mask], outputs=output)
model.compile(optimizer=Adam(learning_rate=3e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# عرض ملخص للنموذج
model.summary()

# دالة لتحضير المدخلات للنموذج باستخدام BERT
def encode_sentences(sentences, tokenizer, max_len=50):
    input_ids = []
    attention_masks = []
    for sent in sentences:
        bert_inp = tokenizer.encode_plus(sent, max_length=max_len, padding='max_length', truncation=True, return_attention_mask=True)
        input_ids.append(bert_inp['input_ids'])
        attention_masks.append(bert_inp['attention_mask'])
    return np.array(input_ids), np.array(attention_masks)

# تحويل الكلمات إلى جمل لاستخدامها في BERT
sentences = data.groupby(['Sentence #'])['Word'].apply(lambda x: ' '.join(x)).values
input_ids, attention_masks = encode_sentences(sentences, bert_tokenizer)

# تدريب النموذج
history = model.fit([input_ids, attention_masks], np.array(y_train), validation_data=([X_test, y_test]), epochs=3, batch_size=32)

# تقييم النموذج
loss, accuracy = model.evaluate([X_test, attention_masks], np.array(y_test))
print(f'Loss: {loss}, Accuracy: {accuracy}')
