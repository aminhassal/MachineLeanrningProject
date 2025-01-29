
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

print("Starting the script")

# تحميل وتحضير البيانات
data = pd.read_csv('libyan_tweets.csv')  # استبدل 'libyan_tweets.csv' بملف البيانات الخاص بك بلهجة ليبية
texts = data['tweet_text'].astype(str)
labels = data['label'].astype(str)

# تنظيف البيانات النصية
def clean_text(text):
    # إزالة الروابط
    text = re.sub(r'http\S+', '', text)
    # إزالة الأحرف الخاصة والأرقام
    text = re.sub(r'[^a-zA-ZÀ-ÿ]', ' ', text)
    # إزالة الفراغات الزائدة
    text = re.sub(r'\s+', ' ', text)
    # تحويل النص إلى حروف صغيرة
    text = text.lower()
    return text

texts = texts.apply(clean_text)

# تقسيم البيانات إلى مجموعات التدريب والاختبار
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# تطبيق الرموز على النصوص وملأ الفراغات
tokenizer = Tokenizer(num_words=10000)  # تحديد حجم المفردات
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)
train_data = pad_sequences(train_sequences)
test_data = pad_sequences(test_sequences, maxlen=train_data.shape[1])
... 
... # ترميز التصنيفات
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
test_labels = label_encoder.transform(test_labels)
... 
... # بناء النموذج
model = Sequential()
model.add(Embedding(10000, 100, input_length=train_data.shape[1]))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
... 
... # تدريب النموذج
model.fit(train_data, train_labels, validation_split=0.2, epochs=10, batch_size=32)
... 
... # تقييم النموذج
loss, accuracy = model.evaluate(test_data, test_labels)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)
... 
... # عمل التنبؤات
new_texts = ['تغريدة 1', 'تغريدة 2', 'تغريدة 3']  # استبدل بتغريدات جديدة بلهجة ليبية ترغب في التنبؤ بها
new_texts = pd.Series(new_texts)
new_texts = new_texts.apply(clean_text)
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_data = pad_sequences(new_sequences, maxlen=train_data.shape[1])
predictions = model.predict(new_data)
predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
