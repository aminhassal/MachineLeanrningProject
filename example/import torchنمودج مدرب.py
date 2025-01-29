import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# نموذج بسيط لشبكة عصبية
class NERModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(NERModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

# إعداد البيانات (مثال بسيط)
class NERDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]

# إعداد البيانات (عينة بسيطة)
sentences = [["مشيت", "لطرابلس"], ["نقيت", "صديقي", "علي"]]
labels = [[0, 1], [0, 2, 1]]  # 0: O, 1: LOC, 2: PER

# تحويل البيانات إلى Tensores
vocab = {"مشيت": 0, "لطرابلس": 1, "نقيت": 2, "صديقي": 3, "علي": 4}
sentences_tensor = [[vocab[word] for word in sentence] for sentence in sentences]
labels_tensor = [[label for label in label_list] for label_list in labels]

# دالة للقيام بتجميع البيانات مع الحشو 
def collate_fn(batch):
    sentences, labels = zip(*batch)
    # تحويل الجمل إلى tensores مع الحشو 
    sentences = [torch.tensor(sentence) for sentence in sentences]
    sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
    # تحويل التصنيفات إلى tensores مع الحشو
    labels = [torch.tensor(label) for label in labels]
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    return sentences, labels

# إنشاء مجموعة بيانات و DataLoader
dataset = NERDataset(sentences_tensor, labels_tensor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# إعداد النموذج
model = NERModel(vocab_size=len(vocab), embedding_dim=10, hidden_dim=20, output_dim=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# تدريب النموذج
for epoch in range(10):
    for batch_sentences, batch_labels in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_sentences)
        loss = criterion(outputs.view(-1, 3), batch_labels.view(-1))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# نموذجك الآن مدرب ويمكنك استخدامه للتنبؤ بالكيانات