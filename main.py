## 1. 사용할 패키지 불러오기
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

## 2. GPU 환경 설정
if torch.cuda.is_available():
    print("{} GPU를 사용합니다.".format(torch.cuda.get_device_name(0)))
    device = torch.device("cuda:0")
else: 
    print("GPU 사용이 불가능합니다.")
    device = torch.device("cpu")

## 3. Pretrained 된 KoBERT 모델 불러오기.
bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")

## 4. Dataset 불러오기
# (1) 원천 데이터셋 불러오기
dataset_train = pd.read_excel('data/train.xlsx')
dataset_train['감정_대분류'][dataset_train['감정_대분류'] == '기쁨 '] = '기쁨'
dataset_train['감정_대분류'][dataset_train['감정_대분류'] == '분노 '] = '분노'
dataset_train['감정_대분류'][dataset_train['감정_대분류'] == '불안 '] = '불안'

dataset_test = pd.read_excel('data/test.xlsx')
dataset_test['감정_대분류'][dataset_test['감정_대분류'] == '기쁨 '] = '기쁨'
dataset_test['감정_대분류'][dataset_test['감정_대분류'] == '분노 '] = '분노'
dataset_test['감정_대분류'][dataset_test['감정_대분류'] == '불안 '] = '불안'


# (2) Text와 label 정보 추출하기
label_col = '감정_대분류'
text_col = '사람문장1'
le = LabelEncoder()
le = le.fit(dataset_train[label_col])
dataset_train[label_col] = le.transform(dataset_train[label_col])
dataset_test[label_col] = le.transform(dataset_test[label_col])

print("Label list: {}".format(le.classes_))

train_text = list(dataset_train[text_col])
train_label = list(dataset_train[label_col])
test_text = list(dataset_test[text_col])
test_label = list(dataset_test[label_col])

## 5. Hyper parameter Setting
# Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 10
max_grad_norm = 1
log_interval = 200
learning_rate =  1e-4

## 6. Test Tokenize, Dataloader 생성
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

class BERTDataset(Dataset):
    def __init__(self, texts, labels, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([text]) for text in texts]
        self.labels = [np.int32(label) for label in labels]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

data_train = BERTDataset(train_text, train_label, tok, max_len, True, False)
data_test = BERTDataset(test_text, test_label, tok, max_len, True, False)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=1)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=1)

## 7. 본 데이터에 맞는 Bert Model 생성
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes= 2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler
        return self.classifier(out)

model = BERTClassifier(bertmodel, num_classes = len(le.classes_), dr_rate=0.5).to(device)

## 7. Optimizer, Loss 정의
# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_correct = (max_indices == Y).sum().data.cpu().numpy()
    return train_correct

## 8. Train
save_best_model = './best_model/best_model.pt'
best_accuracy = 0

for e in range(num_epochs):
    train_N = 0
    train_correct = 0

    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_correct += calc_accuracy(out, label)
        train_N += len(label)

    train_acc = train_correct / train_N
    print("epoch {} train acc {}".format(e+1, train_acc))

    test_N = 0
    test_correct = 0

    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_correct += calc_accuracy(out, label)
        test_N += len(label)

    test_acc = test_correct / test_N
    print("epoch {} test acc {}".format(e+1, test_acc))
    if test_acc > best_accuracy:
        print("Best model saved. Validation Acc: {} -> {}".format(best_accuracy, test_acc)) 
        best_accuracy = test_acc
        torch.save(model, save_best_model)


## 9. Inference
model = torch.load(save_best_model)
model.eval()
gt_list = []
out_list = []
for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
    token_ids = token_ids.long().to(device)
    segment_ids = segment_ids.long().to(device)
    valid_length= valid_length
    out = model(token_ids, valid_length, segment_ids)
    max_vals, max_indices = torch.max(out, 1)
    out_list.append(max_indices.data.cpu().numpy())
    gt_list.append(label.numpy())

gt_list = np.hstack(gt_list)
out_list = np.hstack(out_list)
distribution = confusion_matrix(gt_list, out_list)

plt.figure()
ax = sns.heatmap(distribution, annot=True, cmap='Oranges', fmt='g')
# ax = sns.heatmap(distribution, annot=True, cmap='Oranges', fmt='g', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Test Confusion matrix')
plt.savefig('result/Test_confusion_matrix.png')