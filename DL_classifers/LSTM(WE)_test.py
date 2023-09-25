import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import warnings
import math
from numpy import interp

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,Subset


#构造词典：
AA_aaindex = 'ACDEFGHIKLMNPQRSTVWY'

word2id_dict={'X':0}
for i in range(len(AA_aaindex)):
    word2id_dict[AA_aaindex[i]]=i+1

#使用均衡数据集的文件路径：

train_filepath= '../Datasets/train.csv'
test_filepath= '../Datasets/ind_test.csv'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 加载数据集,返回seq,label
def load_data(file_path):
    data = []
    with open(file_path, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            # one_data = []
            sequence, y_label = line.strip().split(',')
            # print(line.strip().split(','))
            data.append((sequence, y_label))

    return data


train_dataset=load_data(train_filepath)
test_dataset=load_data(test_filepath)
# train_dataset.shape
# train_dataset
# print(train_dataset)


# 自定义MyDataset
# 构建Dataset数据集：将词转为id
class MyDataset(Dataset):

    def __init__(self, examples, word2id_dict):
        super(MyDataset, self).__init__()
        self.word2id_dict = word2id_dict
        self.examples = self.words_to_id(examples)

    # 将语句转为id的形式：并返回seq,label;
    def words_to_id(self, examples):
        temp_example = []

        for i, example in enumerate(examples):
            seq, label = example
            # 词转为id;如果单词不存在，直接使用unk填充：
            seq = [self.word2id_dict.get(AA, self.word2id_dict.get('X')) for AA in seq]
            # 标签
            label = int(label)
            temp_example.append((seq, label))

        return temp_example

    def __getitem__(self, idx):
        # 将单词转换为id
        seq, label = self.examples[idx]

        return seq, label

    def __len__(self):
        return len(self.examples)


#train dataset：
train_set=MyDataset(train_dataset,word2id_dict)
#ind_test：
test_set=MyDataset(test_dataset,word2id_dict)

# print(train_set)



# total model
class Model_LSTM(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, num_classes=2, num_layers=1):
        super(Model_LSTM, self).__init__()
        # dict size：
        self.vocab_size = vocab_size

        # embedding_size层大小：
        self.embedding_size = embedding_size

        # hidden_size：
        self.hidden_size = hidden_size
        # num_classes：
        self.num_classes = num_classes

        # LSTM layers：
        self.num_layers = num_layers

        # 定义embedding层：
        self.embedding = nn.Embedding(
            vocab_size, embedding_size
        )

        # BiLSTM：
        self.lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=False,
            batch_first=True,
        )

        # classfier layer：
        self.cls_layer = nn.Linear(self.hidden_size ,self.num_classes)
        self.dropout1 = nn.Dropout(0.9)


    def forward(self, inputs):
        input_ids = inputs  # (词的id,有效的长度)：

        # embedding layer;
        embeded_input = self.embedding(input_ids) #(batch_size,seq_len)
        # print("embeded_input shape:",embeded_input.shape)
        embeded_output = embeded_input
        # LSTM layer

        lstm_outputs, (last_hidden_state, last_cell_state) = self.lstm(embeded_output)
        # print("Bilstm_outputs shape:",Bilstm_outputs.shape)

        # (batch_size,seq_len,hidden_size * 2)
        # print(Bilstm_outputs.shape)
        lstm_outputs = self.dropout1(lstm_outputs)

        # classfier layer：
        outputs = self.cls_layer(lstm_outputs[:,-1,:])  #[batch_size,hidden_size]

        return (embeded_output, lstm_outputs), outputs


# 模型训练：

warnings.filterwarnings("ignore")
# 指定训练轮次
num_epochs = 30
# 指定学习率
learning_rate = 0.001
# 指定embedding的数量为词表长度
vocab_size = len(word2id_dict)

# embedding向量的维度
embedding_size = 10

# LSTM网络隐状态向量的维度
hidden_size = 64
num_classes = 2

num_layers = 2

train_losses = []
train_acces = []
train_auces = []


roc = []
roc_auc = []

tprs = []
fprs = []

base_fpr = np.linspace(0, 1, 101)
base_fpr[-1] = 1.0

total_SN = []
total_SP = []
total_ACC = []
total_MCC = []

batch_size=128

test_loader=DataLoader(test_set,batch_size=batch_size,shuffle=True,drop_last=False)


#实例化总的模型：
model=Model_LSTM(vocab_size,embedding_size,hidden_size,num_classes,num_layers)
model.to(device)


#load mmodel
model_path="../DL_weights/LSTM(WE)-FinalWeight.pth"
model.load_state_dict(torch.load(model_path,map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))



import math
from sklearn import metrics

test_roc = []
test_roc_auc = []

test_tprs = []
test_fprs = []

test_base_fpr = np.linspace(0, 1, 101)
test_base_fpr[-1] = 1.0

# 独立数据集测试：
model.eval()
with torch.no_grad():
    test_acc = []
    test_loss = []
    test_auc = []

    y_test_true = []
    y_test_score = []

    TP, FP, TN, FN = 0, 0, 0, 0

    SN = 0
    SP = 0
    ACC = 0
    MCC = 0

    for batch_id, data in enumerate(test_loader):

        data[0] = torch.stack(data[0], dim=1)
        # print("data[0]:",data[0])
        x_data = data[0].to(device)
        y_data = data[1].to(device)
        y_data = torch.tensor(y_data, dtype=torch.long)

        y_true_label = y_data
        # y_data = torch.unsqueeze(y_data, axis=-1)


        _, y_test_pred = model(x_data)

        y_test_label = torch.argmax(y_test_pred, dim=1)

        TP += ((y_true_label == 1) & (y_test_label == 1)).sum().item()
        FP += ((y_true_label == 1) & (y_test_label == 0)).sum().item()
        TN += ((y_true_label == 0) & (y_test_label == 0)).sum().item()
        FN += ((y_true_label == 0) & (y_test_label == 1)).sum().item()

        # 计算损失值：
        loss = F.cross_entropy(y_test_pred, y_data)

        #calculate acc
        acc =metrics.accuracy_score(y_data.detach().cpu().numpy(),torch.argmax(y_test_pred,dim=1).detach().cpu().numpy())
        #calculate auc
        auc = metrics.roc_auc_score(y_data[:].detach().cpu().numpy(), y_test_pred[:, 1].detach().cpu().numpy())


        # 计算得分
        y_test_true.append(y_data[:].detach().cpu().numpy())
        y_test_score.append(y_test_pred[:, 1].detach().cpu().numpy())


        test_acc.append(acc)
        test_loss.append(loss.detach().cpu().numpy())
        test_auc.append(auc)

    avg_acc, avg_loss, avg_auc = np.mean(test_acc), np.mean(test_loss), np.mean(test_auc)

    # 合并数据：
    y_test_true = np.concatenate(y_test_true)
    y_test_score = np.concatenate(y_test_score)

    # 保存真实值和预测值的值
    np.save('../np_weights/LSTM(WE)_y_test_true.npy',y_test_true)
    np.save('../np_weights/LSTM(WE)_y_test_score.npy',y_test_score)

    fpr, tpr, _ = metrics.roc_curve(y_test_true, y_test_score)

    auc = metrics.auc(fpr, tpr)

    test_roc.append([fpr, tpr])
    test_roc_auc.append(auc)

    tpr = interp(test_base_fpr, fpr, tpr)
    tpr[0] = 0.0
    test_tprs.append(tpr)
    test_fprs.append(fpr)

    print("[ind_test auc:]", auc)
    print("[ind_test:avg_acc is:{},avg_loss is :{},avg_auc is:{}]".format(avg_acc, avg_loss, avg_auc))

    # eval_SN_SP_ACC_MCC=[]

    # 计算SN，SP，ACC，MCC
    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    MCC = ((TP * TN) - (FP * FN)) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    eval_SN_SP_ACC_MCC=[]

    eval_SN_SP_ACC_MCC.append(SN)
    eval_SN_SP_ACC_MCC.append(SP)
    eval_SN_SP_ACC_MCC.append(ACC)
    eval_SN_SP_ACC_MCC.append(MCC)

    np.save('../np_weights/LSTM(WE)_eval.npy',eval_SN_SP_ACC_MCC)

    print("ind_test TP is {},FP is {},TN is {},FN is {}".format(TP, FP, TN, FN))
    print("ind_test : SN is {},SP is {},ACC is {},MCC is {}".format(SN, SP, ACC, MCC))