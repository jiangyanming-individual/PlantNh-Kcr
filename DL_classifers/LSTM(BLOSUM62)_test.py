import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import warnings
import math
from numpy import interp

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,Subset




train_filepath= '../Datasets/train.csv'
test_filepath= '../Datasets/ind_test.csv'

Amino_acid_sequence = 'ACDEFGHIKLMNPQRSTVWYX'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def read_file(filepath):

    data=[]
    with open(filepath,mode='r',encoding='utf-8') as f:

        for line in f.readlines():
            seq,label=line.strip().split(',')
            data.append((seq,label))


        f.close()
    # print(data)
    return data


def get_BLOSUM62_encoding(data):

    X=[]
    y=[]

    blosum62 = {
        'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, 0],  # A
        'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, 0],  # R
        'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 0],  # N
        'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 0],  # D
        'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, 0],  # C
        'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0],  # Q
        'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 0],  # E
        'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, 0],  # G
        'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0],  # H
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, 0],  # I
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, 0],  # L
        'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0],  # K
        'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, 0],  # M
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, 0],  # F
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, 0],  # P
        'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0],  # S
        'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, 0],  # T
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, 0],  # W
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, 0],  # Y
        'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, 0],  # V
        'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -
    }

    for key in blosum62:
        for index,value in enumerate(blosum62[key]):
            blosum62[key][index]=round((value + 4) / 15,3)

    for seq,label in data:
        # print(seq)
        # print(label)
        one_code=[]
        for aa in seq:
            # print(blosum62.get(aa))
            one_code.extend(blosum62.get(aa)) #(29,21)

        # print("one_code:",one_code)

        X.append(one_code)
        y.append(int(label))

    X=np.array(X)
    print("X shape:",X.shape)

    y=np.array(y)
    print(y.shape)

    return X,y


# data=read_file(train_filepath)
# train_dataset=get_BLOSUM62_encoding(data)

data=read_file(test_filepath)
test_dataset=get_BLOSUM62_encoding(data)


class MyDataset(Dataset):

    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels

    def __getitem__(self, index):
        x_data = np.array(self.datas[index]).astype('float32').reshape((29, 21))
        y_data = self.labels[index]

        return x_data, y_data

    def __len__(self):
        return len(self.datas)


# train_set = MyDataset(train_dataset[0], train_dataset[1])
test_set = MyDataset(test_dataset[0], test_dataset[1])
# print(train_set)



# total model
class Model_LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes=2, num_layers=1):
        super(Model_LSTM, self).__init__()

        self.input_size=input_size

        # hidden_size：
        self.hidden_size = hidden_size
        # num_classes：
        self.num_classes = num_classes

        # LSTM layers：
        self.num_layers = num_layers


        # BiLSTM：
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=False,
            batch_first=True,
        )

        # classfier layer：
        self.cls_layer = nn.Linear(self.hidden_size ,self.num_classes)
        self.dropout1 = nn.Dropout(0.9)


    def forward(self, inputs):

       # LSTM layer
        lstm_outputs, (last_hidden_state, last_cell_state) = self.lstm(inputs)
        # print("Bilstm_outputs shape:",Bilstm_outputs.shape)

        # (batch_size,seq_len,hidden_size * 2)
        # print(Bilstm_outputs.shape)
        lstm_outputs = self.dropout1(lstm_outputs)

        # classfier layer：
        outputs = self.cls_layer(lstm_outputs[:,-1,:])  #[batch_size,hidden_size]

        return (lstm_outputs), outputs


warnings.filterwarnings("ignore")
learning_rate = 0.001

input_size=21
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

model=Model_LSTM(input_size,hidden_size,num_classes,num_layers)
model.to(device)


#load mmodel
model_path="../DL_weights/LSTM(BLOSUM62)-FinalWeight.pth"
model.load_state_dict(torch.load(model_path,map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))



import math
from sklearn import metrics

test_roc = []
test_roc_auc = []

test_tprs = []
test_fprs = []

test_base_fpr = np.linspace(0, 1, 101)
test_base_fpr[-1] = 1.0

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

        x_data = data[0].to(device)
        # print("x_data:",x_data)
        y_data = data[1].to(device)
        # print("y_data:",y_data)

        y_data = torch.tensor(y_data, dtype=torch.long)

        y_true_label = y_data
        # y_data = torch.unsqueeze(y_data, axis=-1)


        _, y_test_pred = model(x_data)

        y_test_label = torch.argmax(y_test_pred, dim=1)

        TP += ((y_true_label == 1) & (y_test_label == 1)).sum().item()
        FP += ((y_true_label == 1) & (y_test_label == 0)).sum().item()
        TN += ((y_true_label == 0) & (y_test_label == 0)).sum().item()
        FN += ((y_true_label == 0) & (y_test_label == 1)).sum().item()

        loss = F.cross_entropy(y_test_pred, y_data)

        #calculate acc
        acc =metrics.accuracy_score(y_data.detach().cpu().numpy(),torch.argmax(y_test_pred,dim=1).detach().cpu().numpy())
        #calculate auc
        auc = metrics.roc_auc_score(y_data[:].detach().cpu().numpy(), y_test_pred[:, 1].detach().cpu().numpy())


        y_test_true.append(y_data[:].detach().cpu().numpy())
        y_test_score.append(y_test_pred[:, 1].detach().cpu().numpy())


        test_acc.append(acc)
        test_loss.append(loss.detach().cpu().numpy())
        test_auc.append(auc)

    avg_acc, avg_loss, avg_auc = np.mean(test_acc), np.mean(test_loss), np.mean(test_auc)

    y_test_true = np.concatenate(y_test_true)
    y_test_score = np.concatenate(y_test_score)

    np.save('../np_weights/LSTM(BLOSUM62)_y_test_true.npy',y_test_true)
    np.save('../np_weights/LSTM(BLOSUM62)_y_test_score.npy',y_test_score)

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

    # SN，SP，ACC，MCC
    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    MCC = ((TP * TN) - (FP * FN)) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    eval_SN_SP_ACC_MCC=[]

    eval_SN_SP_ACC_MCC.append(SN)
    eval_SN_SP_ACC_MCC.append(SP)
    eval_SN_SP_ACC_MCC.append(ACC)
    eval_SN_SP_ACC_MCC.append(MCC)

    np.save('../np_weights/LSTM(BLOSUM62)_eval.npy',eval_SN_SP_ACC_MCC)

    print("ind_test TP is {},FP is {},TN is {},FN is {}".format(TP, FP, TN, FN))
    print("ind_test : SN is {},SP is {},ACC is {},MCC is {}".format(SN, SP, ACC, MCC))