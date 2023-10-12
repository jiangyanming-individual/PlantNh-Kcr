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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

Amino_acid_sequence = 'ACDEFGHIKLMNPQRSTVWYX'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_encode_dataset(filepath):
    data_list = []
    result_seq_datas = []
    result_seq_labels = []
    with open(filepath, encoding='utf-8') as f:

        for line in f.readlines():
            x_data_sequence, label = list(line.strip('\n').split(','))
            data_list.append((x_data_sequence, label))

        # print(data_list)
        # print(len(data_list))

    for data in data_list:
        code = []

        result_seq_labels.append(int(data[1]))
        for seq in data[0]:
            one_code = []
            for amino_acid_index in Amino_acid_sequence:
                if amino_acid_index == seq:
                    flag = 1
                else:
                    flag = 0
                one_code.append(flag)

            code.extend(one_code)

        # one_seq_data=np.array(code).astype('float32').reshape((1,29,29))
        result_seq_datas.append(code)
        # print(one_seq_data)

    return np.array(result_seq_datas), np.array(result_seq_labels, dtype=np.int64)



import numpy as np
train_dataset, train_labels = create_encode_dataset(train_filepath)

# print(type(train_dataset))
# print(train_dataset)
# print(type(train_labels))
# print(len(train_labels))

print(train_dataset.shape)
test_dataset, test_labels = create_encode_dataset(test_filepath)
print(test_dataset.shape)


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


# train_set = MyDataset(train_dataset, train_labels)
test_set = MyDataset(test_dataset, test_labels)


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
        self.Bilstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True,
        )

        # classfier layer：
        self.cls_layer = nn.Linear(self.hidden_size * 2,self.num_classes)
        self.dropout1 = nn.Dropout(0.9)


    def forward(self, inputs):

        # LSTM layer

        Bilstm_outputs, (last_hidden_state, last_cell_state) = self.Bilstm(inputs)
        # print("Bilstm_outputs shape:",Bilstm_outputs.shape)

        # (batch_size,seq_len,hidden_size * 2)
        # print(Bilstm_outputs.shape)
        Bilstm_outputs = self.dropout1(Bilstm_outputs)

        # classfier layer：
        outputs = self.cls_layer(Bilstm_outputs[:,-1,:])  #[batch_size,hidden_size]

        return (Bilstm_outputs), outputs



warnings.filterwarnings("ignore")
num_epochs = 30
learning_rate = 0.001


input_size=len(Amino_acid_sequence)
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
model_path='../DL_weights/BiLSTM(BE)-FinalWeight.pth'
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
        FP += ((y_true_label == 0) & (y_test_label == 1)).sum().item()
        TN += ((y_true_label == 0) & (y_test_label == 0)).sum().item()
        FN += ((y_true_label == 1) & (y_test_label == 0)).sum().item()

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

    np.save('../np_weights/BiLSTM(BE)_y_test_true.npy',y_test_true)
    np.save('../np_weights/BiLSTM(BE)_y_test_score.npy',y_test_score)

    fpr, tpr, _ = metrics.roc_curve(y_test_true, y_test_score)

    auc = metrics.auc(fpr, tpr)

    test_roc.append([fpr, tpr])
    test_roc_auc.append(auc)

    tpr = interp(test_base_fpr, fpr, tpr)
    tpr[0] = 0.0
    test_tprs.append(tpr)
    test_fprs.append(fpr)

    print("[ind_test:avg_acc is:{},avg_loss is :{},avg_auc is:{}]".format(avg_acc, avg_loss, avg_auc))

    # eval_SN_SP_ACC_MCC=[]

    #SN，SP，ACC，MCC
    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    MCC = ((TP * TN) - (FP * FN)) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    eval_SN_SP_ACC_MCC=[]

    eval_SN_SP_ACC_MCC.append(SN)
    eval_SN_SP_ACC_MCC.append(SP)
    eval_SN_SP_ACC_MCC.append(ACC)
    eval_SN_SP_ACC_MCC.append(MCC)

    np.save('../np_weights/BiLSTM(BE)_eval.npy',eval_SN_SP_ACC_MCC)

    print("ind_test TP is {},FP is {},TN is {},FN is {}".format(TP, FP, TN, FN))
    print("ind_test : SN is {},SP is {},ACC is {},MCC is {}".format(SN, SP, ACC, MCC))