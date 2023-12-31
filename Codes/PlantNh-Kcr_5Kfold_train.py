# author:Lenovo
# datetime:2023/7/22 20:28
# software: PyCharm
# project:PlantNh-Kcr

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,roc_curve,auc
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import interp
from sklearn.model_selection import KFold
from sklearn import metrics
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

Amino_acid_sequence = 'ACDEFGHIKLMNPQRSTVWYX'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device='cpu'

# binary encoding
def create_encode_dataset(filepath):
    data_list = []

    with open(filepath, encoding='utf-8') as f:

        for line in f.readlines():
            sequence, label = list(line.strip('\n').split(','))
            data_list.append((sequence, label))

    result_seq_data = []
    result_seq_labels = []

    for data in data_list:
        seq,label=data[0],data[1]
        one_seq = []
        result_seq_labels.append(int(label))
        for amino_acid in seq:
            one_amino_acid = []
            for amino_acid_index in Amino_acid_sequence:
                if amino_acid_index == amino_acid:
                    flag = 1
                else:
                    flag = 0
                one_amino_acid.append(flag)
            one_seq.extend(one_amino_acid)
        result_seq_data.append(one_seq)
    return np.array(result_seq_data), np.array(result_seq_labels, dtype=np.int64)


#the filepath of training and test sets
train_filepath= '../Datasets/train.csv'
test_filepath= '../Datasets/ind_test.csv'


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


train_set = MyDataset(train_dataset, train_labels)
test_set = MyDataset(test_dataset, test_labels)

# total model
class Model_LSTM_MutilHeadSelfAttention(nn.Module):

    def __init__(self,input_size, hidden_size, num_classes=2, num_layers=1, attention=None):
        super(Model_LSTM_MutilHeadSelfAttention, self).__init__()

        self.input_size = input_size

        # hidden_size：
        self.hidden_size = hidden_size
        # num_classes：
        self.num_classes = num_classes

        # LSTM layers：
        self.num_layers = num_layers

        # BiLSTM Layer：
        self.Bilstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True,
        )
        # attention layer：
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size * 2,num_heads=8,batch_first=True,dropout=0.5)

        # classfier layer：
        self.cls_layer = nn.Linear(self.hidden_size * 2, self.num_classes)

        self.dropout1 = nn.Dropout(0.9)

    def forward(self, inputs):
        input_ids = inputs
        # LSTM layer
        Bilstm_outputs, (last_hidden_state, last_cell_state) = self.Bilstm(inputs)

        Bilstm_outputs = self.dropout1(Bilstm_outputs)
        context,_ = self.attention(Bilstm_outputs,Bilstm_outputs,Bilstm_outputs)

        MutilHead_output = context
        return (Bilstm_outputs, MutilHead_output), context


input_size=len(Amino_acid_sequence)
hidden_size = 64
num_classes = 2
num_layers = 1


class KcrNet(nn.Module):

    def __init__(self, input_classes=21, nums_classes=2):
        super(KcrNet, self).__init__()

        self.conv1 = torch.nn.Conv1d(in_channels=input_classes, out_channels=32, kernel_size=5, padding=2, stride=1)

        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=2)

        self.conv3 = torch.nn.Conv1d(in_channels=32, out_channels=29, kernel_size=5, padding=2, stride=2)

        self.BiLSTM_ATT=Model_LSTM_MutilHeadSelfAttention(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)

        self.flatten = torch.nn.Flatten()

        self.linear1 = torch.nn.Linear(in_features=29 * 136, out_features=128)
        self.linear2 = torch.nn.Linear(in_features=128, out_features=nums_classes)


        self.dropout1=torch.nn.Dropout(0.3)
        self.dropout2 = torch.nn.Dropout(0.3)

    def forward(self, x):

        inputs=x

        x = torch.permute(x, [0, 2, 1]) # permute

        # first conv1d
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        First_outputs=x

        # seconde conv1d
        x = self.conv2(x)
        x = F.relu(x)

        x = self.dropout2(x)
        Second_outputs=x


        #third conv1d
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout2(x)
        Third_outputs=x
        # print("final x shape:",x.shape)

        # the output of BiLSTM and attention layers
        visual_outputs,BiLSTM_outputs=self.BiLSTM_ATT(inputs)

        total_outputs=torch.cat([x,BiLSTM_outputs],dim=-1)
        # print("total_outputs shape:",total_outputs.shape)

        # flatten layer
        x = self.flatten(total_outputs)

        x = self.linear1(x)
        x = F.relu(x)  # activate function
        Linear_output=x

        x = self.linear2(x)
        return (inputs,First_outputs,Second_outputs,Third_outputs,visual_outputs,Linear_output),x


model=KcrNet()
print(model)




# the number of epochs
epochs = 50
batch_size = 128
learn_rate = 0.001

train_loss = []
train_acc = []
train_auc = []

eval_losses = []
eval_accuracies = []
eval_auces = []


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



def train(model, train_loader, valid_loader,device):
    print("train is start!")
    model.train()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learn_rate)
    for epoch in range(epochs):

        epoch_loss = []
        epoch_acc = []
        epoch_auc = []
        for batch_id, data in enumerate(train_loader):

            x_data = data[0].to(device)
            y_data = data[1].to(device)
            y_data = torch.tensor(y_data, dtype=torch.long)
            # print(x_data)
            _,y_predict = model(x_data)
            # print("y_predict:",y_predict)

            loss = F.cross_entropy(y_predict, y_data)

            acc = metrics.accuracy_score(y_data.detach().cpu().numpy(),torch.argmax(y_predict,dim=1).detach().cpu().numpy())
            # print("y_data:",y_data)
            auc = metrics.roc_auc_score(y_data.detach().cpu().numpy(), y_predict[:, 1].detach().cpu().numpy())

            epoch_loss.append(loss.detach().cpu().numpy())
            epoch_acc.append(acc)
            epoch_auc.append(auc)
            if (batch_id % 64 == 0):
                print("epoch is :{},batch_id is {},loss is {},acc is:{},auc is:{}".format(epoch, batch_id,loss.detach().cpu().numpy(),acc, auc))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_loss, avg_acc, avg_auc = np.mean(epoch_loss), np.mean(epoch_acc), np.mean(epoch_auc)
        print("[train acc is:{}, loss is :{},auc is:{}]".format(avg_acc, avg_loss, avg_auc))


    print("eval is start...")
    model.eval()
    with torch.no_grad():

        valid_acc = []
        valid_loss = []
        valid_auc = []

        y_true = []
        y_score = []

        TP, FP, TN, FN = 0, 0, 0, 0

        for batch_id, data in enumerate(valid_loader):
            x_data = data[0].to(device)
            y_data = data[1].to(device)
            # y_data = torch.unsqueeze(y_data,dim=1)
            y_data = torch.tensor(y_data, dtype=torch.long)

            y_true_label=y_data

            _,y_predict = model(x_data)
            y_predict_label=torch.argmax(y_predict,dim=1)

            loss = F.cross_entropy(y_predict, y_data)

            acc = metrics.accuracy_score(y_data.detach().cpu().numpy(),torch.argmax(y_predict, dim=1).detach().cpu().numpy())

            auc = roc_auc_score(y_data[:].detach().cpu().numpy(), y_predict[:, 1].detach().cpu().numpy())

            valid_loss.append(loss.detach().cpu().numpy())
            valid_acc.append(acc)
            valid_auc.append(auc)

            y_true.append(y_data[:].detach().cpu().numpy())
            y_score.append(y_predict[:, 1].detach().cpu().numpy())

            TP += ((y_true_label == 1) & (y_predict_label == 1)).sum().item()
            FP += ((y_true_label == 0) & (y_predict_label == 1)).sum().item()
            TN += ((y_true_label == 0) & (y_predict_label == 0)).sum().item()
            FN += ((y_true_label == 1) & (y_predict_label == 0)).sum().item()

            if (batch_id % 64 == 0):
                print("batch_id is {},loss is {},acc is:{}, auc is {}".format(batch_id,loss.detach().cpu().numpy(),acc,auc))

        avg_acc, avg_loss, avg_auc = np.mean(valid_acc), np.mean(valid_loss), np.mean(valid_auc)
        print("[test acc is:{},loss is:{},auc is:{}]".format(avg_acc, avg_loss, avg_auc))


        #concatenate data
        y_true = np.concatenate(y_true)
        y_score = np.concatenate(y_score)

        fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
        res_auc = metrics.auc(fpr, tpr)
        roc.append([fpr, tpr])

        roc_auc.append(res_auc)

        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)
        fprs.append(fpr)

        # np.save('../np_weights/PlantNh-Kcr_y_train.npy',y_true)
        # np.save('../np_weights/PlantNh-Kcr_y_train_score.npy',y_score)

        print("res_auc :",res_auc)
        print("[valid:avg_acc is:{},avg_loss is :{},avg_auc is:{}]".format(avg_acc, avg_loss, avg_auc))

        SN = TP / (TP + FN)
        SP = TN / (TN + FP)
        ACC = (TP + TN) / (TP + TN + FP + FN)
        MCC = ((TP * TN) - (FP * FN)) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

        print("Train TP is {},FP is {},TN is {},FN is {}".format(TP, FP, TN, FN))
        print("Train SN is {},SP is {},ACC is {},MCC is {}".format(SN, SP, ACC, MCC))

        total_SN.append(SN)
        total_SP.append(SP)
        total_ACC.append(ACC)
        total_MCC.append(MCC)


# five-fold cross-validation
kf = KFold(n_splits=5, shuffle=True)
fold = 1



for train_index, valid_index in kf.split(train_set):
    print(f"第{fold}次交叉验证")

    batch_size = 128
    # train dataset and valid dataset

    train_dataset = Subset(train_set, train_index)
    valid_dataset = Subset(train_set, valid_index)

    # DataLoader:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = KcrNet()
    model.to(device)

    train(model,train_loader,valid_loader,device)
    # save model
    torch.save(model.state_dict(), '../model_weights/'+str(fold) + '_PlantNh-Kcr_kfold_model.pth'.format(fold))
    fold += 1


# np.save('../np_weights/PlantNh-Kcr_roc_auc.npy', roc_auc)
# np.save('../np_weights/PlantNh-Kcr_roc.npy', roc)

# SN、SP、ACC、MCC
mean_SN=np.mean(total_SN)
mean_SP=np.mean(total_SP)
mean_ACC=np.mean(total_ACC)
mean_MCC=np.mean(total_MCC)


kfold_SN_SP_ACC_MCC=[]
kfold_SN_SP_ACC_MCC.append(mean_SN)
kfold_SN_SP_ACC_MCC.append(mean_SP)
kfold_SN_SP_ACC_MCC.append(mean_ACC)
kfold_SN_SP_ACC_MCC.append(mean_MCC)

#the mean results of five-fold cross-validation on Sn, Sp, ACC, MCC

np.save('../np_weights/PlantNh-Kcr_5kfold_SN_SP_ACC_MCC.npy', kfold_SN_SP_ACC_MCC)
print("5kfold: SN is: {}, SP is: {}, ACC is: {},MCC is: {}".format(mean_SN,mean_SP,mean_ACC,mean_MCC))

# the visualization result of five-fold cross-validation

def Kf_show(plt, base_fpr, roc, roc_auc):

    plt.figure(dpi=600)
    for i, item in enumerate(roc):
        fpr, tpr = item
        plt.plot(fpr, tpr, label="ROC fold {} (AUC={:.4f})".format(i + 1, roc_auc[i]), lw=1, alpha=0.3)


    #calculate mean value
    plt.plot(base_fpr, np.average(tprs, axis=0),
             label=r'Mean ROC(AUC=%0.2f $\pm6$ %0.2f)' % (np.mean(roc_auc), np.std(roc_auc)),
             lw=1, alpha=0.8, color='b')
    #base line
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, alpha=0.8, color='c')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    plt.legend(loc=4)
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.savefig('../figures/PlantNh-Kcr-5Kfold.png')
    plt.show()

Kf_show(plt, base_fpr, roc, roc_auc)
