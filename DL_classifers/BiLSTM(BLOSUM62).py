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


data=read_file(train_filepath)
train_dataset=get_BLOSUM62_encoding(data)


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

train_set = MyDataset(train_dataset[0], train_dataset[1])
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
            bidirectional=True,
            batch_first=True,
        )

        # classfier layer：
        self.cls_layer = nn.Linear(self.hidden_size * 2 ,self.num_classes)
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
num_epochs = 30
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


def train(model, train_loader,valid_loader,device):
    print("train is start...")
    optimizer = torch.optim.Adam(params=model.parameters(),lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):

        epoch_loss = []
        epoch_acc = []
        epoch_auc = []
        # print("第{}个epoch:".format(epoch))
        for batch_id, data in enumerate(train_loader):

            # print(data[0])

            x_data = data[0].to(device)
            # print("x_data:",x_data)
            y_data = data[1].to(device)
            # print("y_data:",y_data)

            y_data = torch.tensor(y_data, dtype=torch.long)


            # print("label:",data[1])

            _, y_predict = model(x_data)

            # print("y_predict:",y_predict)
            # print("y_predict shape:",y_predict.shape)
            loss = F.cross_entropy(y_predict, y_data)

            # loss = loss_fn(y_predict, y_data)
            # print("loss:",loss)
            # acc = torch.metric.accuracy(y_predict, y_data)
            acc=metrics.accuracy_score(y_data.detach().cpu().numpy(),torch.argmax(y_predict,dim=1).detach().cpu().numpy())
            # print("acc:",acc)

            auc = metrics.roc_auc_score(y_data.detach().cpu().numpy(), y_predict[:, 1].detach().cpu().numpy())

            # print("auc:",auc)
            epoch_acc.append(acc)
            epoch_loss.append(loss.detach().cpu().numpy())
            epoch_auc.append(auc)

            # print("epoch_acc:",epoch_acc)
            # print("epoch_loss:",epoch_loss)
            # print("epoch_auc:",epoch_auc)

            if (batch_id % 64 == 0):
                print("epoch is {},batch_id is {},acc is :{} ,loss is {},auc is {}".format(epoch, batch_id,acc,loss.detach().cpu().numpy(), auc))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_acc, avg_loss, avg_auc = np.mean(epoch_acc), np.mean(epoch_loss), np.mean(epoch_auc)
        print("[train:avg_acc is: {},avg_loss is: {},avg_auc is: {}]".format(avg_acc, avg_loss, avg_auc))

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
            # print("x_data:",x_data)
            y_data = data[1].to(device)
            # print("y_data:",y_data)

            y_data = torch.tensor(y_data, dtype=torch.long)

            y_true_label = y_data

            # y_data = torch.unsqueeze(y_data, axis=-1)
            _, y_valid_pred = model(x_data)

            # y_predic_label
            y_predict_label = torch.argmax(y_valid_pred, dim=1)
            # print("y_predict_label",y_predict_label)

            loss = F.cross_entropy(y_valid_pred, y_data)

            # loss = loss_fn(y_valid_pred, y_data)
            # acc = torch.metric.accuracy(y_valid_pred, y_data)
            acc=metrics.accuracy_score(y_data.detach().cpu().numpy(),torch.argmax(y_valid_pred,dim=1).detach().cpu().numpy())

            #cal auc
            auc = metrics.roc_auc_score(y_data[:].detach().cpu().numpy(), y_valid_pred[:, 1].detach().cpu().numpy())

            y_true.append(y_data[:].detach().cpu().numpy())
            y_score.append(y_valid_pred[:, 1].detach().cpu().numpy())

            valid_acc.append(acc)
            valid_loss.append(loss.detach().cpu().numpy())
            valid_auc.append(auc)

            TP += ((y_true_label == 1) & (y_predict_label == 1)).sum().item()
            FP += ((y_true_label == 0) & (y_predict_label == 1)).sum().item()
            TN += ((y_true_label == 0) & (y_predict_label == 0)).sum().item()
            FN += ((y_true_label == 1) & (y_predict_label == 0)).sum().item()
            # valid_auc.append(auc)

            if batch_id % 64 == 0:
                print("batch_id is {} ,acc is {} ,loss is {} ,auc is {}".format(batch_id, acc, loss.detach().cpu().numpy(),auc))

        avg_acc, avg_loss, avg_auc = np.mean(valid_acc), np.mean(valid_loss), np.mean(valid_auc)
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

        print("res_auc :",res_auc)
        print("[valid:avg_acc is:{},avg_loss is :{},avg_auc is:{}]".format(avg_acc, avg_loss, avg_auc))


        np.save('../np_weights/BiLSTM(BLOSUM62)_y_train.npy',y_true)
        np.save('../np_weights/BiLSTM(BLOSUM62)_y_train_score.npy',y_score)
        # SN，SP，ACC，MCC

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



from sklearn.model_selection import KFold
from sklearn import metrics
from torch.utils.data import Subset

from torch.utils.data import DataLoader

kf = KFold(n_splits=5, shuffle=True)
fold = 1

for train_index, valid_index in kf.split(train_set):
    print(f"第{fold}次交叉验证")

    batch_size = 128

    train_dataset = Subset(train_set, train_index)
    valid_dataset = Subset(train_set, valid_index)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=False)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Model_LSTM(input_size,hidden_size, num_classes, num_layers)

    model.to(device)
    train(model, train_loader,valid_loader,device)

    # save model
    torch.save(model.state_dict(), '../DL_weights/'+str(fold) +'_BiLSTM(BLOSUM62)_kfold_model.pth'.format(fold))

    fold += 1


# np.save('../np_weights/BiLSTM(BLOSUM62)_roc_auc.npy', roc_auc)
# np.save('../np_weights/BiLSTM(BLOSUM62)_roc.npy', roc)


#SN、SP、ACC、MCC
mean_SN=np.mean(total_SN)
mean_SP=np.mean(total_SP)
mean_ACC=np.mean(total_ACC)
mean_MCC=np.mean(total_MCC)


kfold_SN_SP_ACC_MCC=[]
kfold_SN_SP_ACC_MCC.append(mean_SN)
kfold_SN_SP_ACC_MCC.append(mean_SP)
kfold_SN_SP_ACC_MCC.append(mean_ACC)
kfold_SN_SP_ACC_MCC.append(mean_MCC)

np.save('../np_weights/BiLSTM(BLOSUM62)_SN_SP_ACC_MCC.npy', kfold_SN_SP_ACC_MCC)
print("5kfold: SN is: {}, SP is: {}, ACC is: {},MCC is: {}".format(mean_SN,mean_SP,mean_ACC,mean_MCC))



def Kf_show(plt, base_fpr, roc, roc_auc):
    for i, item in enumerate(roc):
        fpr, tpr = item
        plt.plot(fpr, tpr, label="ROC fold {} (AUC={:.4f})".format(i + 1, roc_auc[i]), lw=1, alpha=0.3)

    plt.plot(base_fpr, np.average(tprs, axis=0),
             label=r'Mean ROC(AUC=%0.2f $\pm$ %0.2f)' % (np.mean(roc_auc), np.std(roc_auc)),
             lw=1, alpha=0.8, color='b')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, alpha=0.8, color='c')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    plt.legend(loc=4)
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.savefig('../figures/BiLSTM(BLOSUM62).png')
    plt.show()

Kf_show(plt, base_fpr, roc, roc_auc)


#######################################

# total dataset to training

warnings.filterwarnings("ignore")
num_epochs = 30
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


def total_train(model, train_loader,device):
    print("train is start...")
    optimizer = torch.optim.Adam(params=model.parameters(),lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):

        epoch_loss = []
        epoch_acc = []
        epoch_auc = []
        # print("第{}个epoch:".format(epoch))
        for batch_id, data in enumerate(train_loader):

            # print(data[0])

            x_data = data[0].to(device)
            # print("x_data:",x_data)
            y_data = data[1].to(device)
            # print("y_data:",y_data)

            y_data = torch.tensor(y_data, dtype=torch.long)


            # print("label:",data[1])

            _, y_predict = model(x_data)

            # print("y_predict:",y_predict)
            # print("y_predict shape:",y_predict.shape)
            loss = F.cross_entropy(y_predict, y_data)

            # loss = loss_fn(y_predict, y_data)
            # print("loss:",loss)
            # acc = torch.metric.accuracy(y_predict, y_data)
            acc=metrics.accuracy_score(y_data.detach().cpu().numpy(),torch.argmax(y_predict,dim=1).detach().cpu().numpy())
            # print("acc:",acc)

            auc = metrics.roc_auc_score(y_data.detach().cpu().numpy(), y_predict[:, 1].detach().cpu().numpy())

            # print("auc:",auc)
            epoch_acc.append(acc)
            epoch_loss.append(loss.detach().cpu().numpy())
            epoch_auc.append(auc)

            # print("epoch_acc:",epoch_acc)
            # print("epoch_loss:",epoch_loss)
            # print("epoch_auc:",epoch_auc)

            if (batch_id % 64 == 0):
                print("epoch is {},batch_id is {},acc is :{} ,loss is {},auc is {}".format(epoch, batch_id,acc,loss.detach().cpu().numpy(), auc))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_acc, avg_loss, avg_auc = np.mean(epoch_acc), np.mean(epoch_loss), np.mean(epoch_auc)
        print("[train:avg_acc is: {},avg_loss is: {},avg_auc is: {}]".format(avg_acc, avg_loss, avg_auc))

        if (epoch + 1)  == num_epochs:
            torch.save(model.state_dict(), '../DL_weights/model_weights/BiLSTM(BLOSUM62)-FinalWeight.pth')



from sklearn import metrics
from torch.utils.data import DataLoader
batch_size=128

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model_LSTM(input_size,hidden_size, num_classes, num_layers)

model.to(device)
#training model
total_train(model,train_loader,device)




