# author:Lenovo
# datetime:2023/7/22 20:28
# software: PyCharm
# project:PlantNh-Kcr


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score,roc_curve,auc


Amino_acid_sequence = 'ACDEFGHIKLMNPQRSTVWYX'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#the filepath of training and test sets
train_filepath= '../Datasets/train.csv'
test_filepath= '../Datasets/ind_test.csv'


def read_file(filepath):

    data=[]
    with open(filepath,mode='r',encoding='utf-8') as f:

        for line in f.readlines():
            seq,label=line.strip().split(',')
            data.append((seq,label))


        f.close()
    # print(data)
    return data


def get_AAindex_encode(data):

    X=[]
    y=[]

    with open('../features_encode/AAindex/AAindex_normalized.txt', mode='r') as f:
        records=f.readlines()[1:]
        f.close()

    AA = 'ARNDCQEGHILKMFPSTWYV'

    AAindex_names = []
    AAindex = []

    for i in records:
        # print(i.rstrip().split()[0])
        AAindex_names.append(i.rstrip().split()[0] if i.rstrip() != '' else None)
        AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)

    props = 'FINA910104:LEVM760101:JACR890101:ZIMJ680104:RADA880108:JANJ780101:CHOC760102:NADH010102:KYTJ820101:NAKH900110:GUYH850101:EISD860102:HUTJ700103:OLSK800101:JURD980101:FAUJ830101:OOBM770101:GARJ730101:ROSM880102:RICJ880113:KIDA850101:KLEP840101:FASG760103:WILM950103:WOLS870103:COWR900101:KRIW790101:AURR980116:NAKH920108'.split(':')

    if props:
        tempAAindex_names = []
        tempAAindex = []

        for p in props:

            if AAindex_names.index(p) != -1:
                tempAAindex_names.append(p)
                tempAAindex.append(AAindex[AAindex_names.index(p)])


        if len(tempAAindex_names) != 0:
            AAindex_names = tempAAindex_names
            AAindex = tempAAindex


    # print("AAindex:",AAindex)
    seq_index = {} #(0-19)
    for i in range(len(AA)):
        seq_index[AA[i]] = i


    for seq,label in data:
        one_code=[]
        for aa in seq:
            if aa == 'X':
                for aaindex in AAindex:  # Using X
                    one_code.append(0)
                continue
            for aaindex in AAindex:
                # print(type(aaindex[seq_index.get(aa)]))
                one_code.append(aaindex[seq_index.get(aa)])
        X.append(one_code) #(29,29)
        # print(one_code)
        y.append(int(label))

    X=np.array(X)
    # print(X.shape)
    n,seq_len=X.shape
    print("X shape :",X.shape)

    y=np.array(y)
    print("y shape:",y.shape)
    return X,y

data=read_file(train_filepath)
train_dataset=get_AAindex_encode(data)




class MyDataset(Dataset):

    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels

    def __getitem__(self, index):
        x_data = np.array(self.datas[index]).astype('float32').reshape((29, 29))
        y_data = self.labels[index]

        return x_data, y_data

    def __len__(self):
        return len(self.datas)


train_set = MyDataset(train_dataset[0], train_dataset[1])
print(train_set)
# test_set = MyDataset(test_dataset, test_labels)


class KcrNet(nn.Module):

    def __init__(self, input_classes=21, nums_classes=2):
        super(KcrNet, self).__init__()

        self.conv1 = torch.nn.Conv1d(in_channels=input_classes, out_channels=32, kernel_size=5, padding=2, stride=1)

        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=2)

        self.conv3 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=2)

        #flatten layer
        self.flatten = torch.nn.Flatten()
        # linear layer
        self.linear1 = torch.nn.Linear(in_features=32 * 8, out_features=128)
        self.linear2 = torch.nn.Linear(in_features=128, out_features=nums_classes)


        self.dropout1=torch.nn.Dropout(0.3)
        self.dropout2 = torch.nn.Dropout(0.3)

    def forward(self, x):
        x = torch.permute(x, [0, 2, 1]) # permute
        # 1 Conv1D layer
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # 2 Conv1D layer
        x = self.conv2(x)
        x = F.relu(x)

        x = self.dropout1(x)

        # 3 Conv1D layer
        x = self.conv3(x)
        x = F.relu(x)

        x = self.dropout2(x)


        # flatten layer
        x = self.flatten(x)

        x = self.linear1(x)
        x = F.relu(x)  # activate function
        x = self.linear2(x)
        return x



model=KcrNet(input_classes=29,nums_classes=2)
print(model)


import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import interp
import warnings
warnings.filterwarnings("ignore")

epochs = 30
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

            # 进行模型训练：
            y_predict = model(x_data)
            # print("y_predict:",y_predict)

            loss = F.cross_entropy(y_predict, y_data)

            # loss=loss_fn(y_predict,y_data)

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

            y_predict = model(x_data)
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


        #concate data
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
        #
        np.save('../np_weights/CNN(AAindex)_y_train.npy',y_true)
        np.save('../np_weights/CNN(AAindex)_y_train_score.npy',y_score)

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
from sklearn.model_selection import KFold
from sklearn import metrics
from torch.utils.data import Subset
from torch.utils.data import DataLoader

kf = KFold(n_splits=5, shuffle=True)
fold = 1



for train_index, valid_index in kf.split(train_set):
    print(f"第{fold}次交叉验证")

    batch_size = 128
    # create training and valid sets

    train_dataset = Subset(train_set, train_index)
    valid_dataset = Subset(train_set, valid_index)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=KcrNet(input_classes=29,nums_classes=2)
    model.to(device)

    train(model,train_loader,valid_loader,device)



    # save model
    torch.save(model.state_dict(), '../DL_weights/'+str(fold) + '_CNN(AAindex)_kfold_model.pth'.format(fold))

    fold += 1


# save tpr,fpr,auc
# np.save('../np_weights/CNN(AAindex)_roc_auc.npy', roc_auc)
# np.save('../np_weights/CNN(AAindex)_roc.npy', roc)


#the result of SN,SP,ACC,MCC
mean_SN=np.mean(total_SN)
mean_SP=np.mean(total_SP)
mean_ACC=np.mean(total_ACC)
mean_MCC=np.mean(total_MCC)


kfold_SN_SP_ACC_MCC=[]
kfold_SN_SP_ACC_MCC.append(mean_SN)
kfold_SN_SP_ACC_MCC.append(mean_SP)
kfold_SN_SP_ACC_MCC.append(mean_ACC)
kfold_SN_SP_ACC_MCC.append(mean_MCC)

#the results
np.save('../np_weights/CNN(AAindex)_5kfold_SN_SP_ACC_MCC.npy', kfold_SN_SP_ACC_MCC)
print("5kfold: SN is: {}, SP is: {}, ACC is: {},MCC is: {}".format(mean_SN,mean_SP,mean_ACC,mean_MCC))


# the visualization of five-fold cross-validation
def Kf_show(plt, base_fpr, roc, roc_auc):

    for i, item in enumerate(roc):
        fpr, tpr = item
        plt.plot(fpr, tpr, label="ROC fold {} (AUC={:.4f})".format(i + 1, roc_auc[i]), lw=1, alpha=0.3)

    # mean
    plt.plot(base_fpr, np.average(tprs, axis=0),
             label=r'Mean ROC(AUC=%0.2f $\pm$ %0.2f)' % (np.mean(roc_auc), np.std(roc_auc)),
             lw=1, alpha=0.8, color='b')
    # base line
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, alpha=0.8, color='c')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    plt.legend(loc=4)
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    # plt.savefig('../figures/CNN(AAindex).png')
    plt.show()

Kf_show(plt, base_fpr, roc, roc_auc)

######################################

# total dataset to training
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import interp
import warnings
warnings.filterwarnings("ignore")


epochs = 30
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



def total_train(model, train_loader,device):
    print("train is start!")
    model.train()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learn_rate)
    # loss_fn = FocalLoss(alpha=0.8, gamma=1.5)
    for epoch in range(epochs):

        epoch_loss = []
        epoch_acc = []
        epoch_auc = []
        for batch_id, data in enumerate(train_loader):

            x_data = data[0].to(device)
            y_data = data[1].to(device)
            y_data = torch.tensor(y_data, dtype=torch.long)
            # print(x_data)

            # 进行模型训练：
            y_predict = model(x_data)
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

        # 取每次epoch的均值
        avg_loss, avg_acc, avg_auc = np.mean(epoch_loss), np.mean(epoch_acc), np.mean(epoch_auc)
        print("[train acc is:{}, loss is :{},auc is:{}]".format(avg_acc, avg_loss, avg_auc))

        if (epoch + 1)  == epochs:
            #save model
            torch.save(model.state_dict(), '../DL_weights/CNN(AAindex)-FinalWeight.pth')


from sklearn import metrics
from torch.utils.data import DataLoader
batch_size=128

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=KcrNet(input_classes=29,nums_classes=2)
model.to(device)

# training model
total_train(model,train_loader,device)
