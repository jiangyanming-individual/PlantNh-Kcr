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

# 对数据进行二进制编码：
Amino_acid_sequence = 'ACDEFGHIKLMNPQRSTVWYX'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 对数据集进行编码的操作：
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
        # 取一条氨基酸序列和对应的lable；
        code = []  # 一条序列
        # seq_index=1

        result_seq_labels.append(int(data[1]))
        for seq in data[0]:
            one_code = []  # 一条序列
            for amino_acid_index in Amino_acid_sequence:
                if amino_acid_index == seq:
                    flag = 1
                else:
                    flag = 0
                one_code.append(flag)

            code.extend(one_code)

        result_seq_datas.append(code)
        # print(one_seq_data)

    return np.array(result_seq_datas), np.array(result_seq_labels, dtype=np.int64)


#数据集的文件路径：
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



# 构建数据集：
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


# 形成数据集：tuple
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
            dropout=0.5
        )
        # attention layer：
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size * 2,num_heads=8,batch_first=True,dropout=0.5)

        # classfier layer：
        self.cls_layer = nn.Linear(self.hidden_size * 2,self.num_classes)

        self.dropout1 = nn.Dropout(0.9)
        self.dropout2=nn.Dropout(0.3)

    def forward(self, inputs):
        input_ids = inputs  # (词的id,有效的长度)：
        # LSTM layer

        Bilstm_outputs, (last_hidden_state, last_cell_state) = self.Bilstm(inputs)
        # print("Bilstm_outputs shape:",Bilstm_outputs.shape)
        # (batch_size,seq_len,hidden_size * 2)

        Bilstm_outputs = self.dropout1(Bilstm_outputs)

        # print("Bilstm_outputs shape:",Bilstm_outputs.shape)

        context,_ = self.attention(Bilstm_outputs,Bilstm_outputs,Bilstm_outputs)
        # print("context shape:",context.shape)
        # context = self.dropout2(context)

        MutilHead_output = context

        # print("context shape:",context.shape)


        return (Bilstm_outputs, MutilHead_output), context

import warnings
# 模型训练：
warnings.filterwarnings("ignore")
# LSTM网络隐状态向量的维度

input_size=len(Amino_acid_sequence)
hidden_size = 64
num_classes = 2
num_layers = 1


class KcrNet(nn.Module):

    def __init__(self, input_classes=21, nums_classes=2):
        super(KcrNet, self).__init__()
        # 定义卷积层：
        self.conv1 = torch.nn.Conv1d(in_channels=input_classes, out_channels=32, kernel_size=5, padding=2, stride=1)
        # 定义pooling层：
        # self.maxpool1=torch.nn.MaxPool1d(kernel_size=1,stride=2)

        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=2)
        # self.maxpool2=torch.nn.MaxPool1d(kernel_size=1,stride=2)

        self.conv3 = torch.nn.Conv1d(in_channels=32, out_channels=29, kernel_size=5, padding=2, stride=2)

        self.BiLSTM_ATT=Model_LSTM_MutilHeadSelfAttention(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)

        # 定义全连接层：
        self.flatten = torch.nn.Flatten()
        # 定义感知层；
        self.linear1 = torch.nn.Linear(in_features=29 * 136, out_features=128)
        self.linear2 = torch.nn.Linear(in_features=128, out_features=nums_classes)


        self.dropout1=torch.nn.Dropout(0.3)
        self.dropout2 = torch.nn.Dropout(0.3)

    def forward(self, x):

        inputs=x

        x = torch.permute(x, [0, 2, 1]) # 对数据进行重新排列

        # 第一层卷积
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        First_outputs=x

        # 第二层卷积
        x = self.conv2(x)
        x = F.relu(x)

        x = self.dropout2(x)
        Second_outputs=x

        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout2(x)
        Third_outputs=x
        # print("final x shape:",x.shape)

        visual_outputs,BiLSTM_outputs=self.BiLSTM_ATT(inputs)


        total_outputs=torch.cat([x,BiLSTM_outputs],dim=-1)
        # print("total_outputs shape:",total_outputs.shape)

        # 全连接层：
        x = self.flatten(total_outputs)

        x = self.linear1(x)
        x = F.relu(x)  # 激活函数
        Linear_output=x

        x = self.linear2(x)
        return (inputs,First_outputs,Second_outputs,Third_outputs,visual_outputs,Linear_output),x


model=KcrNet()
print(model)


import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import interp
import warnings
warnings.filterwarnings("ignore")

# 模型准备
epochs = 50
batch_size = 128
learn_rate = 0.001

# 使用低级api进行构建训练；
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
            _,y_predict = model(x_data)
            # print("y_predict:",y_predict)

            # 二分问题使用的损失函数 binary_cross_entropy函数；
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

        # 取每次epoch的均值
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

            # 计算损失值：
            loss = F.cross_entropy(y_predict, y_data)
            # loss = loss_fn(y_predict, y_data)

            acc = metrics.accuracy_score(y_data.detach().cpu().numpy(),torch.argmax(y_predict, dim=1).detach().cpu().numpy())

            auc = roc_auc_score(y_data[:].detach().cpu().numpy(), y_predict[:, 1].detach().cpu().numpy())

            valid_loss.append(loss.detach().cpu().numpy())
            valid_acc.append(acc)
            valid_auc.append(auc)

            y_true.append(y_data[:].detach().cpu().numpy())
            y_score.append(y_predict[:, 1].detach().cpu().numpy())

            TP += ((y_true_label == 1) & (y_predict_label == 1)).sum().item()
            FP += ((y_true_label == 1) & (y_predict_label == 0)).sum().item()
            TN += ((y_true_label == 0) & (y_predict_label == 0)).sum().item()
            FN += ((y_true_label == 0) & (y_predict_label == 1)).sum().item()

            if (batch_id % 64 == 0):
                print("batch_id is {},loss is {},acc is:{}, auc is {}".format(batch_id,loss.detach().cpu().numpy(),acc,auc))

        avg_acc, avg_loss, avg_auc = np.mean(valid_acc), np.mean(valid_loss), np.mean(valid_auc)
        print("[test acc is:{},loss is:{},auc is:{}]".format(avg_acc, avg_loss, avg_auc))


        #合并
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


# 进行五折交叉验证：
from sklearn.model_selection import KFold
from sklearn import metrics
from torch.utils.data import Subset

# 定义DataLoader
from torch.utils.data import DataLoader

kf = KFold(n_splits=5, shuffle=True)
fold = 1



for train_index, valid_index in kf.split(train_set):
    print(f"第{fold}次交叉验证")

    batch_size = 128
    # 创建训练集和验证集：

    train_dataset = Subset(train_set, train_index)
    valid_dataset = Subset(train_set, valid_index)

    # 形成DataLoader:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = KcrNet()
    model.to(device)

    train(model,train_loader,valid_loader,device)

    # save tpr,fpr,auc

    # 保存模型：


    torch.save(model.state_dict(), '../model_weights/'+str(fold) + '_PlantNh-Kcr_kfold_model.pth'.format(fold))

    fold += 1

# np.save('../np_weights/PlantNh-Kcr_roc_auc.npy', roc_auc)
# np.save('../np_weights/PlantNh-Kcr_roc.npy', roc)

#五倍交叉验证 SN、SP、ACC、MCC
mean_SN=np.mean(total_SN)
mean_SP=np.mean(total_SP)
mean_ACC=np.mean(total_ACC)
mean_MCC=np.mean(total_MCC)


kfold_SN_SP_ACC_MCC=[]
kfold_SN_SP_ACC_MCC.append(mean_SN)
kfold_SN_SP_ACC_MCC.append(mean_SP)
kfold_SN_SP_ACC_MCC.append(mean_ACC)
kfold_SN_SP_ACC_MCC.append(mean_MCC)

#平均的五倍交叉验证结果：
# np.save('../np_weights/PlantNh-Kcr_5kfold_SN_SP_ACC_MCC.npy', kfold_SN_SP_ACC_MCC)
print("5kfold: SN is: {}, SP is: {}, ACC is: {},MCC is: {}".format(mean_SN,mean_SP,mean_ACC,mean_MCC))


# 五折交叉验证可视化：
def Kf_show(plt, base_fpr, roc, roc_auc):
    # 五折交叉验证图：
    for i, item in enumerate(roc):
        fpr, tpr = item
        plt.plot(fpr, tpr, label="ROC fold {} (AUC={:.4f})".format(i + 1, roc_auc[i]), lw=1, alpha=0.3)

    # 求平均值：mean
    plt.plot(base_fpr, np.average(tprs, axis=0),
             label=r'Mean ROC(AUC=%0.2f $\pm6$ %0.2f)' % (np.mean(roc_auc), np.std(roc_auc)),
             lw=1, alpha=0.8, color='b')
    # 基准线：
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, alpha=0.8, color='c')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    plt.legend(loc=4)
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.savefig('../figures/PlantNh-Kcr.png')
    plt.show()

Kf_show(plt, base_fpr, roc, roc_auc)

####################################

# 用全部的数据集进行训练：

import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import interp
import warnings

warnings.filterwarnings("ignore")

# 模型准备
epochs = 50
batch_size = 128
learn_rate = 0.001

# 使用低级api进行构建训练；
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


def total_train(model, train_loader, device):
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
            _, y_predict = model(x_data)
            # print("y_predict:",y_predict)

            # 二分问题使用的损失函数 binary_cross_entropy函数；
            loss = F.cross_entropy(y_predict, y_data)

            # loss=loss_fn(y_predict,y_data)

            acc = metrics.accuracy_score(y_data.detach().cpu().numpy(),
                                         torch.argmax(y_predict, dim=1).detach().cpu().numpy())
            # print("y_data:",y_data)
            auc = metrics.roc_auc_score(y_data.detach().cpu().numpy(), y_predict[:, 1].detach().cpu().numpy())

            epoch_loss.append(loss.detach().cpu().numpy())
            epoch_acc.append(acc)
            epoch_auc.append(auc)
            if (batch_id % 64 == 0):
                print("epoch is :{},batch_id is {},loss is {},acc is:{},auc is:{}".format(epoch, batch_id,
                                                                                          loss.detach().cpu().numpy(),
                                                                                          acc, auc))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # 取每次epoch的均值
        avg_loss, avg_acc, avg_auc = np.mean(epoch_loss), np.mean(epoch_acc), np.mean(epoch_auc)
        print("[train acc is:{}, loss is :{},auc is:{}]".format(avg_acc, avg_loss, avg_auc))

        if (epoch + 1) == epochs:
            # 保存模型：
            torch.save(model.state_dict(), '../model_weights/PlantNh-Kcr-FinalWeight.pth')


# 进行一次总的模型训练
from sklearn import metrics
# 定义DataLoader
from torch.utils.data import DataLoader

batch_size = 128

# 形成DataLoader:
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = KcrNet()
model.to(device)
# 训练model
total_train(model, train_loader, device)

