import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import warnings
import math
from numpy import interp

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,Subset



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


# 模型训练：

warnings.filterwarnings("ignore")
# 指定训练轮次
num_epochs = 30
# 指定学习率
learning_rate = 0.001


input_size=len(Amino_acid_sequence)
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


def train(model, train_loader,valid_loader,device):
    print("train is start...")
    # 优化器：
    optimizer = torch.optim.Adam(params=model.parameters(),lr=learning_rate)
    # 指定损失函数

    # loss_fn = FocalLoss(alpha=0.83, gamma=2)
    # 模型训练：
    model.train()
    for epoch in range(num_epochs):

        epoch_loss = []
        epoch_acc = []
        epoch_auc = []
        # print("第{}个epoch:".format(epoch))
        for batch_id, data in enumerate(train_loader):


            x_data = data[0].to(device)
            # print("x_data:",x_data)
            y_data = data[1].to(device)
            # print("y_data:",y_data)
            y_data = torch.tensor(y_data, dtype=torch.long)

            # print("data[0][0]",data[0][0])
            # print("data[0][0]",data[0][1])
            # print("label:",data[1])

            _, y_predict = model(x_data)

            # print("y_predict:",y_predict)
            # print("y_predict shape:",y_predict.shape)
            loss = F.cross_entropy(y_predict, y_data)

            # loss = loss_fn(y_predict, y_data)
            # print("loss:",loss)
            # 指定评估函数：
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

    # 交叉验证模型只验证一次
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
            # print(data[0])

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

            # 计算损失值：
            loss = F.cross_entropy(y_valid_pred, y_data)

            # loss = loss_fn(y_valid_pred, y_data)
            # acc = torch.metric.accuracy(y_valid_pred, y_data)
            acc=metrics.accuracy_score(y_data.detach().cpu().numpy(),torch.argmax(y_valid_pred,dim=1).detach().cpu().numpy())

            #cal auc
            auc = metrics.roc_auc_score(y_data[:].detach().cpu().numpy(), y_valid_pred[:, 1].detach().cpu().numpy())


            # 计算得分
            y_true.append(y_data[:].detach().cpu().numpy())
            y_score.append(y_valid_pred[:, 1].detach().cpu().numpy())

            valid_acc.append(acc)
            valid_loss.append(loss.detach().cpu().numpy())
            valid_auc.append(auc)

            TP += ((y_true_label == 1) & (y_predict_label == 1)).sum().item()
            FP += ((y_true_label == 1) & (y_predict_label == 0)).sum().item()
            TN += ((y_true_label == 0) & (y_predict_label == 0)).sum().item()
            FN += ((y_true_label == 0) & (y_predict_label == 1)).sum().item()
            # valid_auc.append(auc)

            if batch_id % 64 == 0:
                print("batch_id is {} ,acc is {} ,loss is {} ,auc is {}".format(batch_id, acc, loss.detach().cpu().numpy(),auc))

        avg_acc, avg_loss, avg_auc = np.mean(valid_acc), np.mean(valid_loss), np.mean(valid_auc)

        # 合并数据：
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


        np.save('../np_weights/BiLSTM(BE)_y_train.npy',y_true)
        np.save('../np_weights/BiLSTM(BE)_y_train_score.npy',y_score)
        # 计算SN，SP，ACC，MCC

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

    # 实例化总的模型：
    model = Model_LSTM(input_size, hidden_size, num_classes, num_layers)
    model.to(device)
    train(model, train_loader,valid_loader,device)


    # save tpr,fpr,auc

    # 保存模型：
    torch.save(model.state_dict(), '../DL_weights/'+str(fold) + '_BiLSTM(BE)_kfold_model.pth'.format(fold))

    fold += 1


np.save('../np_weights/BiLSTM(BE)_roc_auc.npy', roc_auc)
np.save('../np_weights/BiLSTM(BE)_roc.npy', roc)


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
np.save('../np_weights/BiLSTM(BE)_SN_SP_ACC_MCC.npy', kfold_SN_SP_ACC_MCC)
print("5kfold: SN is: {}, SP is: {}, ACC is: {},MCC is: {}".format(mean_SN,mean_SP,mean_ACC,mean_MCC))



# 五折交叉验证可视化：

def Kf_show(plt, base_fpr, roc, roc_auc):
    # 五折交叉验证图：
    for i, item in enumerate(roc):
        fpr, tpr = item
        plt.plot(fpr, tpr, label="ROC fold {} (AUC={:.4f})".format(i + 1, roc_auc[i]), lw=1, alpha=0.3)

    # 求平均值：mean
    plt.plot(base_fpr, np.average(tprs, axis=0),
             label=r'Mean ROC(AUC=%0.2f $\pm$ %0.2f)' % (np.mean(roc_auc), np.std(roc_auc)),
             lw=1, alpha=0.8, color='b')
    # 基准线：
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, alpha=0.8, color='c')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    plt.legend(loc=4)
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.savefig('../figures/BiLSTM(BE).png')
    plt.show()

Kf_show(plt, base_fpr, roc, roc_auc)


##########################################

# total  dataset to training
# 模型训练：

warnings.filterwarnings("ignore")
# 指定训练轮次
num_epochs = 30
# 指定学习率
learning_rate = 0.001
batch_size = 128

input_size = len(Amino_acid_sequence)
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


def total_train(model, train_loader, device):
    print("train is start...")
    # 优化器：
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    # 指定损失函数

    # loss_fn = FocalLoss(alpha=0.83, gamma=2)
    # 模型训练：
    model.train()
    for epoch in range(num_epochs):

        epoch_loss = []
        epoch_acc = []
        epoch_auc = []
        # print("第{}个epoch:".format(epoch))
        for batch_id, data in enumerate(train_loader):

            x_data = data[0].to(device)
            # print("x_data:",x_data)
            y_data = data[1].to(device)
            # print("y_data:",y_data)
            y_data = torch.tensor(y_data, dtype=torch.long)

            # print("data[0][0]",data[0][0])
            # print("data[0][0]",data[0][1])
            # print("label:",data[1])

            _, y_predict = model(x_data)

            # print("y_predict:",y_predict)
            # print("y_predict shape:",y_predict.shape)
            loss = F.cross_entropy(y_predict, y_data)

            # loss = loss_fn(y_predict, y_data)
            # print("loss:",loss)
            # 指定评估函数：
            # acc = torch.metric.accuracy(y_predict, y_data)
            acc = metrics.accuracy_score(y_data.detach().cpu().numpy(),
                                         torch.argmax(y_predict, dim=1).detach().cpu().numpy())
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
                print("epoch is {},batch_id is {},acc is :{} ,loss is {},auc is {}".format(epoch, batch_id, acc,
                                                                                           loss.detach().cpu().numpy(),
                                                                                           auc))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_acc, avg_loss, avg_auc = np.mean(epoch_acc), np.mean(epoch_loss), np.mean(epoch_auc)
        print("[train:avg_acc is: {},avg_loss is: {},avg_auc is: {}]".format(avg_acc, avg_loss, avg_auc))

        if (epoch + 1) == num_epochs:
            # 保存模型：
            torch.save(model.state_dict(), '../DL_weights/BiLSTM(BE)-FinalWeight.pth')


# 进行一次总的模型训练
from sklearn import metrics
# 定义DataLoader
from torch.utils.data import DataLoader

# 形成DataLoader:
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 实例化总的模型：
model = Model_LSTM(input_size, hidden_size, num_classes, num_layers)
model.to(device)

# 训练model
total_train(model, train_loader, device)



