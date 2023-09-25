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



#构造词典：
AA_aaindex = 'ACDEFGHIKLMNPQRSTVWY'

word2id_dict={'X':0}
for i in range(len(AA_aaindex)):
    word2id_dict[AA_aaindex[i]]=i+1




#数据集的文件路径：
train_filepath= '../Datasets/train.csv'
test_filepath= '../Datasets/ind_test.csv'


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


class KcrNet(nn.Module):

    def __init__(self, vocab_size,embedding_size,input_classes=21, nums_classes=2):
        super(KcrNet, self).__init__()

        #定义Embedding层：
        self.vocab_size = vocab_size

        # embedding_size层大小：
        self.embedding_size = embedding_size

        self.embedding=nn.Embedding(
            vocab_size,embedding_size
        )

        # 定义卷积层：
        self.conv1 = torch.nn.Conv1d(in_channels=input_classes, out_channels=32, kernel_size=5, padding=2, stride=1)
        # 定义pooling层：
        # self.maxpool1=torch.nn.MaxPool1d(kernel_size=3,stride=1)

        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=2)
        # self.maxpool2=torch.nn.MaxPool1d(kernel_size=3,stride=1)

        self.conv3 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=2)
        # self.maxpool3=torch.nn.MaxPool1d(kernel_size=3,stride=1)
        # self.attention=QKV_SelfAttention()
        # 定义全连接层：
        self.flatten = torch.nn.Flatten()
        # 定义感知层；
        self.linear1 = torch.nn.Linear(in_features=32 * 8, out_features=128)
        self.linear2 = torch.nn.Linear(in_features=128, out_features=nums_classes)


        self.dropout1=torch.nn.Dropout(0.3)
        self.dropout2 = torch.nn.Dropout(0.3)

    def forward(self, x):

        inputs=self.embedding(x)
        # 1 Conv1D layer
        x = self.conv1(inputs)
        x = F.relu(x)
        # x=self.maxpool1(x)
        # print("x shape",x.shape)
        x = self.dropout1(x)

        # 2 Conv1D layer
        x = self.conv2(x)
        x = F.relu(x)
        # x=self.maxpool2(x)
        # print("x shape", x.shape)
        x = self.dropout1(x)

        # 3 Conv1D layer
        x = self.conv3(x)
        x = F.relu(x)
        # x=self.maxpool3(x)
        # print("x shape", x.shape)
        x = self.dropout2(x)

        # print("x shape:",x.shape)
        # 全连接层：
        x = self.flatten(x)

        x = self.linear1(x)
        x = F.relu(x)  # 激活函数
        x = self.linear2(x)
        return x






import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import interp
import warnings
warnings.filterwarnings("ignore")

# 模型准备
epochs = 30
batch_size = 128
learn_rate = 0.001



# 指定embedding的数量为词表长度
vocab_size = len(word2id_dict)

# embedding向量的维度
embedding_size = 29

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


model=KcrNet(vocab_size=vocab_size,embedding_size=embedding_size,input_classes=29,nums_classes=2)
print(model)


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

            data[0] = torch.stack(data[0], dim=1)
            # print("data[0]:",data[0])
            x_data = data[0].to(device)
            y_data = data[1].to(device)
            y_data = torch.tensor(y_data, dtype=torch.long)
            # print(x_data)

            # 进行模型训练：
            y_predict = model(x_data)
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
            data[0] = torch.stack(data[0], dim=1)
            # print("data[0]:",data[0])
            x_data = data[0].to(device)
            y_data = data[1].to(device)
            y_data = torch.tensor(y_data, dtype=torch.long)


            y_true_label=y_data

            y_predict = model(x_data)
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
        #
        np.save('../np_weights/CNN(WE)_y_train.npy',y_true)
        np.save('../np_weights/CNN(WE)_y_train_score.npy',y_score)

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
    model=KcrNet(vocab_size=vocab_size,embedding_size=embedding_size,input_classes=29,nums_classes=2)
    model.to(device)

    train(model,train_loader,valid_loader,device)


    # 保存模型：
    torch.save(model.state_dict(), '../DL_weights/'+str(fold) + '_CNN(WE)_kfold_model.pth'.format(fold))

    fold += 1


# save tpr,fpr,auc
np.save('../np_weights/CNN(WE)_roc_auc.npy', roc_auc)
np.save('../np_weights/CNN(WE)_roc.npy', roc)


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
np.save('../np_weights/CNN(WE)_5kfold_SN_SP_ACC_MCC.npy', kfold_SN_SP_ACC_MCC)
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
    plt.savefig('../figures/CNN(WE).png')
    plt.show()

Kf_show(plt, base_fpr, roc, roc_auc)

####################################

# total dataset to training
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import interp
import warnings

warnings.filterwarnings("ignore")

# 模型准备
epochs = 30
batch_size = 128
learn_rate = 0.001

# 指定embedding的数量为词表长度
vocab_size = len(word2id_dict)

# embedding向量的维度
embedding_size = 29

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

model = KcrNet(vocab_size=vocab_size, embedding_size=embedding_size, input_classes=29, nums_classes=2)
print(model)


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

            data[0] = torch.stack(data[0], dim=1)
            # print("data[0]:",data[0])
            x_data = data[0].to(device)
            y_data = data[1].to(device)
            y_data = torch.tensor(y_data, dtype=torch.long)
            # print(x_data)

            # 进行模型训练：
            y_predict = model(x_data)
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
            torch.save(model.state_dict(), '../DL_weights/CNN(WE)-FinalWeight.pth')


# 进行一次总的模型训练
from sklearn import metrics
# 定义DataLoader
from torch.utils.data import DataLoader

batch_size = 128
# 形成DataLoader:
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = KcrNet(vocab_size=vocab_size, embedding_size=embedding_size, input_classes=29, nums_classes=2)
model.to(device)
# 训练model
total_train(model, train_loader, device)

