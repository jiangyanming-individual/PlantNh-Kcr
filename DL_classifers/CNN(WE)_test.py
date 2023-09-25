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
# train_set=MyDataset(train_dataset,word2id_dict)
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
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=False)

#实例化模型和加载模型

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = KcrNet(vocab_size=vocab_size,embedding_size=embedding_size,input_classes=29,nums_classes=2)
model.to(device)


#load mmodel
model_path='../DL_weights/CNN(WE)-FinalWeight.pth'
model.load_state_dict(torch.load(model_path,map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))



import math
from sklearn import metrics

test_roc = []
test_roc_auc = []

test_tprs = []
test_fprs = []

test_base_fpr = np.linspace(0, 1, 101)
test_base_fpr[-1] = 1.0


eval_SN_SP_ACC_MCC=[]
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

        y_test_pred = model(x_data)

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
    np.save('../np_weights/CNN(WE)_y_test_true.npy',y_test_true)
    np.save('../np_weights/CNN(WE)_y_test_score.npy',y_test_score)

    fpr, tpr, _ = metrics.roc_curve(y_test_true, y_test_score)

    auc = metrics.auc(fpr, tpr)

    test_roc.append([fpr, tpr])
    test_roc_auc.append(auc)

    tpr = interp(test_base_fpr, fpr, tpr)
    tpr[0] = 0.0
    test_tprs.append(tpr)
    test_fprs.append(fpr)

    print("[ind_test:avg_acc is:{},avg_loss is :{},auc is:{}]".format(avg_acc, avg_loss, auc))

    # eval_SN_SP_ACC_MCC=[]

    # 计算SN，SP，ACC，MCC
    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    MCC = ((TP * TN) - (FP * FN)) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    eval_SN_SP_ACC_MCC.append(SN)
    eval_SN_SP_ACC_MCC.append(SP)
    eval_SN_SP_ACC_MCC.append(ACC)
    eval_SN_SP_ACC_MCC.append(MCC)

    np.save('../np_weights/CNN(WE)_eval.npy',eval_SN_SP_ACC_MCC)

    print("ind_test TP is {},FP is {},TN is {},FN is {}".format(TP, FP, TN, FN))
    print("ind_test : SN is {},SP is {},ACC is {},MCC is {}".format(SN, SP, ACC, MCC))
