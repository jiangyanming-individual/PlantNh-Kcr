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


# 对数据进行二进制编码：
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
        # print(i.rstrip().split()[0])  #得到AAindex的names
        AAindex_names.append(i.rstrip().split()[0] if i.rstrip() != '' else None)
        AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)

    props = 'FINA910104:LEVM760101:JACR890101:ZIMJ680104:RADA880108:JANJ780101:CHOC760102:NADH010102:KYTJ820101:NAKH900110:GUYH850101:EISD860102:HUTJ700103:OLSK800101:JURD980101:FAUJ830101:OOBM770101:GARJ730101:ROSM880102:RICJ880113:KIDA850101:KLEP840101:FASG760103:WILM950103:WOLS870103:COWR900101:KRIW790101:AURR980116:NAKH920108'.split(':')

    if props:
        tempAAindex_names = []
        tempAAindex = []

        for p in props:
            # 如果29种的一种存在
            if AAindex_names.index(p) != -1:
                tempAAindex_names.append(p)
                tempAAindex.append(AAindex[AAindex_names.index(p)])

        # 如果找到了，就将前29种的性质直接替代AAindx；
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
                for aaindex in AAindex:  # 为X 全部赋值为0
                    one_code.append(0)
                continue
            for aaindex in AAindex:
                # print(type(aaindex[seq_index.get(aa)]))
                one_code.append(aaindex[seq_index.get(aa)])  # 添加存在的aaindex;
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

# data=read_file(train_filepath)
# train_dataset=get_AAindex_encode(data)

data=read_file(test_filepath)
test_dataset=get_AAindex_encode(data)


# 构建数据集：
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


# 形成数据集：tuple
# train_set = MyDataset(train_dataset[0], train_dataset[1])
# print(train_set)

test_set = MyDataset(test_dataset[0], test_dataset[1])

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
        self.cls_layer = nn.Linear(self.hidden_size * 2 ,self.num_classes)
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
# 指定embedding的数量为词表长度

input_size=29

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
model=Model_LSTM(input_size,hidden_size,num_classes,num_layers)
model.to(device)


#load mmodel
model_path="../DL_weights/BiLSTM(AAindex)-FinalWeight.pth"
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
    np.save('../np_weights/BiLSTM(AAindex)_y_test_true.npy',y_test_true)
    np.save('../np_weights/BiLSTM(AAindex)_y_test_score.npy',y_test_score)

    fpr, tpr, _ = metrics.roc_curve(y_test_true, y_test_score)

    auc = metrics.auc(fpr, tpr)

    test_roc.append([fpr, tpr])
    test_roc_auc.append(auc)

    tpr = interp(test_base_fpr, fpr, tpr)
    tpr[0] = 0.0
    test_tprs.append(tpr)
    test_fprs.append(fpr)

    print("ind_test auc:",auc)
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

    np.save('../np_weights/BiLSTM(AAindex)_eval.npy',eval_SN_SP_ACC_MCC)

    print("ind_test TP is {},FP is {},TN is {},FN is {}".format(TP, FP, TN, FN))
    print("ind_test : SN is {},SP is {},ACC is {},MCC is {}".format(SN, SP, ACC, MCC))