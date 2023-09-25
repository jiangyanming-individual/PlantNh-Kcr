# author:Lenovo
# datetime:2023/7/11 8:52
# software: PyCharm
# project:pytorch项目


import os, re, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from numpy import interp


"""
model
"""
class CNN_Feature2d(nn.Module):
    def __init__(self, device, out_channels, conv_kernel_size=5, pool_kernel_size=2, dense_size=64, dropout=0.5):
        super(CNN_Feature2d, self).__init__()
        self.device = device
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=conv_kernel_size, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=conv_kernel_size, stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=conv_kernel_size, stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size)
        )
        self.dropout3 = nn.Dropout(dropout)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=2 * out_channels, kernel_size=conv_kernel_size, stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size)
        )
        self.dropout4 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(9 * out_channels, dense_size)
        self.fc2 = nn.Linear(dense_size, 1)

    def forward(self, X, is_train=False):
        out = X.view(-1, 3, 29, 29) #reshape
        out = self.conv1(out)
        # print('Conv1: ', out.size())
        if is_train:
            out = self.dropout1(out)
        out = self.conv2(out)
        # print('Conv2: ', out.size())
        if is_train:
            out = self.dropout2(out)
        out = self.conv3(out)
        # print('Conv3: ', out.size())
        if is_train:
            out = self.dropout3(out)
        out = out.view(out.size(0), -1)

        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out

    def predict(self, loader):
        score = []
        for batch_X, batch_y in loader:
            Xb = batch_X.to(self.device)
            yb = batch_y.to(self.device)
            y_ = self.forward(Xb)
            score.append(y_.cpu().data)
        return torch.cat(score).numpy()[:, 0]


"""
generate Datasets
"""
class DealDataset(Dataset):
    def __init__(self, np_data):
        self.__np_data = np_data
        self.X = torch.from_numpy(self.__np_data[:, 1:])
        self.y = torch.from_numpy(self.__np_data[:, 0]).view(-1, 1)
        self.len = self.__np_data.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


"""
encoding
"""
def encoding(samples):

        with open('../features_encode/AAindex/AAindex_normalized.txt') as f:
            records = f.readlines()[1:]

        AAindex = []
        AAindexName = []
        for i in records:
            AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
            AAindexName.append(i.rstrip().split()[0] if i.rstrip() != '' else None)

        # 前29种物理化学性质：
        props = 'FINA910104:LEVM760101:JACR890101:ZIMJ680104:RADA880108:JANJ780101:CHOC760102:NADH010102:KYTJ820101:NAKH900110:GUYH850101:EISD860102:HUTJ700103:OLSK800101:JURD980101:FAUJ830101:OOBM770101:GARJ730101:ROSM880102:RICJ880113:KIDA850101:KLEP840101:FASG760103:WILM950103:WOLS870103:COWR900101:KRIW790101:AURR980116:NAKH920108'.split(
            ':')
        if props:
            tmpIndexNames = []
            tmpIndex = []
            for p in props:
                if AAindexName.index(p) != -1:
                    tmpIndexNames.append(p)
                    tmpIndex.append(AAindex[AAindexName.index(p)])
            if len(tmpIndexNames) != 0:
                AAindexName = tmpIndexNames
                AAindex = tmpIndex


        """
        create dict
        """
        AA_aaindex = 'ARNDCQEGHILKMFPSTWYVX'
        index = {}
        for i in range(len(AA_aaindex)):
            index[AA_aaindex[i]] = i

        blosum62 = {
            'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],  # A
            'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],  # R
            'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],  # N
            'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],  # D
            'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],  # C
            'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],  # Q
            'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],  # E
            'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],  # G
            'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],  # H
            'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],  # I
            'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],  # L
            'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],  # K
            'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],  # M
            'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],  # F
            'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],  # P
            'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],  # S
            'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],  # T
            'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],  # W
            'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],  # Y
            'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],  # V
            'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -X
        }


        for key in blosum62:
            for j, value in enumerate(blosum62[key]):
                blosum62[key][j] = round((value + 4) / 15, 3)  # 四舍五入保留三位小数；

        encoding_aaindex = []


        #AAindex encoding
        for i in samples:
            sequence, label = i[0], i[1]
            code = [int(label)]
            for aa in sequence:
                if aa == 'X':
                    for j in AAindex:
                        code.append(0)
                    continue
                for j in AAindex:
                    code.append(j[index[aa]])
            encoding_aaindex.append(code)
        # print(encoding_aaindex)



        # BLOSUM62 encoding
        encoding_blosum = []
        for i in samples:
            sequence, label = i[0], i[1]
            code = [int(label)]
            for aa in sequence:
                code = code + blosum62[aa]
                code += [0.267] * 9  # 0.267是它的平均值；
            encoding_blosum.append(code)
        # print(encoding_blosum)


        #binary encoding
        AA = 'ARNDCQEGHILKMFPSTWYV'
        encoding_binary = []
        for i in samples:
            sequence, label = i[0], i[1]

            # print(len(sequence))
            # print(len(label))
            # print(i)
            code = [int(label)]
            # print("code:",code)
            for aa in sequence:
                if aa == 'X':
                    code = code + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    code += [0.05] * 9
                    continue
                for aa1 in AA:
                    tag = 1 if aa == aa1 else 0
                    code.append(tag)
                code += [0.05] * 9

            # print(len(code))
            encoding_binary.append(code)
        # print(encoding_binary)

        encoding_aaindex = np.array(encoding_aaindex)
        # print(encoding_aaindex.shape)
        encoding_blosum = np.array(encoding_blosum)
        # print(encoding_blosum.shape)
        encoding_binary = np.array(encoding_binary)
        # print(encoding_binary.shape)

        encodings = np.hstack((encoding_aaindex[:, 0:], encoding_binary[:, 1:], encoding_blosum[:, 1:]))
        return True, encodings.astype(np.float32) #(batch_size, 2524)第一维度是label

    # except Exception as e:
    #     return False, None



# read file
def read_file(fileapth):


    data_list=[]
    with open(file=fileapth,mode='r',encoding='utf-8') as f:

        for line in f.readlines():
            # print(line.strip())
            data_list.append(line.strip().split(','))

    return data_list

#独立测试：


train_filepath = '../Datasets/train.csv'
test_filepath = '../Datasets/ind_test.csv'

data_list = read_file(test_filepath)
ok_encoding, data_set = encoding(data_list)
print(data_set)
print(data_set.shape)

# 独立测试

from sklearn.metrics import confusion_matrix
import math
from sklearn import metrics
from numpy import interp

if __name__ == '__main__':

    batch_size = 128
    learn_rate = 0.001

    base_fpr = np.linspace(0, 1, 101)
    base_fpr[-1] = 1.0

    test_set = DealDataset(data_set.astype(np.float32))

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=False)

    # 实例化模型和加载模型
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    model = CNN_Feature2d(device, 128, 5, 2, 64, 0.5)
    model.to(device)

    # load mmodel :需要修改
    model_path = '../nhkcr/models/Merged_5.pkl'

    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu")))

    test_roc = []
    test_roc_auc = []

    test_tprs = []
    test_fprs = []

    test_base_fpr = np.linspace(0, 1, 101)
    test_base_fpr[-1] = 1.0

    eval_SN_SP_ACC_MCC = []
    # 独立数据集测试：
    model.eval()
    with torch.no_grad():

        y_test_true = []
        y_test_score = []
        y_pred_label = []

        for batch_id, data in enumerate(test_loader):
            x_data = data[0].to(device)
            y_data = data[1].to(device)

            # print("x_data:",x_data)
            y_true_label = y_data
            # y_data = torch.unsqueeze(y_data, axis=-1)
            y_test_pred = model(x_data)
            #print(y_test_pred)

            # 计算得分
            y_test_true.append(y_data[:].detach().cpu().numpy())
            y_test_score.append(y_test_pred[:].detach().cpu().numpy())

        # 合并数据：
        y_test_true = np.concatenate(y_test_true)
        y_test_score = np.concatenate(y_test_score)

        # 保存真实值和预测值的值
        np.save('../np_weights/nhKcr_y_test_true.npy', y_test_true)
        np.save('../np_weights/nhKcr_y_test_score.npy', y_test_score)

        fpr, tpr, _ = metrics.roc_curve(y_test_true, y_test_score)
        final_auc = metrics.roc_auc_score(y_test_true,y_test_score )
        print("final_auc:",final_auc)

        #混淆矩阵：
        y_test_score = np.round(y_test_score)
        conf_metrix = confusion_matrix(y_true=y_test_true, y_pred=y_test_score)
        print("conf_metrix:", conf_metrix)


        TN = conf_metrix[0][0]
        FP = conf_metrix[0][1]
        FN = conf_metrix[1][0]
        TP = conf_metrix[1][1]

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



        print("ind_test TP is {},FP is {},TN is {},FN is {}".format(TP, FP, TN, FN))
        print("ind_test : SN is {},SP is {},ACC is {},MCC is {},AUC is {}".format(SN, SP, ACC, MCC, final_auc))


